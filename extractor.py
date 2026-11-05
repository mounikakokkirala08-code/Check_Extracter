"""
extractor.py
------------
Core extraction module for scanned/photographed bank cheques using Vision-Language
Models (VLMs). Supports both Groq (100% Free Tier, Qwen 3.8 27B Vision) and Google Gemini.

Key Features:
- 100% in-memory image processing (no images saved to disk).
- Strict JSON structured output enforced via Pydantic response schema.
- Automatic retry on malformed JSON / schema validation failures.
- Robust handling of thinking/reasoning models (cleans <think> tags).
- Pure-Python amount-in-words to numeric parser for mismatch detection.
- Clean error classification (429 Rate Limit, Invalid API Key, API failures).
- In-memory dummy/sample cheque generator for learning and quick testing.
"""

import os
import re
import io
import json
import base64
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, date
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# ==============================================================================
# Custom Exceptions for Clean Error Handling
# ==============================================================================
class RateLimitError(Exception):
    """Raised when the API returns HTTP 429 (quota or rate limit reached)."""
    pass

class AuthenticationError(Exception):
    """Raised when the API key is missing, invalid, or unauthorized."""
    pass

class ExtractionError(Exception):
    """Raised when extraction fails due to API errors or malformed response."""
    pass


# ==============================================================================
# Pydantic Response Schema
# ==============================================================================
class ChequeDetails(BaseModel):
    """
    Structured fields extracted from a bank cheque image.
    Every field defaults to None (null in JSON) if unreadable or not present.
    """
    payee_name: Optional[str] = Field(
        default=None,
        description="Name of the person or entity to whom the cheque is payable. Null if unreadable."
    )
    payer_name: Optional[str] = Field(
        default=None,
        description="Name of the account holder / drawer / issuer of the cheque. Null if unreadable."
    )
    payee_account_number: Optional[str] = Field(
        default=None,
        description="Payee account number, if explicitly specified on the cheque. Null if not present."
    )
    payer_account_number: Optional[str] = Field(
        default=None,
        description="Printed or written bank account number of the payer / drawer. Null if unreadable."
    )
    amount_numeric: Optional[float] = Field(
        default=None,
        description="Cheque amount in figures/digits (e.g. 12500.00). Exclude currency symbols. Null if unreadable."
    )
    amount_words: Optional[str] = Field(
        default=None,
        description="Cheque amount written out in words. Null if unreadable."
    )
    transaction_date: Optional[str] = Field(
        default=None,
        description="Cheque date in YYYY-MM-DD format. Null if unreadable."
    )
    cheque_number: Optional[str] = Field(
        default=None,
        description="6-digit cheque number, typically from the bottom MICR band or top corner. Null if unreadable."
    )
    bank_name: Optional[str] = Field(
        default=None,
        description="Name of the issuing bank. Null if unreadable."
    )
    ifsc_or_micr_code: Optional[str] = Field(
        default=None,
        description="Bank branch IFSC code (e.g., SBIN0001234) or MICR code. Null if unreadable."
    )
    signature_present: Optional[bool] = Field(
        default=None,
        description="True if a signature or drawer sign is visibly present in the signature area, False otherwise."
    )


# ==============================================================================
# VLM Base Prompt
# ==============================================================================
BASE_PROMPT = """You are an expert bank cheque processing assistant. Read handwriting, stamps and printed text. Return null for anything you cannot read. Do not hallucinate.

Carefully inspect the provided cheque image and extract all visible details according to the schema:
- payee_name: Full name of the payee (the party to whom money is to be paid).
- payer_name: Name of the account holder / drawer / issuer.
- payee_account_number: Payee's account number, if written (e.g. in 'A/c Payee' crossing).
- payer_account_number: Payer's bank account number.
- amount_numeric: Numerical amount as a float (e.g. 12500.00). Do not include currency symbols ($ or Rs or €).
- amount_words: The full amount string written in words.
- transaction_date: Date written on the cheque, normalized to YYYY-MM-DD format.
- cheque_number: 6-digit cheque serial number (usually first number enclosed in quotes/delimiters in MICR strip).
- bank_name: Issuing bank name.
- ifsc_or_micr_code: IFSC code or 9-digit MICR code if legible.
- signature_present: true if a handwritten signature is visible in the signature area, false if unsigned.

Remember: If any field cannot be read with confidence or is absent, return null. Never guess."""


# ==============================================================================
# Helper to Clean Model Output (Removes <think> tags & Markdown wrappers)
# ==============================================================================
def clean_json_response(raw_text: str) -> str:
    """
    Cleans raw response from reasoning models (e.g., Qwen thinking tokens)
    and strips markdown code fences so Pydantic can parse pure JSON.
    """
    if not raw_text:
        return ""

    # 1. Remove reasoning / thought blocks <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    # 2. Extract content from ```json ... ``` code fence if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # 3. Find outer curly braces { ... }
    brace_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
    if brace_match:
        return brace_match.group(1).strip()

    return cleaned


# ==============================================================================
# Pure-Python Number Words Parser & Amount Match Check
# ==============================================================================
def words_to_number(text: str) -> Optional[float]:
    """
    Parses a written currency string (e.g., 'Twelve Thousand Five Hundred Only')
    into a numeric float value (12500.0) without requiring external heavy libraries.
    """
    if not text or not isinstance(text, str):
        return None

    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())

    noise_words = {
        "only", "rupees", "rupee", "rs", "inr", "dollars", "dollar",
        "usd", "sum", "of", "and", "exact", "cents", "cent", "paise", "paisa"
    }

    tokens = [t for t in clean.split() if t not in noise_words]
    if not tokens:
        return None

    units = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19
    }

    tens = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }

    multipliers = {
        "hundred": 100,
        "thousand": 1_000,
        "lakh": 100_000,
        "lakhs": 100_000,
        "lac": 100_000,
        "lacs": 100_000,
        "million": 1_000_000,
        "crore": 10_000_000,
        "crores": 10_000_000,
        "billion": 1_000_000_000
    }

    current = 0.0
    total = 0.0
    recognized_any = False

    for token in tokens:
        if token.isdigit():
            current += float(token)
            recognized_any = True
        elif token in units:
            current += units[token]
            recognized_any = True
        elif token in tens:
            current += tens[token]
            recognized_any = True
        elif token == "hundred":
            current = (current if current != 0 else 1.0) * 100
            recognized_any = True
        elif token in multipliers:
            scale = multipliers[token]
            total += (current if current != 0 else 1.0) * scale
            current = 0.0
            recognized_any = True

    total += current
    return total if recognized_any else None


def verify_amount_match(amount_numeric: Optional[float], amount_words: Optional[str]) -> Dict[str, Any]:
    """
    Checks whether amount_numeric and amount_words represent the same value.
    Flags discrepancies between the numeric figure and the written text.
    """
    if amount_numeric is None and amount_words is None:
        return {
            "status": "INCONCLUSIVE",
            "is_match": None,
            "message": "Neither numeric amount nor amount in words was detected.",
            "parsed_words_value": None
        }

    if amount_numeric is None:
        return {
            "status": "INCONCLUSIVE",
            "is_match": None,
            "message": "Numeric amount is missing/unreadable. Cannot verify match.",
            "parsed_words_value": None
        }

    if not amount_words:
        return {
            "status": "INCONCLUSIVE",
            "is_match": None,
            "message": "Amount in words is missing/unreadable. Cannot verify match.",
            "parsed_words_value": None
        }

    parsed_value = words_to_number(amount_words)

    if parsed_value is None:
        return {
            "status": "INCONCLUSIVE",
            "is_match": None,
            "message": f"Could not parse words '{amount_words}' into numbers for automated comparison.",
            "parsed_words_value": None
        }

    is_match = abs(parsed_value - amount_numeric) < 0.01

    if is_match:
        return {
            "status": "MATCH",
            "is_match": True,
            "message": f"Amounts match: Numeric ({amount_numeric:,.2f}) matches words ('{amount_words}').",
            "parsed_words_value": parsed_value
        }
    else:
        return {
            "status": "MISMATCH",
            "is_match": False,
            "message": f"Discrepancy detected! Numeric amount is {amount_numeric:,.2f}, but words state '{amount_words}' (parsed as {parsed_value:,.2f}).",
            "parsed_words_value": parsed_value
        }


# ==============================================================================
# Cheque Expiry & Stale Date Validator (Banking 90-Day Rule)
# ==============================================================================
def verify_cheque_date(date_str: Optional[str]) -> Dict[str, Any]:
    """
    Validates the cheque transaction date according to standard banking regulations:
    - Standard validity window: 90 days (approx. 3 months) from issuance date.
    - Post-dated cheque (PDC): Date is in the future. Cannot be deposited early.
    - Stale / Expired: Issued > 90 days ago. Requires drawer re-validation.
    
    Returns:
        dict: {
            "status": "VALID" | "STALE" | "POST_DATED" | "INCONCLUSIVE",
            "days": Optional[int],
            "message": str,
            "issue_date": Optional[str]
        }
    """
    if not date_str or not isinstance(date_str, str):
        return {
            "status": "INCONCLUSIVE",
            "days": None,
            "message": "Transaction date is missing or unreadable.",
            "issue_date": None
        }

    cleaned = date_str.strip()
    dt = None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d.%m.%Y"):
        try:
            dt = datetime.strptime(cleaned, fmt).date()
            break
        except ValueError:
            continue

    if dt is None:
        return {
            "status": "INCONCLUSIVE",
            "days": None,
            "message": f"Could not parse date format: '{date_str}'. Expected YYYY-MM-DD.",
            "issue_date": None
        }

    today = date.today()
    delta_days = (today - dt).days

    if delta_days < 0:
        return {
            "status": "POST_DATED",
            "days": delta_days,
            "message": f"Post-Dated Cheque (PDC): Dated {abs(delta_days)} days in the future ({dt}). Cannot be cleared before this date.",
            "issue_date": str(dt)
        }
    elif delta_days > 90:
        return {
            "status": "STALE",
            "days": delta_days,
            "message": f"Stale / Expired Cheque: Issued {delta_days} days ago ({dt}). Exceeds the standard 90-day banking validity window.",
            "issue_date": str(dt)
        }
    else:
        remaining = 90 - delta_days
        return {
            "status": "VALID",
            "days": delta_days,
            "message": f"Valid Cheque Date: Issued {delta_days} days ago ({dt}). Valid for {remaining} more days.",
            "issue_date": str(dt)
        }


# ==============================================================================
# In-Memory Image Preprocessor (Rotate, Contrast, Sharpness)
# ==============================================================================
def preprocess_image(
    image: Image.Image,
    rotation: int = 0,
    contrast: float = 1.0,
    sharpness: float = 1.0
) -> Image.Image:
    """
    Applies non-destructive, in-memory enhancements to improve VLM optical readability
    without writing any temporary files to disk.
    
    Args:
        image: Original PIL Image.
        rotation: Angle in degrees (0, 90, 180, 270).
        contrast: Contrast multiplier (1.0 = neutral, >1.0 increases contrast).
        sharpness: Sharpness multiplier (1.0 = neutral, >1.0 sharpens edges).
        
    Returns:
        Enhanced in-memory PIL Image.
    """
    processed = image.copy()

    # 1. Rotation (clockwise / counter-clockwise)
    if rotation % 360 != 0:
        processed = processed.rotate(-rotation, expand=True)

    # 2. Contrast adjustment
    if abs(contrast - 1.0) > 0.01:
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(contrast)

    # 3. Sharpness adjustment
    if abs(sharpness - 1.0) > 0.01:
        enhancer = ImageEnhance.Sharpness(processed)
        processed = enhancer.enhance(sharpness)

    return processed


# ==============================================================================
# Groq Vision Engine (100% Free, Qwen 3.8 27B Vision)
# ==============================================================================
def _extract_with_groq(image: Image.Image, api_key: str, model_name: str) -> ChequeDetails:
    """Invokes Groq Vision-Language Model in memory."""
    try:
        from groq import Groq, APIError, RateLimitError as GroqRateLimitError, AuthenticationError as GroqAuthError
    except ImportError as e:
        raise ExtractionError(f"Required package 'groq' is not installed: {e}")

    # Convert in-memory image to JPEG base64 Data URI
    buffered = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=95)
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_image}"

    client = Groq(api_key=api_key.strip())

    # Map decommissioned model to active vision model
    model = model_name or "qwen/qwen3.8-27b"
    if "llama-3.2" in model or "decommissioned" in model:
        model = "qwen/qwen3.8-27b"

    schema_json = json.dumps(ChequeDetails.model_json_schema(), indent=2)
    prompt_with_schema = (
        f"{BASE_PROMPT}\n\n"
        "Return strictly a JSON object conforming to this schema:\n"
        f"{schema_json}\n\n"
        "Ensure all keys are present. Use null for unreadable or missing values."
    )

    def _call(p: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": p},
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw = response.choices[0].message.content
            return clean_json_response(raw)
        except GroqRateLimitError as rle:
            raise RateLimitError("Groq API rate limit reached (429). Please wait a few seconds and try again.") from rle
        except GroqAuthError as gae:
            raise AuthenticationError("Invalid Groq API key. Please check your GROQ_API_KEY in .env or the sidebar.") from gae
        except APIError as ae:
            err_str = str(ae)
            if "429" in err_str or "rate_limit_exceeded" in err_str:
                raise RateLimitError("Groq API rate limit reached (429). Please wait a moment.") from ae
            if "401" in err_str or "invalid_api_key" in err_str:
                raise AuthenticationError("Invalid Groq API key.") from ae
            raise ExtractionError(f"Groq API error: {err_str}") from ae
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                raise RateLimitError("Groq rate limit reached. Please wait a moment.") from e
            raise ExtractionError(f"Failed to communicate with Groq API: {err_str}") from e

    # First attempt
    content = _call(prompt_with_schema)
    try:
        return ChequeDetails.model_validate_json(content)
    except (json.JSONDecodeError, ValidationError, ValueError):
        # Retry once if JSON is malformed
        retry_prompt = prompt_with_schema + "\n\nCRITICAL: Previous response was malformed. Output ONLY valid JSON."
        content_retry = _call(retry_prompt)
        return ChequeDetails.model_validate_json(content_retry)


# ==============================================================================
# Google Gemini Vision Engine
# ==============================================================================
def _extract_with_gemini(image: Image.Image, api_key: str, model_name: str) -> ChequeDetails:
    """Invokes Google Gemini Vision-Language Model in memory."""
    try:
        from google import genai
        from google.genai import types
        from google.genai import errors
    except ImportError as e:
        raise ExtractionError(f"Required package 'google-genai' is not installed: {e}")

    client = genai.Client(api_key=api_key.strip())
    model = model_name or "gemini-2.5-flash"

    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ChequeDetails,
        temperature=0.1,
    )

    def _call(p: str) -> str:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[image, p],
                config=generation_config
            )
            return response.text
        except errors.APIError as api_err:
            err_code = getattr(api_err, "code", None)
            err_str = str(api_err)

            if err_code == 429 or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                raise RateLimitError(
                    "Google Gemini API rate limit reached (HTTP 429). "
                    "Please wait 15–30 seconds before retrying."
                ) from api_err

            if "CONSUMER_SUSPENDED" in err_str:
                raise AuthenticationError(
                    "Google Cloud project suspended (403 CONSUMER_SUSPENDED). "
                    "The Google Cloud project linked to this key has been suspended. "
                    "Switch to Groq in the sidebar or create a new key at https://aistudio.google.com/app/apikey"
                ) from api_err

            if "API_KEY_INVALID" in err_str or "API KEY NOT VALID" in err_str or (err_code == 400 and "API_KEY" in err_str):
                raise AuthenticationError(
                    "Invalid Gemini API key. Please check your GEMINI_API_KEY in .env or the sidebar."
                ) from api_err

            if "PERMISSION_DENIED" in err_str or err_code == 403:
                raise AuthenticationError(
                    "Permission Denied (403). Your API key does not have access or its Google Cloud project is disabled."
                ) from api_err

            raise ExtractionError(f"Gemini API error ({err_code}): {err_str}") from api_err
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                raise RateLimitError("Gemini API rate limit reached. Please wait a moment.") from e
            raise ExtractionError(f"Failed to communicate with Gemini API: {err_str}") from e

    # First attempt
    response_text = _call(BASE_PROMPT)
    try:
        if not response_text:
            raise ValueError("Empty response received from VLM.")
        return ChequeDetails.model_validate_json(response_text)
    except (json.JSONDecodeError, ValidationError, ValueError):
        # Retry once if JSON is malformed
        retry_prompt = (
            f"{BASE_PROMPT}\n\n"
            "CRITICAL: The previous output failed JSON validation. "
            "Ensure the output strictly conforms to the JSON schema."
        )
        try:
            second_text = _call(retry_prompt)
            return ChequeDetails.model_validate_json(second_text)
        except Exception as retry_err:
            raise ExtractionError(f"Failed to extract valid JSON after retry: {retry_err}") from retry_err


# ==============================================================================
# Unified VLM Extraction Dispatcher
# ==============================================================================
def extract_cheque_details(
    image: Image.Image,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    provider: Optional[str] = None
) -> Tuple[ChequeDetails, Dict[str, Any]]:
    """
    Extracts structured cheque details using either Groq (Qwen 3.8 27B Vision) or Google Gemini.
    Auto-detects the provider based on key format or environment variables.
    """
    groq_env = os.getenv("GROQ_API_KEY", "")
    gemini_env = os.getenv("GEMINI_API_KEY", "")

    selected_key = api_key or groq_env or gemini_env
    if not selected_key or selected_key.strip() in ("", "your_groq_api_key_here", "your_gemini_api_key_here"):
        raise AuthenticationError(
            "Missing API key. Please provide a GROQ_API_KEY or GEMINI_API_KEY in .env or the sidebar."
        )

    # Auto-detect provider if not specified
    if not provider:
        if selected_key.startswith("gsk_") or groq_env:
            provider = "groq"
        else:
            provider = "gemini"

    if provider.lower() == "groq":
        model = model_name or os.getenv("MODEL_NAME") or "qwen/qwen3.8-27b"
        if "gemini" in model.lower() or "llama-3.2" in model.lower():
            model = "qwen/qwen3.8-27b"
        parsed_details = _extract_with_groq(image, selected_key, model)
    else:
        model = model_name or os.getenv("MODEL_NAME") or "gemini-2.5-flash"
        parsed_details = _extract_with_gemini(image, selected_key, model)

    verification = verify_amount_match(
        amount_numeric=parsed_details.amount_numeric,
        amount_words=parsed_details.amount_words
    )

    return parsed_details, verification


# ==============================================================================
# In-Memory Sample Dummy Cheque Generator (No Disk Writes)
# ==============================================================================
def generate_sample_cheque(
    payer_name: str = "Alexander Wright",
    payee_name: str = "Elena Rostova",
    amount_numeric: float = 18500.00,
    amount_words: str = "Eighteen Thousand Five Hundred Only",
    account_no: str = "987654321012",
    cheque_no: str = "000452",
    bank_name: str = "HORIZON PACIFIC BANK",
    ifsc: str = "HPBK0004921",
    date_str: str = "2026-05-18",
    mismatch_test: bool = False
) -> Image.Image:
    """
    Generates a realistic dummy/specimen cheque image entirely in memory using Pillow.
    Provides immediate 1-click testing without searching for or saving test images.
    """
    if mismatch_test:
        amount_words = "Ten Thousand Only"  # Intentional discrepancy with 18500.00

    img = Image.new("RGB", (800, 360), color="#F7FAFC")
    draw = ImageDraw.Draw(img)

    draw.rectangle([(10, 10), (790, 350)], outline="#2D3748", width=2)
    draw.rectangle([(14, 14), (786, 346)], outline="#CBD5E0", width=1)

    draw.text((230, 150), "SPECIMEN / VOID", fill="#E2E8F0")

    draw.text((35, 25), bank_name, fill="#1A365D")
    draw.text((35, 45), f"Metro Plaza Branch | IFSC: {ifsc}", fill="#4A5568")

    draw.line([(25, 18), (130, 18)], fill="#2D3748", width=2)
    draw.line([(25, 28), (130, 28)], fill="#2D3748", width=2)
    draw.text((30, 19), "A/C PAYEE ONLY", fill="#2D3748")

    draw.text((600, 26), f"DATE: {date_str}", fill="#1A202C")
    draw.rectangle([(645, 22), (765, 46)], outline="#718096", width=1)

    draw.text((35, 95), "PAY", fill="#2D3748")
    draw.text((85, 93), payee_name, fill="#1A365D")
    draw.line([(80, 115), (600, 115)], fill="#A0AEC0", width=1)
    draw.text((615, 95), "OR BEARER", fill="#718096")

    draw.text((35, 145), "RUPEES", fill="#2D3748")
    draw.text((110, 143), amount_words, fill="#1A365D")
    draw.line([(105, 165), (550, 165)], fill="#A0AEC0", width=1)

    draw.rectangle([(575, 135), (765, 175)], outline="#2B6CB0", width=2, fill="#FFFFFF")
    draw.text((585, 145), f"Rs. {amount_numeric:,.2f}", fill="#1A365D")

    draw.text((35, 210), "A/C NO.", fill="#2D3748")
    draw.rectangle([(100, 205), (320, 235)], outline="#4A5568", width=1, fill="#FFFFFF")
    draw.text((115, 212), account_no, fill="#1A202C")

    draw.text((590, 205), "Authorized Signatory", fill="#718096")
    draw.text((590, 265), payer_name, fill="#2D3748")
    sig_points = [(590, 248), (620, 230), (645, 255), (680, 232), (720, 245), (745, 235)]
    draw.line(sig_points, fill="#1A365D", width=2)

    draw.rectangle([(14, 300), (786, 346)], fill="#EDF2F7")
    draw.text((180, 315), f"⑈{cheque_no}⑈ 110002034⑆ 001234⑈ 10", fill="#2D3748")

    return img
