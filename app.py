"""
app.py
------
Streamlit user interface for the AI Cheque Reader.

Modern, lightweight UI featuring:
1. Cheque Expiry & Stale Date Validator (Banking 90-day rule + future date warning)
2. In-Memory Image Controls (Rotate, Contrast, Sharpness adjustment before VLM extraction)
3. Human-in-the-Loop Editable Form (Operator review & correction before export)
4. Batch Processing Mode (Upload up to 5 cheques at once -> Consolidated CSV/JSON)
5. Light blue theme, custom CSS, gradient header, and status badges
6. Zero disk writes (100% in-memory processing)
"""

import os
import re
import csv
import json
import io
from datetime import datetime, date
from typing import List, Dict, Any
from PIL import Image, UnidentifiedImageError
import streamlit as st
from dotenv import load_dotenv

# Import modular VLM extraction and validation routines from extractor.py
from extractor import (
    extract_cheque_details,
    verify_amount_match,
    verify_cheque_date,
    preprocess_image,
    generate_sample_cheque,
    ChequeDetails,
    RateLimitError,
    AuthenticationError,
    ExtractionError
)

# Load environment variables (.env)
load_dotenv()


# ==============================================================================
# CSS Theme Injection Helper
# ==============================================================================
def load_css():
    """
    Injects custom CSS to style the Streamlit app with an attractive, modern,
    light-blue theme without requiring external UI libraries.
    """
    custom_css = """
    <style>
        /* ---------------------------------------------------------------------
           1. CORE PAGE RESETS & CHROME CLEANUP
           --------------------------------------------------------------------- */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }

        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #0D2B45;
        }

        /* ---------------------------------------------------------------------
           2. GRADIENT HEADER BANNER
           --------------------------------------------------------------------- */
        .cheque-header-banner {
            background: linear-gradient(135deg, #81D4FA 0%, #2196F3 60%, #0288D1 100%);
            padding: 22px 28px;
            border-radius: 16px;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 20px;
            box-shadow: 0 8px 24px rgba(33, 150, 243, 0.22);
            margin-bottom: 20px;
        }
        .cheque-header-icon {
            font-size: 2.8rem;
            background: rgba(255, 255, 255, 0.22);
            width: 68px;
            height: 68px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 16px;
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.35);
        }
        .cheque-header-title {
            margin: 0;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #FFFFFF !important;
            line-height: 1.2;
        }
        .cheque-header-subtitle {
            margin: 5px 0 0 0;
            font-size: 1.02rem;
            color: rgba(255, 255, 255, 0.94);
            font-weight: 400;
        }

        /* ---------------------------------------------------------------------
           3. WHITE CARD CONTAINERS & SHADOWS
           --------------------------------------------------------------------- */
        .white-card {
            background-color: #FFFFFF;
            border-radius: 14px;
            padding: 20px;
            border: 1px solid #D1E5F8;
            box-shadow: 0 6px 18px rgba(13, 43, 69, 0.05);
            margin-bottom: 18px;
        }
        .right-pane-card {
            background-color: #FFFFFF;
            border-radius: 14px;
            padding: 22px;
            border: 2px solid #BBDEFB;
            box-shadow: 0 8px 24px rgba(33, 150, 243, 0.09);
        }

        /* ---------------------------------------------------------------------
           4. TOP 4 HIGHLIGHT METRIC CARDS
           --------------------------------------------------------------------- */
        .highlight-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }
        .highlight-card {
            background: linear-gradient(145deg, #FFFFFF, #F1F8FE);
            border: 1px solid #C2E0FB;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 2px 8px rgba(33, 150, 243, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .highlight-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15);
        }
        .highlight-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            font-weight: 700;
            color: #0288D1;
            letter-spacing: 0.06em;
            margin-bottom: 4px;
        }
        .highlight-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0D2B45;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* ---------------------------------------------------------------------
           5. FIELD DETAIL ROWS (TAB 1)
           --------------------------------------------------------------------- */
        .field-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            border-bottom: 1px solid #EDF4FB;
            transition: background-color 0.15s ease;
        }
        .field-row:last-child {
            border-bottom: none;
        }
        .field-row:hover {
            background-color: #F8FBFE;
        }
        .field-title {
            font-size: 0.88rem;
            font-weight: 600;
            color: #4A6572;
        }
        .field-content {
            font-size: 0.95rem;
            font-weight: 600;
            color: #0D2B45;
            text-align: right;
        }
        .null-tag {
            color: #94A3B8;
            font-style: italic;
            font-weight: 400;
            font-size: 0.86rem;
        }

        /* ---------------------------------------------------------------------
           6. VALIDATION & STATUS BADGES (TAB 3)
           --------------------------------------------------------------------- */
        .status-badge {
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 12px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            font-size: 0.92rem;
            line-height: 1.4;
            border: 1px solid transparent;
        }
        .badge-success {
            background-color: #E8F5E9;
            color: #1B5E20;
            border-color: #A5D6A7;
        }
        .badge-warning {
            background-color: #FFF8E1;
            color: #795548;
            border-color: #FFE082;
        }
        .badge-error {
            background-color: #FFEBEE;
            color: #B71C1C;
            border-color: #FFCDD2;
        }
        .badge-icon {
            font-size: 1.3rem;
            flex-shrink: 0;
            margin-top: -2px;
        }

        .human-badge {
            display: inline-block;
            background: #E8F5E9;
            color: #2E7D32;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid #A5D6A7;
            margin-bottom: 10px;
        }
        .ai-badge {
            display: inline-block;
            background: #E3F2FD;
            color: #0288D1;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid #90CAF9;
            margin-bottom: 10px;
        }

        /* ---------------------------------------------------------------------
           7. SIDEBAR CARDS & STEP INDICATORS
           --------------------------------------------------------------------- */
        .sidebar-card {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #CDE4FA;
            box-shadow: 0 2px 8px rgba(13, 43, 69, 0.04);
            margin-bottom: 16px;
        }
        .step-item {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            font-size: 0.88rem;
            color: #0D2B45;
        }
        .step-item:last-child {
            margin-bottom: 0;
        }
        .step-num {
            background: #2196F3;
            color: #FFFFFF;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 700;
            flex-shrink: 0;
        }
        .privacy-box {
            background-color: #E1F5FE;
            border-left: 4px solid #0288D1;
            padding: 10px 14px;
            border-radius: 0 8px 8px 0;
            font-size: 0.83rem;
            color: #0277BD;
            margin-top: 14px;
            line-height: 1.4;
        }

        /* ---------------------------------------------------------------------
           8. STREAMLIT BUTTONS & INTERACTIVE CONTROLS
           --------------------------------------------------------------------- */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2196F3 0%, #0288D1 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.4rem !important;
            font-weight: 700 !important;
            font-size: 1.02rem !important;
            box-shadow: 0 4px 14px rgba(33, 150, 243, 0.35) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #42A5F5 0%, #039BE5 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(33, 150, 243, 0.5) !important;
        }

        div.stButton > button[kind="secondary"] {
            border-radius: 10px !important;
            border: 1px solid #B3D7F8 !important;
            background: #FFFFFF !important;
            color: #0288D1 !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            background: #EAF4FF !important;
            border-color: #2196F3 !important;
            transform: translateY(-1px) !important;
        }

        div.stDownloadButton > button {
            background: linear-gradient(135deg, #0288D1 0%, #01579B 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.6rem 1.4rem !important;
            font-weight: 700 !important;
            font-size: 0.98rem !important;
            box-shadow: 0 4px 12px rgba(2, 136, 209, 0.3) !important;
            transition: all 0.25s ease !important;
        }
        div.stDownloadButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(2, 136, 209, 0.45) !important;
        }

        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            color: #4A6572 !important;
            padding: 10px 16px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #0288D1 !important;
            border-bottom: 3px solid #0288D1 !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="AI Cheque Reader",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply our custom stylesheet
load_css()


# ==============================================================================
# Session State Initialization (In-Memory Processing)
# ==============================================================================
# Single Cheque State
if "raw_cheque_image" not in st.session_state:
    st.session_state["raw_cheque_image"] = None
if "image_source_name" not in st.session_state:
    st.session_state["image_source_name"] = None
if "extraction_result" not in st.session_state:
    st.session_state["extraction_result"] = None
if "verification_result" not in st.session_state:
    st.session_state["verification_result"] = None
if "date_verification_result" not in st.session_state:
    st.session_state["date_verification_result"] = None
if "human_verified" not in st.session_state:
    st.session_state["human_verified"] = False

# Image Filter State
if "img_rotation" not in st.session_state:
    st.session_state["img_rotation"] = 0
if "img_contrast" not in st.session_state:
    st.session_state["img_contrast"] = 1.0
if "img_sharpness" not in st.session_state:
    st.session_state["img_sharpness"] = 1.0

# Batch Processing State
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = []


# ==============================================================================
# Header Banner
# ==============================================================================
st.markdown("""
<div class="cheque-header-banner">
    <div class="cheque-header-icon">🏦</div>
    <div>
        <h1 class="cheque-header-title">AI Cheque Reader</h1>
        <p class="cheque-header-subtitle">Extract, validate, and verify bank cheques using Vision-Language Models</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# Sidebar: Configuration, Mode & Quick Test
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings & Mode")

    # Mode Selector: Single Cheque vs Batch Mode
    app_mode = st.radio(
        "Application Mode",
        options=["Single Cheque Inspection", "Batch Mode (Up to 5 Cheques)"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🤖 VLM Provider")

    groq_env = os.getenv("GROQ_API_KEY", "") or os.getenv("Groq_API_Key", "")
    gemini_env = os.getenv("GEMINI_API_KEY", "")

    default_prov_idx = 0 if groq_env or not gemini_env else 1
    provider_choice = st.radio(
        "Select Provider",
        options=["Groq (100% Free, Qwen 3.8)", "Google Gemini"],
        index=default_prov_idx
    )
    is_groq = "Groq" in provider_choice

    if is_groq:
        api_key_input = st.text_input(
            "Groq API Key",
            value=groq_env if groq_env != "your_groq_api_key_here" else "",
            type="password",
            help="Free key from https://console.groq.com/keys"
        )
        model_name_input = st.text_input(
            "Model Name in Use",
            value="qwen/qwen3.8-27b",
            help="Active multimodal vision model on Groq"
        )
    else:
        api_key_input = st.text_input(
            "Gemini API Key",
            value=gemini_env if gemini_env != "your_gemini_api_key_here" else "",
            type="password",
            help="Key from https://aistudio.google.com/"
        )
        model_name_input = st.text_input(
            "Model Name in Use",
            value=os.getenv("MODEL_NAME", "gemini-2.5-flash")
        )

    st.markdown("---")

    # Quick Test Generator (Single Mode)
    if app_mode == "Single Cheque Inspection":
        st.markdown("### 🧪 Quick Test Samples")
        st.caption("Generate in-memory specimen cheques with 1 click:")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Sample (Valid)", use_container_width=True):
                st.session_state["raw_cheque_image"] = generate_sample_cheque(mismatch_test=False)
                st.session_state["image_source_name"] = "Sample Specimen (Valid)"
                st.session_state["extraction_result"] = None
                st.session_state["verification_result"] = None
                st.session_state["date_verification_result"] = None
                st.session_state["human_verified"] = False
                st.session_state["img_rotation"] = 0
                st.session_state["img_contrast"] = 1.0
                st.session_state["img_sharpness"] = 1.0
                st.rerun()

        with col_b2:
            if st.button("Sample (Mismatch)", use_container_width=True):
                st.session_state["raw_cheque_image"] = generate_sample_cheque(mismatch_test=True)
                st.session_state["image_source_name"] = "Sample Specimen (Mismatch)"
                st.session_state["extraction_result"] = None
                st.session_state["verification_result"] = None
                st.session_state["date_verification_result"] = None
                st.session_state["human_verified"] = False
                st.session_state["img_rotation"] = 0
                st.session_state["img_contrast"] = 1.0
                st.session_state["img_sharpness"] = 1.0
                st.rerun()

    # Privacy Note
    st.markdown("""
    <div class="privacy-box">
        🔒 <b>Privacy Note</b><br>
        Use sample cheques only. Nothing is stored; all image processing occurs strictly in RAM.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# MODE 1: SINGLE CHEQUE INSPECTION
# ==============================================================================
if app_mode == "Single Cheque Inspection":
    col_left, col_right = st.columns([1, 1], gap="large")

    # --------------------------------------------------------------------------
    # LEFT COLUMN: Upload, In-Memory Preprocessing & Image Display
    # --------------------------------------------------------------------------
    with col_left:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown("#### 📷 Cheque Image")

        uploaded_file = st.file_uploader(
            "Upload a scanned or photographed cheque (PNG, JPG):",
            type=["png", "jpg", "jpeg"],
            help="Upload any dummy or specimen bank cheque."
        )

        if uploaded_file is not None:
            try:
                image_bytes = uploaded_file.getvalue()
                loaded_image = Image.open(io.BytesIO(image_bytes))
                st.session_state["raw_cheque_image"] = loaded_image
                st.session_state["image_source_name"] = uploaded_file.name
            except (UnidentifiedImageError, ValueError) as err:
                st.error(f"Invalid image file: Could not parse '{uploaded_file.name}'.")
                st.session_state["raw_cheque_image"] = None

        # Display and Preprocess
        if st.session_state["raw_cheque_image"] is not None:
            raw_img: Image.Image = st.session_state["raw_cheque_image"]
            src_label = st.session_state.get("image_source_name", "Cheque Image")

            # ------------------------------------------------------------------
            # FEATURE 2: In-Memory Image Controls (Rotate, Contrast, Sharpness)
            # ------------------------------------------------------------------
            with st.expander("🛠️ Image Preprocessing (Enhance Readability)", expanded=False):
                st.caption("Adjust rotation and clarity before sending to the VLM (processed 100% in memory):")

                c_ctrl1, c_ctrl2 = st.columns(2)
                with c_ctrl1:
                    rot_choice = st.selectbox(
                        "Rotate Image",
                        options=[0, 90, 180, 270],
                        index=[0, 90, 180, 270].index(st.session_state["img_rotation"]),
                        format_func=lambda r: f"{r}° (No rotation)" if r == 0 else f"{r}° Clockwise"
                    )
                    st.session_state["img_rotation"] = rot_choice

                with c_ctrl2:
                    if st.button("↺ Reset Filters", use_container_width=True):
                        st.session_state["img_rotation"] = 0
                        st.session_state["img_contrast"] = 1.0
                        st.session_state["img_sharpness"] = 1.0
                        st.rerun()

                c_ctrl3, c_ctrl4 = st.columns(2)
                with c_ctrl3:
                    st.session_state["img_contrast"] = st.slider(
                        "Contrast",
                        min_value=0.5,
                        max_value=2.0,
                        value=float(st.session_state["img_contrast"]),
                        step=0.1
                    )
                with c_ctrl4:
                    st.session_state["img_sharpness"] = st.slider(
                        "Sharpness",
                        min_value=0.5,
                        max_value=2.5,
                        value=float(st.session_state["img_sharpness"]),
                        step=0.1
                    )

            # Apply preprocessor to the raw image in memory
            active_img = preprocess_image(
                image=raw_img,
                rotation=st.session_state["img_rotation"],
                contrast=st.session_state["img_contrast"],
                sharpness=st.session_state["img_sharpness"]
            )

            st.caption(f"Source: **{src_label}** | Dimensions: {active_img.width} × {active_img.height} px")
            st.image(active_img, use_container_width=True)

            if st.button("🗑️ Clear Cheque", use_container_width=True):
                st.session_state["raw_cheque_image"] = None
                st.session_state["image_source_name"] = None
                st.session_state["extraction_result"] = None
                st.session_state["verification_result"] = None
                st.session_state["date_verification_result"] = None
                st.session_state["human_verified"] = False
                st.rerun()
        else:
            active_img = None
            st.info("👆 Upload an image or select a **Sample Cheque** from the left sidebar to begin.")

        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # RIGHT COLUMN: Extraction Trigger & Structured Details Card
    # --------------------------------------------------------------------------
    with col_right:
        st.markdown('<div class="right-pane-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Extracted Details")

        extract_clicked = st.button("⚡ Extract Details", type="primary", use_container_width=True)

        if extract_clicked:
            if active_img is None:
                st.warning("Please upload a cheque image or select a sample first.")
            elif not api_key_input:
                st.error("API key is missing. Please enter your API key in the sidebar.")
            else:
                with st.spinner("Reading your cheque with VLM..."):
                    try:
                        prov_key = "groq" if is_groq else "gemini"
                        details, verification = extract_cheque_details(
                            image=active_img,
                            api_key=api_key_input,
                            model_name=model_name_input,
                            provider=prov_key
                        )
                        # FEATURE 1: Verify Cheque Date (Banking 90-day validity rule)
                        date_verification = verify_cheque_date(details.transaction_date)

                        st.session_state["extraction_result"] = details
                        st.session_state["verification_result"] = verification
                        st.session_state["date_verification_result"] = date_verification
                        st.session_state["human_verified"] = False

                        st.toast("Cheque details extracted successfully!", icon="✅")
                        st.balloons()
                    except AuthenticationError as auth_err:
                        st.error(f"🔐 Authentication Error: {auth_err}")
                    except RateLimitError as rate_err:
                        st.warning(f"⏳ Rate Limit Reached (429): {rate_err}")
                    except ExtractionError as ext_err:
                        st.error(f"⚠️ Extraction Failed: {ext_err}")
                    except Exception as gen_err:
                        st.error(f"❌ Unexpected Error: {gen_err}")

        # If results exist, display them
        if st.session_state["extraction_result"] is not None:
            details: ChequeDetails = st.session_state["extraction_result"]
            verification: dict = st.session_state["verification_result"]
            date_ver: dict = st.session_state.get("date_verification_result", {})

            # Status Tag: AI vs Human Verified
            if st.session_state.get("human_verified", False):
                st.markdown('<span class="human-badge">✍️ Human Verified & Saved</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="ai-badge">🤖 AI Extracted (Awaiting Review)</span>', unsafe_allow_html=True)

            # ------------------------------------------------------------------
            # TOP 4 HIGHLIGHT METRIC CARDS
            # ------------------------------------------------------------------
            payee_disp = details.payee_name or "Not Specified"
            amt_disp = f"{details.amount_numeric:,.2f}" if details.amount_numeric is not None else "Unreadable"
            date_disp = details.transaction_date or "Not Specified"
            chk_disp = details.cheque_number or "Unreadable"

            st.markdown(f"""
            <div class="highlight-grid">
                <div class="highlight-card">
                    <div class="highlight-label">Payee Name</div>
                    <div class="highlight-value" title="{payee_disp}">{payee_disp}</div>
                </div>
                <div class="highlight-card">
                    <div class="highlight-label">Amount</div>
                    <div class="highlight-value" title="{amt_disp}">{amt_disp}</div>
                </div>
                <div class="highlight-card">
                    <div class="highlight-label">Date</div>
                    <div class="highlight-value" title="{date_disp}">{date_disp}</div>
                </div>
                <div class="highlight-card">
                    <div class="highlight-label">Cheque No.</div>
                    <div class="highlight-value" title="{chk_disp}">{chk_disp}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ------------------------------------------------------------------
            # THREE TABS: Extracted Fields, Raw JSON, Validation
            # ------------------------------------------------------------------
            tab_fields, tab_json, tab_validation = st.tabs([
                "📋 Extracted Fields",
                "🔍 Raw JSON",
                "🛡️ Validation"
            ])

            # ------------------- TAB 1: EXTRACTED FIELDS ----------------------
            with tab_fields:
                # FEATURE 4: Human-in-the-Loop Editable Form
                is_edit_mode = st.toggle("✏️ Enable Edit Mode (Human-in-the-Loop)", value=False)

                if is_edit_mode:
                    st.caption("Correct any blurry or unreadable fields before exporting:")
                    with st.form("human_review_form"):
                        e_payee = st.text_input("Payee Name", value=details.payee_name or "")
                        e_payer = st.text_input("Payer Name", value=details.payer_name or "")
                        
                        f_col1, f_col2 = st.columns(2)
                        with f_col1:
                            e_amount_num = st.number_input(
                                "Amount (Numeric)",
                                value=float(details.amount_numeric) if details.amount_numeric is not None else 0.0,
                                step=0.01,
                                format="%.2f"
                            )
                        with f_col2:
                            e_amount_words = st.text_input("Amount (In Words)", value=details.amount_words or "")

                        f_col3, f_col4 = st.columns(2)
                        with f_col3:
                            e_date = st.text_input("Transaction Date (YYYY-MM-DD)", value=details.transaction_date or "")
                        with f_col4:
                            e_cheque_no = st.text_input("Cheque Number", value=details.cheque_number or "")

                        f_col5, f_col6 = st.columns(2)
                        with f_col5:
                            e_bank = st.text_input("Bank Name", value=details.bank_name or "")
                        with f_col6:
                            e_ifsc = st.text_input("IFSC / MICR Code", value=details.ifsc_or_micr_code or "")

                        f_col7, f_col8 = st.columns(2)
                        with f_col7:
                            e_payer_acc = st.text_input("Payer Account Number", value=details.payer_account_number or "")
                        with f_col8:
                            e_payee_acc = st.text_input("Payee Account Number", value=details.payee_account_number or "")

                        e_sig = st.checkbox("Signature Present", value=bool(details.signature_present))

                        save_edits = st.form_submit_button("💾 Save & Re-verify Fields", type="primary", use_container_width=True)

                        if save_edits:
                            # Update details with human corrections
                            updated_details = ChequeDetails(
                                payee_name=e_payee or None,
                                payer_name=e_payer or None,
                                payee_account_number=e_payee_acc or None,
                                payer_account_number=e_payer_acc or None,
                                amount_numeric=e_amount_num if e_amount_num > 0 else None,
                                amount_words=e_amount_words or None,
                                transaction_date=e_date or None,
                                cheque_number=e_cheque_no or None,
                                bank_name=e_bank or None,
                                ifsc_or_micr_code=e_ifsc or None,
                                signature_present=e_sig
                            )
                            # Re-verify amount and date with edited data
                            st.session_state["extraction_result"] = updated_details
                            st.session_state["verification_result"] = verify_amount_match(
                                updated_details.amount_numeric,
                                updated_details.amount_words
                            )
                            st.session_state["date_verification_result"] = verify_cheque_date(updated_details.transaction_date)
                            st.session_state["human_verified"] = True
                            st.toast("Fields updated and verified by Human Reviewer!", icon="✍️")
                            st.rerun()

                else:
                    # Clean Read-Only Key-Value Rows
                    def val_fmt(v):
                        if v is None:
                            return '<span class="null-tag">null (unreadable / absent)</span>'
                        if isinstance(v, bool):
                            return "✍️ Yes (Detected)" if v else "❌ No (Unsigned)"
                        if isinstance(v, (int, float)):
                            return f"{v:,.2f}"
                        return str(v)

                    st.markdown(f"""
                    <div class="white-card" style="padding: 10px 16px; margin-top: 10px;">
                        <div class="field-row">
                            <span class="field-title">Payer / Issuer Name</span>
                            <span class="field-content">{val_fmt(details.payer_name)}</span>
                        </div>
                        <div class="field-row">
                            <span class="field-title">Amount in Words</span>
                            <span class="field-content">{val_fmt(details.amount_words)}</span>
                        </div>
                        <div class="field-row">
                            <span class="field-title">Payer Account Number</span>
                            <span class="field-content">{val_fmt(details.payer_account_number)}</span>
                        </div>
                        <div class="field-row">
                            <span class="field-title">Payee Account Number</span>
                            <span class="field-content">{val_fmt(details.payee_account_number)}</span>
                        </div>
                        <div class="field-row">
                            <span class="field-title">Bank Name</span>
                            <span class="field-content">{val_fmt(details.bank_name)}</span>
                        </div>
                        <div class="field-row">
                            <span class="field-title">IFSC / MICR Code</span>
                            <span class="field-content">{val_fmt(details.ifsc_or_micr_code)}</span>
                        </div>
                        <div class="field-row">
                            <span class="field-title">Signature Detected</span>
                            <span class="field-content">{val_fmt(details.signature_present)}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ------------------- TAB 2: RAW JSON ------------------------------
            with tab_json:
                json_dict = details.model_dump()
                # Attach metadata
                json_dict["_metadata"] = {
                    "human_verified": st.session_state.get("human_verified", False),
                    "amount_match_status": verification.get("status"),
                    "date_validity_status": date_ver.get("status")
                }
                json_str = json.dumps(json_dict, indent=2)

                st.json(json_dict)

                st.download_button(
                    label="⬇️ Download Verified JSON",
                    data=json_str,
                    file_name="cheque_details_verified.json",
                    mime="application/json",
                    use_container_width=True
                )

            # ------------------- TAB 3: VALIDATION ----------------------------
            with tab_validation:
                # 1. FEATURE 1: Date Expiry & Stale Cheque Check (90-Day Rule)
                d_status = date_ver.get("status")
                d_msg = date_ver.get("message", "")

                if d_status == "VALID":
                    st.markdown(f"""
                    <div class="status-badge badge-success">
                        <span class="badge-icon">📅</span>
                        <div>
                            <b>Valid Cheque Date (90-Day Banking Rule)</b><br>
                            {d_msg}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif d_status == "STALE":
                    st.markdown(f"""
                    <div class="status-badge badge-error">
                        <span class="badge-icon">🚨</span>
                        <div>
                            <b>Stale / Expired Cheque Detected!</b><br>
                            {d_msg}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif d_status == "POST_DATED":
                    st.markdown(f"""
                    <div class="status-badge badge-warning">
                        <span class="badge-icon">⏳</span>
                        <div>
                            <b>Post-Dated Cheque (PDC)</b><br>
                            {d_msg}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-badge badge-warning">
                        <span class="badge-icon">⚠️</span>
                        <div>
                            <b>Date Check Notice</b><br>
                            {d_msg}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 2. Amount Match Check
                amt_status = verification.get("status")
                amt_msg = verification.get("message", "")

                if amt_status == "MATCH":
                    st.markdown(f"""
                    <div class="status-badge badge-success">
                        <span class="badge-icon">✅</span>
                        <div>
                            <b>Amount matches words</b><br>
                            {amt_msg}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif amt_status == "MISMATCH":
                    st.markdown(f"""
                    <div class="status-badge badge-error">
                        <span class="badge-icon">🚨</span>
                        <div>
                            <b>Amount mismatch detected!</b><br>
                            {amt_msg}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-badge badge-warning">
                        <span class="badge-icon">⚠️</span>
                        <div>
                            <b>Amount check inconclusive</b><br>
                            {amt_msg}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 3. Missing Fields Audit
                missing_fields = []
                for field_name, value in details.model_dump().items():
                    if value is None:
                        readable_name = field_name.replace("_", " ").title()
                        missing_fields.append(readable_name)

                if missing_fields:
                    items_html = "".join([f"<li><b>{f}</b>: unreadable or omitted</li>" for f in missing_fields])
                    st.markdown(f"""
                    <div class="status-badge badge-warning">
                        <span class="badge-icon">⚠️</span>
                        <div>
                            <b>Missing / Unreadable Fields ({len(missing_fields)})</b>
                            <ul style="margin: 6px 0 0 0; padding-left: 20px;">
                                {items_html}
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="status-badge badge-success">
                        <span class="badge-icon">✨</span>
                        <div><b>All 11 fields successfully detected without omissions!</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                # 4. Signature Status Badge
                if details.signature_present is True:
                    st.markdown("""
                    <div class="status-badge badge-success">
                        <span class="badge-icon">✍️</span>
                        <div><b>Signature Present</b>: Visibly present in authorized signatory area.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="status-badge badge-warning">
                        <span class="badge-icon">⚠️</span>
                        <div><b>Unsigned / No Signature</b>: No visible handwritten signature detected.</div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.caption("Extracted fields, highlight metrics, and validation audits will appear here after clicking **Extract Details**.")

        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# MODE 2: BATCH PROCESSING (Up to 5 Cheques)
# ==============================================================================
else:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown("### 📦 Batch Cheque Processing (Up to 5 Cheques)")
    st.markdown(
        "Upload multiple cheques simultaneously. The VLM will extract details from each cheque, "
        "verify amounts and dates, and generate a **consolidated CSV spreadsheet and JSON export** in memory."
    )

    batch_files = st.file_uploader(
        "Select up to 5 Cheque Images (JPG / PNG):",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload up to 5 cheque images."
    )

    if batch_files:
        if len(batch_files) > 5:
            st.warning("⚠️ You selected more than 5 files. Only the first 5 will be processed.")
            batch_files = batch_files[:5]

        st.markdown(f"**Selected {len(batch_files)} file(s):**")
        
        # Display small preview row of uploaded files
        prev_cols = st.columns(len(batch_files))
        for idx, f in enumerate(batch_files):
            with prev_cols[idx]:
                try:
                    p_img = Image.open(io.BytesIO(f.getvalue()))
                    st.image(p_img, caption=f.name, use_container_width=True)
                except Exception:
                    st.caption(f.name)

        # Batch Run Button
        if st.button("⚡ Process All Cheques in Batch", type="primary", use_container_width=True):
            if not api_key_input:
                st.error("API key is missing. Please enter your API key in the sidebar.")
            else:
                batch_records = []
                progress_bar = st.progress(0, text="Starting batch extraction...")
                prov_key = "groq" if is_groq else "gemini"

                for i, f in enumerate(batch_files):
                    progress_bar.progress(
                        (i) / len(batch_files),
                        text=f"Processing {f.name} ({i + 1}/{len(batch_files)})..."
                    )
                    try:
                        img_obj = Image.open(io.BytesIO(f.getvalue()))
                        details, v_amt = extract_cheque_details(
                            image=img_obj,
                            api_key=api_key_input,
                            model_name=model_name_input,
                            provider=prov_key
                        )
                        v_date = verify_cheque_date(details.transaction_date)

                        rec = details.model_dump()
                        rec["filename"] = f.name
                        rec["amount_match"] = v_amt.get("status")
                        rec["date_status"] = v_date.get("status")
                        rec["date_age_days"] = v_date.get("days")
                        batch_records.append(rec)
                    except Exception as b_err:
                        batch_records.append({
                            "filename": f.name,
                            "payee_name": None,
                            "payer_name": None,
                            "amount_numeric": None,
                            "amount_words": None,
                            "transaction_date": None,
                            "cheque_number": None,
                            "bank_name": None,
                            "ifsc_or_micr_code": None,
                            "payer_account_number": None,
                            "payee_account_number": None,
                            "signature_present": None,
                            "amount_match": "ERROR",
                            "date_status": "ERROR",
                            "error_note": str(b_err)
                        })

                progress_bar.progress(1.0, text="Batch processing complete!")
                st.session_state["batch_results"] = batch_records
                st.toast(f"Successfully processed {len(batch_records)} cheques!", icon="🎉")
                st.balloons()

    # If batch results exist, render consolidated table and download buttons
    if st.session_state["batch_results"]:
        records: List[Dict[str, Any]] = st.session_state["batch_results"]
        st.markdown("---")
        st.markdown(f"#### 📊 Consolidated Batch Results ({len(records)} Cheques)")

        # Display clean tabular view using Streamlit table
        display_data = []
        for r in records:
            display_data.append({
                "File": r.get("filename"),
                "Payee": r.get("payee_name") or "null",
                "Payer": r.get("payer_name") or "null",
                "Amount": f"{r.get('amount_numeric'):,.2f}" if r.get("amount_numeric") is not None else "null",
                "Amount Words": r.get("amount_words") or "null",
                "Date": r.get("transaction_date") or "null",
                "Date Status": r.get("date_status"),
                "Cheque No.": r.get("cheque_number") or "null",
                "Bank": r.get("bank_name") or "null",
                "Signature": "✍️ Yes" if r.get("signature_present") else "❌ No",
                "Amount Match": r.get("amount_match")
            })

        st.dataframe(display_data, use_container_width=True)

        # ----------------------------------------------------------------------
        # CSV & JSON In-Memory Exporters (Zero Disk Writes)
        # ----------------------------------------------------------------------
        csv_buffer = io.StringIO()
        fieldnames = [
            "filename", "payee_name", "payer_name", "amount_numeric", "amount_words",
            "transaction_date", "date_status", "date_age_days", "cheque_number",
            "bank_name", "payer_account_number", "payee_account_number",
            "ifsc_or_micr_code", "signature_present", "amount_match"
        ]
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)
        csv_content = csv_buffer.getvalue()

        json_content = json.dumps(records, indent=2)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="⬇️ Download Batch CSV Spreadsheet",
                data=csv_content,
                file_name="batch_cheques_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                label="⬇️ Download Batch JSON",
                data=json_content,
                file_name="batch_cheques_summary.json",
                mime="application/json",
                use_container_width=True
            )

    st.markdown('</div>', unsafe_allow_html=True)
