# 🏦 AI Cheque Reader (Vision-Language Model Powered)

A lightweight, production-grade application that extracts, verifies, and audits financial details from scanned or photographed bank cheques using **Vision-Language Models (VLMs)**.

> **Zero Traditional OCR**: Avoids Tesseract, EasyOCR, and heavy CV models (`torch`, `opencv`). Instead, it uses fast, lightweight multimodal models (**Groq Qwen 3.8 27B** or **Google Gemini Flash**) to interpret cursive handwriting, typography, bank stamps, and MICR code bands directly from raw pixels.

---

## 🌟 Advanced Features
- **100% Free & Lightweight**: Works out-of-the-box with Groq's free tier (no credit card) and Gemini Flash.
- **In-Memory Lifespan**: All images and files are processed strictly in RAM (`io.BytesIO`)—never written or cached to disk.
- **📅 Cheque Expiry & Stale Date Validator**: Enforces standard banking regulations:
  - Identifies **Post-Dated Cheques (PDC)** with future issuance dates.
  - Flags **Stale / Expired Cheques** exceeding the 90-day (3-month) presentation window.
  - Computes remaining days of validity.
- **🛠️ In-Memory Image Preprocessing**: Non-destructive interactive controls to rotate (0°–270°), boost contrast, and sharpen dark or blurry mobile photos before sending them to the AI.
- **✏️ Human-in-the-Loop Review**: Allows operators to review and edit extracted fields before downloading, dynamically re-triggering amount and date audits.
- **📦 Batch Processing Mode**: Upload up to 5 cheques simultaneously, view an aggregated data table, and export a consolidated **CSV spreadsheet** or **JSON summary**.
- **⚖️ Discrepancy Checker**: Pure-Python amount parser cross-checks `amount_numeric` against `amount_words` to flag mismatches.
- **🎨 Light Blue Theme**: Styled with a sky-blue gradient header, 4 highlight metric cards, and validation badges.

---

## 📂 Project Structure
```text
cheque-extractor/
├── .streamlit/
│   └── config.toml     # Light blue UI theme configuration
├── app.py              # Streamlit dashboard (Single & Batch modes, Human-in-the-loop, Image tools)
├── extractor.py        # VLM dispatchers, Pydantic schema, date validator & image preprocessor
├── requirements.txt    # Minimal dependencies
├── .env.example        # Environment template (GROQ_API_KEY, GEMINI_API_KEY, MODEL_NAME)
└── README.md           # Documentation & learning guide
```

---

## 🚀 Setup & Run Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Your Free API Key
Copy the environment template:
```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```
Edit `.env` with your free **Groq API key** from [console.groq.com/keys](https://console.groq.com/keys):
```ini
GROQ_API_KEY=gsk_your_key_here
MODEL_NAME=qwen/qwen3.8-27b
```
*(Or use your free Google Gemini API key from [aistudio.google.com](https://aistudio.google.com/)).*

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧭 How to Use the 4 New Features

### 1. Date Expiry & Stale Cheque Checker
* Under the **Validation** tab, the system calculates the date difference between the cheque's issue date and today:
  * **Valid (0–90 days)**: Displays a green badge with remaining valid days.
  * **Stale (>90 days)**: Displays a red warning indicating the cheque has expired.
  * **Post-Dated (Future)**: Displays an amber alert warning that the cheque cannot be cleared yet.

### 2. In-Memory Image Preprocessing
* On the left column, expand **"🛠️ Image Preprocessing"**:
  * Adjust rotation (90°, 180°, 270°) to fix sideways phone camera uploads.
  * Increase **Contrast** (e.g. 1.3×) and **Sharpness** (e.g. 1.5×) to bring out faint handwriting.
  * Click **Extract Details** to pass the enhanced image to the AI.

### 3. Human-in-the-Loop Editable Form
* On the right column under **"Extracted Fields"**, toggle **"✏️ Enable Edit Mode"**.
* Modify any blurred or unreadable fields directly in the form.
* Click **"💾 Save & Re-verify Fields"**—the application updates all metrics, re-runs the amount and date checks, and records the session as **Human Verified**.

### 4. Batch Processing Mode
* In the left sidebar, switch **Application Mode** from *Single Cheque* to **Batch Mode (Up to 5 Cheques)**.
* Upload up to 5 cheque photos at once.
* Click **"⚡ Process All Cheques in Batch"** to watch the progress bar extract all cheques sequentially.
* Click **"⬇️ Download Batch CSV Spreadsheet"** to export an aggregated accounting report.
