import streamlit as st
import pandas as pd
import pdfplumber
import re
import os
import joblib

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =========================================================
# CONFIGURATION & RECOVERY
# =========================================================
st.set_page_config(page_title="Universal Bank AI", layout="wide")

MODEL_FILE = "trained_categories.pkl"

def load_rules():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except:
            return {}
    return {}

def save_rules(rules):
    joblib.dump(rules, MODEL_FILE)

trained_rules = load_rules()

# =========================================================
# AI MODEL INITIALIZATION
# =========================================================
@st.cache_resource
def load_ai():
    try:
        api_key = st.secrets["WATSONX_APIKEY"].strip()
        project_id = st.secrets["WATSONX_PROJECT_ID"].strip()

        return ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            credentials={"apikey": api_key, "url": "https://ca-tor.ml.cloud.ibm.com"},
            project_id=project_id,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 500,
                GenParams.TEMPERATURE: 0.1,
            },
        )
    except:
        return None

ai_model = load_ai()

# =========================================================
# DETECTION & CATEGORIZATION
# =========================================================
def detect_bank(text, file_name=""):
    text = text.lower()
    fn = file_name.lower()

    if "triangle mastercard" in text or "triangle" in fn:
        return "TRIANGLE"
    if "royal bank" in text or "rbc" in text or "rbc" in fn:
        if "operating loan" in text or "credit line" in fn:
            return "RBC_LOAN"
        return "RBC"
    if "scotiabank" in text or "scotia" in fn or "scene" in fn:
        return "SCOTIA"
    
    return "GENERIC"

base_rules = {
    "tim hortons": "Food",
    "freshco": "Groceries",
    "esso": "Fuel",
    "uber": "Transportation",
    "amazon": "Shopping",
    "cdn tire": "Shopping",
    "loan payment": "Loan Payment",
    "interest": "Interest",
    "fee": "Bank Fee",
    "transfer": "Transfer",
    "robert half": "Staffing/Income",
}

def categorize(desc):
    d = desc.lower()
    for k, v in trained_rules.items():
        if k in d: return v
    for k, v in base_rules.items():
        if k in d: return v
    return "Other"

# =========================================================
# PDF EXTRACTION LOGIC (FIXED FOR DATES)
# =========================================================
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    return text

def parse_pdf_content(text, bank_label):
    rows = []
    
    # Pattern 1: DD MMM (e.g., 11 Dec) - Standard RBC
    pattern_dd_mmm = re.compile(r"(\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\s+(.*?)\s+([\d,]+\.\d{2})")
    
    # Pattern 2: MMM DD (e.g., Dec 11) - RBC Loan / Triangle
    pattern_mmm_dd = re.compile(r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2})\s+(.*?)\s+(-?\$?[\d,]+\.\d{2})")

    # Try DD MMM
    for match in pattern_dd_mmm.finditer(text):
        date, desc, amt_str = match.groups()
        amt = float(amt_str.replace(",", ""))
        # RBC Chequing logic: if it's in the 'Debits' column area or has spending keywords
        if any(w in desc.lower() for w in ["payment", "fee", "sent", "debit"]):
            amt = -abs(amt)
        rows.append({"Date": date, "Description": desc.strip(), "Amount": amt, "Bank Name": bank_label})

    # Try MMM DD (If first pattern yielded little or if it's a Loan/Triangle)
    if len(rows) < 3: 
        for match in pattern_mmm_dd.finditer(text):
            date, desc, amt_str = match.groups()
            amt = float(amt_str.replace("$", "").replace(",", ""))
            # Triangle amounts are positive for spending on statements; flip them
            if bank_label == "TRIANGLE" and amt > 0:
                amt = -amt
            rows.append({"Date": date, "Description": desc.strip(), "Amount": amt, "Bank Name": bank_label})

    return pd.DataFrame(rows)

# =========================================================
# CSV / EXCEL PARSER (FOR SCOTIA & OTHERS)
# =========================================================
def parse_table_file(file, bank_label):
    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        df.columns = [c.lower().strip() for c in df.columns]

        # Flexible column mapping
        date_col = next((c for c in df.columns if "date" in c), None)
        desc_col = next((c for c in df.columns if "description" in c or "details" in c), None)
        amt_col = next((c for c in df.columns if "amount" in c or "amt" in c or "value" in c), None)

        if desc_col and amt_col:
            res = pd.DataFrame()
            res["Date"] = df[date_col].astype(str) if date_col else ""
            res["Description"] = df[desc_col].astype(str)
            res["Amount"] = pd.to_numeric(df[amt_col], errors='coerce')
            res["Bank Name"] = bank_label
            return res.dropna(subset=["Amount"])
    except:
        pass
    return pd.DataFrame()

# =========================================================
# PROCESS FILE & UI
# =========================================================
def process_file(file):
    # Initial detection based on filename
    bank = detect_bank("", file.name)
    
    if file.name.endswith(("csv", "xlsx", "xls")):
        df = parse_table_file(file, bank)
    else:
        text = extract_text_from_pdf(file)
        bank = detect_bank(text, file.name)
        df = parse_pdf_content(text, bank)
        
        if df.empty and ai_model:
            # Fallback to AI if regex fails
            df = ai_parse(text)
            df["Bank Name"] = bank

    if not df.empty:
        df["Category"] = df["Description"].apply(categorize)
        # Ensure exact column order requested
        expected_cols = ["Date", "Description", "Amount", "Category", "Bank Name"]
        for col in expected_cols:
            if col not in df.columns: df[col] = ""
        df = df[expected_cols]

    return df

# (AI Parse function remains as in your original snippet)
def ai_parse(text):
    if ai_model is None: return pd.DataFrame()
    prompt = f"Extract bank transactions as a table: Date | Description | Amount. Spendings negative. \n\n{text[:5000]}"
    try:
        response = ai_model.generate(prompt)
        raw = response["results"][0]["generated_text"]
        rows = []
        for line in raw.split("\n"):
            if "|" in line:
                p = [x.strip() for x in line.split("|")]
                if len(p) >= 3:
                    try:
                        amt = float(re.sub(r"[^\d.-]", "", p[2]))
                        rows.append({"Date": p[0], "Description": p[1], "Amount": amt})
                    except: pass
        return pd.DataFrame(rows)
    except: return pd.DataFrame()

# =========================================================
# STREAMLIT UI
# =========================================================
st.title("🏦 Universal Canadian Bank Analyzer")

files = st.file_uploader("Upload Statements (RBC, Scotia, Triangle)", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

if files:
    all_dfs = []
    for f in files:
        with st.spinner(f"Processing {f.name}..."):
            df_file = process_file(f)
            if not df_file.empty:
                all_dfs.append(df_file)
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        st.subheader("Combined Transaction Report")
        st.dataframe(final_df, use_container_width=True)

        # Financial Metrics
        spending = final_df[final_df.Amount < 0].Amount.sum()
        income = final_df[final_df.Amount > 0].Amount.sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Spending", f"${abs(spending):,.2f}", delta_color="inverse")
        m2.metric("Total Income", f"${income:,.2f}")
        m3.metric("Net Flow", f"${income + spending:,.2f}")

        # Visuals
        st.bar_chart(final_df[final_df.Amount < 0].groupby("Category")["Amount"].sum().abs())
    else:
        st.error("Could not extract data. Please check file formats.")
