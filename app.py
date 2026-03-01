import streamlit as st
import pandas as pd
import pdfplumber
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =========================================================
# PAGE CONFIG & AI SETUP
# =========================================================
st.set_page_config(page_title="Universal Bank AI", layout="wide")

@st.cache_resource
def get_ai_model():
    try:
        api_key = st.secrets["WATSONX_APIKEY"].strip()
        project_id = st.secrets["WATSONX_PROJECT_ID"].strip()
        return ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials={"apikey": api_key, "url": "https://ca-tor.ml.cloud.ibm.com"},
            project_id=project_id,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 600,
                GenParams.TEMPERATURE: 0.1,
            }
        )
    except Exception as e:
        st.error(f"AI setup failed: {e}")
        return None

ai_model = get_ai_model()

# =========================================================
# BANK DETECTION LOGIC
# =========================================================
def detect_bank(text):
    text = text.lower()
    if "triangle" in text or "canadian tire" in text:
        return "TRIANGLE"
    if "royal bank" in text or "rbc" in text:
        if "operating loan" in text or "business loan" in text:
            return "RBC_LOAN"
        return "RBC_BUSINESS"
    return "GENERIC"

# =========================================================
# SPECIALIZED PARSERS
# =========================================================

def parse_rbc_business(file):
    """Specific geometric parser for RBC Business tables."""
    data = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            table = page.extract_table(table_settings={
                "vertical_strategy": "text", 
                "horizontal_strategy": "text",
                "snap_y_tolerance": 5,
            })
            if not table: continue
            
            for row in table:
                # Clean row: remove None and empty strings
                row = [str(item).strip() if item else "" for item in row]
                # Look for rows that start with a date (e.g., '11 Dec' or '02 Jan')
                if len(row) >= 4 and re.match(r"^\d{1,2}\s[a-zA-Z]{3}", row[0]):
                    date, desc = row[0], row[1]
                    debit = row[2].replace(",", "")
                    credit = row[3].replace(",", "")
                    
                    try:
                        if debit:
                            amt = -float(debit)
                        elif credit:
                            amt = float(credit)
                        else:
                            continue
                        data.append([date, desc, amt])
                    except ValueError:
                        continue
    return pd.DataFrame(data, columns=["Date", "Description", "Amount"])

def parse_triangle_ai(file, model):
    """AI parser with strict sign-logic for Triangle Mastercard."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    prompt = f"""
    Extract transactions from this Canadian Tire Triangle Mastercard statement.
    
    SIGN LOGIC IS CRITICAL:
    1. Look for the 'Transactions' section.
    2. PAYMENTS/CREDITS are usually shown with a MINUS sign (e.g. -500.00). These are INCOME. Extract as POSITIVE (+).
    3. PURCHASES/INTEREST are shown as POSITIVE. These are SPENDING. Extract as NEGATIVE (-).
    
    Return ONLY a markdown table: Date | Description | Amount
    Text:
    {text[:6000]}
    """
    return call_ai_for_table(prompt, model)

def call_ai_for_table(prompt, model):
    try:
        response = model.generate(prompt)
        raw = response["results"][0]["generated_text"]
        rows = []
        for line in raw.split("\n"):
            if "|" in line and "Date" not in line and "---" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    try:
                        amt = float(re.sub(r"[^\d.-]", "", parts[2]))
                        rows.append([parts[0], parts[1], amt])
                    except: continue
        return pd.DataFrame(rows, columns=["Date", "Description", "Amount"])
    except:
        return pd.DataFrame()

# =========================================================
# MAIN PROCESSING ENGINE
# =========================================================
def process_file(file, model):
    with pdfplumber.open(file) as pdf:
        header = pdf.pages[0].extract_text()
    
    bank = detect_bank(header)
    st.sidebar.info(f"Processing: {bank} ({file.name})")

    if bank == "RBC_BUSINESS":
        df = parse_rbc_business(file)
    elif bank == "TRIANGLE":
        df = parse_triangle_ai(file, model)
    else:
        # Fallback for Generic/RBC Loan
        prompt = f"Extract transactions from this {bank} statement. Spending negative, Income positive. \nText: {header[:4000]}"
        df = call_ai_for_table(prompt, model)

    if not df.empty:
        df["Category"] = df["Description"].apply(categorize)
    return df

# =========================================================
# CATEGORIZATION & UTILS (Rules + Learning)
# =========================================================
rules = {
    "uber": "Transportation", "amazon": "Shopping", "walmart": "Shopping",
    "costco": "Groceries", "metro": "Groceries", "starbucks": "Food",
    "interest": "Interest Charge", "fee": "Fees", "payment": "Bill Payment"
}

def categorize(desc):
    desc_lower = desc.lower()
    for key, cat in rules.items():
        if key in desc_lower: return cat
    return "Other"

# =========================================================
# STREAMLIT UI
# =========================================================
st.title("🏦 Universal Bank AI Analyzer")

uploaded_files = st.file_uploader("Upload Statements", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_dfs = []
    for f in uploaded_files:
        res = process_file(f, ai_model)
        if not res.empty:
            all_dfs.append(res)
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        st.subheader("Extracted Transactions")
        st.dataframe(final_df, use_container_width=True)
        
        # Financial Metrics
        spending = final_df[final_df["Amount"] < 0]["Amount"].sum()
        income = final_df[final_df["Amount"] > 0]["Amount"].sum()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Spending", f"${abs(spending):,.2f}", delta_color="inverse")
        c2.metric("Total Income / Payments", f"${income:,.2f}")
        c3.metric("Net Flow", f"${income + spending:,.2f}")
        
        st.bar_chart(final_df[final_df["Amount"] < 0].groupby("Category")["Amount"].sum().abs())
    else:
        st.error("No transactions could be parsed. Check PDF format.")
