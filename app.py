import streamlit as st
import pandas as pd
import pdfplumber
import re
import os
import joblib

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

st.set_page_config(page_title="Universal Bank AI", layout="wide")

MODEL_FILE = "trained_categories.pkl"

# =========================================================
# LOAD TRAINED RULES
# =========================================================
def load_rules():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return {}

def save_rules(rules):
    joblib.dump(rules, MODEL_FILE)

trained_rules = load_rules()

# =========================================================
# AI MODEL
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
# BANK DETECTION
# =========================================================
def detect_bank(text):
    text = text.lower()

    if "triangle mastercard" in text:
        return "TRIANGLE"

    if "royal bank of canada" in text:
        if "operating loan" in text:
            return "RBC_LOAN"
        return "RBC"

    if "td canada trust" in text:
        return "TD"

    if "bank of montreal" in text or "bmo" in text:
        return "BMO"

    if "scotiabank" in text:
        return "SCOTIA"

    if "cibc" in text:
        return "CIBC"

    if "national bank of canada" in text:
        return "NBC"

    return "GENERIC"

# =========================================================
# UNIVERSAL PDF TEXT EXTRACTOR
# =========================================================
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

# =========================================================
# UNIVERSAL TRANSACTION REGEX
# =========================================================
def extract_transactions(text):
    rows = []

    pattern = re.findall(
        r"(\d{1,2}\s[A-Za-z]{3})\s+(.*?)\s+(-?\$?\d{1,3}(?:,\d{3})*\.\d{2})",
        text,
    )

    for date, desc, amt in pattern:
        amt = float(amt.replace("$", "").replace(",", ""))
        rows.append([date, desc.strip(), amt])

    return pd.DataFrame(rows, columns=["Date", "Description", "Amount"])

# =========================================================
# TRIANGLE FIXED PARSER
# =========================================================
def parse_triangle(text):
    rows = []

    matches = re.findall(
        r"(Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep)\s+\d{1,2}.*?([A-Za-z0-9#\-\s]+?)\s(-?\d+\.\d{2})",
        text,
    )

    for m in matches:
        desc = m[1].strip()
        amt = float(m[2])

        if amt < 0:
            amt = abs(amt)
        else:
            amt = -amt

        rows.append(["", desc, amt])

    return pd.DataFrame(rows, columns=["Date", "Description", "Amount"])

# =========================================================
# RBC LOAN PARSER
# =========================================================
def parse_rbc_loan(text):
    rows = []

    matches = re.findall(
        r"(Dec|Jan)\s+\d{1,2}.*?([A-Za-z\s\-]+)\s(-?\$?\d+\.\d{2})",
        text,
    )

    for m in matches:
        desc = m[1].strip()
        amt = float(m[2].replace("$", ""))
        rows.append(["", desc, amt])

    return pd.DataFrame(rows, columns=["Date", "Description", "Amount"])

# =========================================================
# AI PARSER
# =========================================================
def ai_parse(text):
    if ai_model is None:
        return pd.DataFrame()

    prompt = f"""
Extract all bank transactions as a table:

Date | Description | Amount
Income positive
Spending negative

{text[:6000]}
"""

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
                        rows.append([p[0], p[1], amt])
                    except:
                        pass

        return pd.DataFrame(rows, columns=["Date", "Description", "Amount"])

    except:
        return pd.DataFrame()

# =========================================================
# CATEGORIZATION
# =========================================================
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
}

def categorize(desc):
    d = desc.lower()

    for k, v in trained_rules.items():
        if k in d:
            return v

    for k, v in base_rules.items():
        if k in d:
            return v

    return "Other"

# =========================================================
# CSV / EXCEL PARSER
# =========================================================
def parse_table_file(file):
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    df.columns = [c.lower() for c in df.columns]

    possible_desc = ["description", "details", "merchant"]
    possible_amt = ["amount", "amt", "value"]

    desc_col = next((c for c in df.columns if c in possible_desc), None)
    amt_col = next((c for c in df.columns if c in possible_amt), None)

    if desc_col and amt_col:
        df = df[[desc_col, amt_col]]
        df.columns = ["Description", "Amount"]
        df["Date"] = ""
        return df[["Date", "Description", "Amount"]]

    return pd.DataFrame()

# =========================================================
# PROCESS FILE
# =========================================================
def process_file(file):

    if file.name.endswith(("csv", "xlsx", "xls")):
        df = parse_table_file(file)
        return df

    text = extract_text_from_pdf(file)

    bank = detect_bank(text)
    st.sidebar.write(f"Detected Bank: {bank}")

    if bank == "TRIANGLE":
        df = parse_triangle(text)

    elif bank == "RBC_LOAN":
        df = parse_rbc_loan(text)

    else:
        df = extract_transactions(text)

        if df.empty:
            df = ai_parse(text)

    if not df.empty:
        df["Category"] = df["Description"].apply(categorize)

    return df

# =========================================================
# SIDEBAR TRAINING
# =========================================================
st.sidebar.header("Train AI Categorization")

keyword = st.sidebar.text_input("Keyword")
category = st.sidebar.text_input("Category")

if st.sidebar.button("Add Rule"):
    trained_rules[keyword.lower()] = category
    save_rules(trained_rules)
    st.sidebar.success("Saved")

# =========================================================
# UI
# =========================================================
st.title("🏦 Universal Canadian Bank Analyzer")

files = st.file_uploader(
    "Upload Statements (PDF / CSV / Excel)",
    type=["pdf", "csv", "xlsx"],
    accept_multiple_files=True,
)

if files:
    dfs = []

    for f in files:
        df = process_file(f)
        if not df.empty:
            dfs.append(df)

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)

        st.dataframe(final_df, use_container_width=True)

        spending = final_df[final_df.Amount < 0].Amount.sum()
        income = final_df[final_df.Amount > 0].Amount.sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Spending", f"${abs(spending):,.2f}")
        c2.metric("Income", f"${income:,.2f}")
        c3.metric("Net", f"${income + spending:,.2f}")

        st.bar_chart(
            final_df[final_df.Amount < 0]
            .groupby("Category")["Amount"]
            .sum()
            .abs()
        )

        # Train directly from uncategorized
        others = final_df[final_df.Category == "Other"]["Description"].unique()

        if len(others) > 0:
            st.sidebar.subheader("Train from Uncategorized")
            sel = st.sidebar.selectbox("Select", others)
            new_cat = st.sidebar.text_input("Assign Category")

            if st.sidebar.button("Train"):
                trained_rules[sel.lower()] = new_cat
                save_rules(trained_rules)
                st.sidebar.success("Learned!")

    else:
        st.error("No transactions extracted")
