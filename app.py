import streamlit as st
import pandas as pd
import pdfplumber
import re
import os
import joblib
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Universal Bank AI", layout="wide")

MODEL_PATH = "category_training.pkl"

# =========================================================
# LOAD TRAINED RULES
# =========================================================
def load_training():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return {}

def save_training(data):
    joblib.dump(data, MODEL_PATH)

trained_rules = load_training()

# =========================================================
# AI MODEL
# =========================================================
@st.cache_resource
def get_ai_model():
    try:
        api_key = st.secrets["WATSONX_APIKEY"].strip()
        project_id = st.secrets["WATSONX_PROJECT_ID"].strip()

        return ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            credentials={
                "apikey": api_key,
                "url": "https://ca-tor.ml.cloud.ibm.com",
            },
            project_id=project_id,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 400,
                GenParams.TEMPERATURE: 0.1,
            },
        )
    except Exception as e:
        st.error(f"AI setup failed: {e}")
        return None

ai_model = get_ai_model()

# =========================================================
# BANK DETECTION
# =========================================================
def detect_bank(text):
    text = text.lower()

    if "triangle mastercard" in text:
        return "TRIANGLE"

    if "business account statement" in text:
        return "RBC_BUSINESS"

    if "operating loan statement" in text:
        return "RBC_LOAN"

    return "GENERIC"

# =========================================================
# TRIANGLE PARSER (FIXED)
# =========================================================
def parse_triangle(file):
    rows = []

    with pdfplumber.open(file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages)

    pattern = re.compile(
        r"(Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep)\s+\d{1,2}.*?([A-Za-z0-9#\-\s]+?)\s(-?\d+\.\d{2})"
    )

    matches = pattern.findall(text)

    for m in matches:
        desc = m[1].strip()
        amt = float(m[2])

        # Critical logic
        if amt < 0:
            amt = abs(amt)  # payments = income
        else:
            amt = -amt  # purchases = spending

        rows.append(["", desc, amt])

    return pd.DataFrame(rows, columns=["Date", "Description", "Amount"])

# =========================================================
# RBC BUSINESS PARSER (FIXED)
# =========================================================
def parse_rbc_business(file):
    rows = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text().split("\n")

            for line in text:
                match = re.search(
                    r"(\d{1,2}\s\w{3})\s+(.*?)\s+(-?\d+\.\d{2})?\s+(-?\d+\.\d{2})",
                    line,
                )

                if match:
                    date = match.group(1)
                    desc = match.group(2)

                    debit = match.group(3)
                    credit = match.group(4)

                    if debit:
                        amount = -float(debit)
                    else:
                        amount = float(credit)

                    rows.append([date, desc, amount])

    return pd.DataFrame(rows, columns=["Date", "Description", "Amount"])

# =========================================================
# RBC LOAN PARSER
# =========================================================
def parse_rbc_loan(file):
    rows = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text().split("\n")

            for line in text:
                match = re.search(
                    r"(Dec|Jan)\s+\d{1,2}.*?([A-Za-z\s\-]+)\s(-?\$?\d+\.\d{2})",
                    line,
                )

                if match:
                    desc = match.group(2).strip()
                    amt = float(match.group(3).replace("$", ""))

                    rows.append(["", desc, amt])

    return pd.DataFrame(rows, columns=["Date", "Description", "Amount"])

# =========================================================
# AI FALLBACK
# =========================================================
def ai_parse(text, model):
    prompt = f"""
Extract transactions into table:
Date | Description | Amount
Income positive, expenses negative.

{text[:5000]}
"""

    try:
        result = model.generate(prompt)
        raw = result["results"][0]["generated_text"]

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
    "esso": "Fuel",
    "freshco": "Groceries",
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

    # trained rules first
    for key, val in trained_rules.items():
        if key in d:
            return val

    for key, val in base_rules.items():
        if key in d:
            return val

    return "Other"

# =========================================================
# MAIN PROCESS
# =========================================================
def process_file(file):
    with pdfplumber.open(file) as pdf:
        header = pdf.pages[0].extract_text()

    bank = detect_bank(header)
    st.sidebar.info(f"Detected: {bank}")

    if bank == "TRIANGLE":
        df = parse_triangle(file)

    elif bank == "RBC_BUSINESS":
        df = parse_rbc_business(file)

    elif bank == "RBC_LOAN":
        df = parse_rbc_loan(file)

    else:
        df = ai_parse(header, ai_model)

    if not df.empty:
        df["Category"] = df["Description"].apply(categorize)

    return df

# =========================================================
# SIDEBAR TRAINING
# =========================================================
st.sidebar.header("Train Categorization")

new_keyword = st.sidebar.text_input("Keyword")
new_category = st.sidebar.text_input("Category")

if st.sidebar.button("Add Rule"):
    trained_rules[new_keyword.lower()] = new_category
    save_training(trained_rules)
    st.sidebar.success("Rule saved!")

# =========================================================
# UI
# =========================================================
st.title("🏦 Universal Bank AI Analyzer")

files = st.file_uploader(
    "Upload Statements",
    type=["pdf"],
    accept_multiple_files=True
)

if files:
    dfs = []

    for f in files:
        df = process_file(f)
        if not df.empty:
            dfs.append(df)

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)

        st.subheader("Transactions")
        st.dataframe(final_df, use_container_width=True)

        spending = final_df[final_df.Amount < 0].Amount.sum()
        income = final_df[final_df.Amount > 0].Amount.sum()

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Spending", f"${abs(spending):,.2f}")
        c2.metric("Income / Payments", f"${income:,.2f}")
        c3.metric("Net", f"${income + spending:,.2f}")

        st.bar_chart(
            final_df[final_df.Amount < 0]
            .groupby("Category")["Amount"]
            .sum()
            .abs()
        )

        # TRAIN FROM "OTHER"
        st.sidebar.subheader("Auto-train from Other")

        others = final_df[final_df.Category == "Other"]["Description"].unique()

        if len(others) > 0:
            selected = st.sidebar.selectbox("Uncategorized", others)

            new_cat = st.sidebar.text_input("Map to category")

            if st.sidebar.button("Train AI Mapping"):
                trained_rules[selected.lower()] = new_cat
                save_training(trained_rules)
                st.sidebar.success("AI learned new mapping!")

    else:
        st.error("No transactions extracted.")
