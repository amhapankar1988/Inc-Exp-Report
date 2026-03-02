import streamlit as st
import pandas as pd
import pdfplumber
import re
import os
import joblib
import numpy as np
from datetime import datetime
from difflib import get_close_matches

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

st.set_page_config(page_title="AI Financial Analyzer v4", layout="wide")

# ======================================================
# DATABASE FILES (PERSISTENT MEMORY)
# ======================================================
MERCHANT_DB = "merchant_memory.pkl"
CATEGORY_DB = "category_memory.pkl"

def load_db(path):
    if os.path.exists(path):
        return joblib.load(path)
    return {}

def save_db(path, data):
    joblib.dump(data, path)

merchant_memory = load_db(MERCHANT_DB)
category_memory = load_db(CATEGORY_DB)

# ======================================================
# AI MODEL
# ======================================================
@st.cache_resource
def load_ai():
    try:
        api_key = st.secrets["WATSONX_APIKEY"]
        project_id = st.secrets["WATSONX_PROJECT_ID"]

        return ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            credentials={"apikey": api_key,
                         "url": "https://ca-tor.ml.cloud.ibm.com"},
            project_id=project_id,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 400,
                GenParams.TEMPERATURE: 0
            }
        )
    except:
        return None

ai_model = load_ai()

# ======================================================
# BANK DETECTION ENGINE
# ======================================================
def detect_bank(text):
    t = text.lower()

    patterns = {
        "RBC": ["royal bank of canada"],
        "TD": ["td canada trust"],
        "Scotiabank": ["scotiabank"],
        "BMO": ["bank of montreal", "bmo"],
        "CIBC": ["cibc"],
        "National Bank": ["national bank"],
        "Canadian Tire Bank": ["triangle mastercard"]
    }

    for bank, keys in patterns.items():
        if any(k in t for k in keys):
            return bank

    return "Unknown"

# ======================================================
# SMART PDF EXTRACTION
# ======================================================
def extract_pdf_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def extract_transactions(text):
    rows = []

    matches = re.findall(
        r"(\d{1,2}\s[A-Za-z]{3})\s+(.*?)\s+(-?\$?\d{1,3}(?:,\d{3})*\.\d{2})",
        text
    )

    for m in matches:
        rows.append([
            m[0],
            m[1],
            float(m[2].replace("$","").replace(",",""))
        ])

    return pd.DataFrame(rows,
                        columns=["Date","Description","Amount"])

# ======================================================
# MERCHANT ENGINE
# ======================================================
def normalize_merchant(desc):

    desc_clean = desc.lower()

    # exact memory match
    for known in merchant_memory:
        if known in desc_clean:
            return merchant_memory[known]

    # fuzzy match
    matches = get_close_matches(
        desc_clean,
        merchant_memory.keys(),
        n=1,
        cutoff=0.8
    )

    if matches:
        return merchant_memory[matches[0]]

    # fallback heuristic
    return desc_clean.split()[0].title()

# ======================================================
# HYBRID CATEGORY CLASSIFIER
# ======================================================
CATEGORIES = [
"Housing / Mortgage","Utilities","Groceries","Food & Dining",
"Transportation","Fuel","Shopping","Subscriptions",
"Insurance","Health & Medical","Fitness","Travel",
"Entertainment","Education","Investments","Taxes",
"Transfers","Income","Loan Payment","Interest Charges",
"Bank Fees","Business Expenses","Other"
]

def classify_transactions(df):

    df["Merchant"] = df["Description"].apply(normalize_merchant)

    # Rule-based first
    df["Category"] = df["Merchant"].map(category_memory)

    # AI for missing
    missing = df[df["Category"].isna()]

    if not missing.empty and ai_model is not None:

        sample = missing[["Description","Amount"]].to_string(index=False)

        prompt = f"""
Classify transactions into:
{', '.join(CATEGORIES)}

Return:
Description | Category
Transactions:
{sample}
"""

        response = ai_model.generate(prompt)
        result = response["results"][0]["generated_text"]

        mapping = {}

        for line in result.split("\n"):
            if "|" in line:
                p = [x.strip() for x in line.split("|")]
                if len(p)>=2:
                    mapping[p[0]] = p[1]

        for i,row in missing.iterrows():
            df.loc[i,"Category"] = mapping.get(row["Description"],"Other")

    df["Category"].fillna("Other", inplace=True)

    return df

# ======================================================
# FRAUD / ANOMALY ENGINE
# ======================================================
def detect_anomalies(df):

    if len(df) < 5:
        df["Anomaly"] = False
        return df

    z = np.abs((df["Amount"] - df["Amount"].mean()) /
               df["Amount"].std())

    df["Anomaly"] = z > 3

    return df

# ======================================================
# FINANCIAL INTELLIGENCE ENGINE
# ======================================================
def generate_insights(df):

    insights = []

    spending = df[df.Amount < 0]["Amount"].sum()
    income = df[df.Amount > 0]["Amount"].sum()

    if income != 0:
        savings = (income + spending)/income
        insights.append(f"Savings Rate: {round(savings*100,2)}%")

    top_category = (
        df[df.Amount<0]
        .groupby("Category")["Amount"]
        .sum()
        .idxmin()
    )

    insights.append(f"Highest Spending Category: {top_category}")

    recurring = df["Merchant"].value_counts()
    recurring = recurring[recurring > 2]

    if not recurring.empty:
        insights.append("Recurring Payments Detected")

    return insights

# ======================================================
# MAIN PROCESS
# ======================================================
def process_file(file):

    if file.name.endswith(("csv","xlsx","xls")):
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        df.columns = [c.lower() for c in df.columns]

        df = df.rename(columns={
            "description":"Description",
            "amount":"Amount",
            "date":"Date"
        })

        bank = "Imported"

    else:
        text = extract_pdf_text(file)
        bank = detect_bank(text)
        df = extract_transactions(text)

    if df.empty:
        return df

    df["Bank"] = bank
    df = classify_transactions(df)
    df = detect_anomalies(df)
    df["Duplicate"] = df.duplicated(
        subset=["Date","Amount","Description"]
    )

    df["Month"] = pd.to_datetime(df["Date"],
                                  errors="coerce").dt.to_period("M")

    return df

# ======================================================
# UI DASHBOARD
# ======================================================
st.title("AI Financial Analyzer v4 – Production Fintech")

files = st.file_uploader("Upload Statements",
                         type=["pdf","csv","xlsx"],
                         accept_multiple_files=True)

if files:

    all_data = []

    for f in files:
        df = process_file(f)
        if not df.empty:
            all_data.append(df)

    if all_data:

        df = pd.concat(all_data,ignore_index=True)

        st.subheader("Normalized Dataset")
        st.dataframe(df, use_container_width=True)

        st.subheader("Category Analytics")
        st.bar_chart(
            df[df.Amount<0]
            .groupby("Category")["Amount"]
            .sum()
            .abs()
        )

        st.subheader("Monthly Cashflow")
        st.line_chart(
            df.groupby("Month")["Amount"].sum()
        )

        st.subheader("Insights")
        for i in generate_insights(df):
            st.write("•",i)

        st.download_button(
            "Download CSV Report",
            df.to_csv(index=False),
            file_name="financial_report_v4.csv"
        )
