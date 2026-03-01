import streamlit as st
import pdfplumber
import pandas as pd
import re
from collections import defaultdict

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Universal Bank AI", layout="wide")

# =========================================================
# AI MODEL
# =========================================================
@st.cache_resource
def get_ai_model():
    try:
        api_key = st.secrets["WATSONX_APIKEY"].strip()
        project_id = st.secrets["WATSONX_PROJECT_ID"].strip()

        parameters = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 600,
            GenParams.TEMPERATURE: 0,
        }

        model = ModelInference(
            model_id="meta-llama/llama-3-1-8b-instruct",
            credentials={
                "apikey": api_key,
                "url": "https://ca-tor.ml.cloud.ibm.com",
            },
            project_id=project_id,
            params=parameters
        )

        return model

    except Exception as e:
        st.error(f"AI setup failed: {e}")
        return None


# =========================================================
# CATEGORY ENGINE (FAST RULE ENGINE)
# =========================================================
category_rules = {
    "starbucks": "Food and Dining",
    "tim hortons": "Food and Dining",
    "uber": "Transportation",
    "lyft": "Transportation",
    "metro": "Groceries",
    "walmart": "Shopping",
    "amazon": "Shopping",
    "netflix": "Entertainment",
    "spotify": "Entertainment",
    "shell": "Transportation",
    "esso": "Transportation",
}


def rule_based_category(desc):
    desc = desc.lower()

    for key, val in category_rules.items():
        if key in desc:
            return val

    if "interest" in desc:
        return "Interest Charge"

    if "fee" in desc:
        return "Fees and Charges"

    if "deposit" in desc:
        return "Deposits"

    return None


# =========================================================
# TEXT EXTRACTION
# =========================================================
def extract_text_from_pdf(file):
    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

    return text


# =========================================================
# REGEX TRANSACTION PARSER (VERY IMPORTANT)
# =========================================================
def regex_extract_transactions(text):
    pattern = r'(\d{2}[-/]\d{2}[-/]\d{2,4})\s+(.*?)\s+(-?\$?\d+\.\d{2})'
    matches = re.findall(pattern, text)

    data = []

    for date, desc, amount in matches:
        cat = rule_based_category(desc) or "Other"
        data.append([date, desc, amount, cat])

    df = pd.DataFrame(
        data,
        columns=["Date", "Description", "Amount", "Category"]
    )

    return df


# =========================================================
# AI CLASSIFIER (USED ONLY WHEN NEEDED)
# =========================================================
def ai_classify_remaining(df, model, context):

    uncategorized = df[df["Category"] == "Other"]

    if uncategorized.empty:
        return df

    text_block = "\n".join(
        uncategorized["Description"].astype(str).tolist()
    )

    prompt = f"""
You categorize bank transactions in Canada.

Categories:
Food and Dining
Groceries
Transportation
Shopping
Entertainment
Utilities
Mortgage
Fees and Charges
Interest Charge
Health and Fitness
Other

Rules:
{context}

Transactions:
{text_block}

Return format:
Description | Category
"""

    try:
        response = model.generate(prompt)
        result = response["results"][0]["generated_text"]

        mapping = {}

        for line in result.split("\n"):
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    mapping[parts[0].strip()] = parts[1].strip()

        df["Category"] = df.apply(
            lambda row: mapping.get(row["Description"], row["Category"]),
            axis=1
        )

    except Exception as e:
        st.warning("AI categorization fallback used")

    return df


# =========================================================
# MAIN PROCESSOR
# =========================================================
def process_file(file, model, context):
    name = file.name.lower()

    if name.endswith("pdf"):
        raw_text = extract_text_from_pdf(file)
        df = regex_extract_transactions(raw_text)

    elif name.endswith("csv"):
        df = pd.read_csv(file)

    else:
        df = pd.read_excel(file)

    # Clean amounts
    df["Amount"] = (
        df["Amount"]
        .astype(str)
        .str.replace(r"[^\d.-]", "", regex=True)
    )

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    df = ai_classify_remaining(df, model, context)

    return df


# =========================================================
# UI
# =========================================================
st.title("🏦 Universal Bank AI Analyzer (Pro Version)")

if "context" not in st.session_state:
    st.session_state.context = ""

with st.sidebar:
    st.header("AI Learning Rules")

    rule = st.text_area("Example: Costco is Groceries")

    if st.button("Save Rule"):
        st.session_state.context += f"\n{rule}"
        st.success("AI learned new rule")

files = st.file_uploader(
    "Upload statements",
    type=["pdf", "csv", "xlsx"],
    accept_multiple_files=True
)

if files:
    model = get_ai_model()

    results = []

    progress = st.progress(0)

    for i, f in enumerate(files):
        with st.spinner(f"Processing {f.name}"):
            df = process_file(f, model, st.session_state.context)
            results.append(df)

        progress.progress((i + 1) / len(files))

    final_df = pd.concat(results)

    st.subheader("Transactions")
    st.dataframe(final_df, use_container_width=True)

    # =========================================================
    # ANALYTICS
    # =========================================================
    col1, col2, col3 = st.columns(3)

    total_spent = final_df[final_df["Amount"] < 0]["Amount"].sum()
    total_income = final_df[final_df["Amount"] > 0]["Amount"].sum()

    with col1:
        st.metric("Total Spending", f"${abs(total_spent):,.2f}")

    with col2:
        st.metric("Total Income", f"${total_income:,.2f}")

    with col3:
        st.metric("Net", f"${(total_income + total_spent):,.2f}")

    st.subheader("Spending by Category")

    spending = (
        final_df[final_df["Amount"] < 0]
        .groupby("Category")["Amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
    )

    st.bar_chart(spending)
