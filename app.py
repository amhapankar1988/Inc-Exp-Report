import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
from datetime import datetime

# Optional AI
AI_AVAILABLE = True
try:
    from ibm_watsonx_ai.foundation_models import Model
except:
    AI_AVAILABLE = False

st.set_page_config(page_title="AI Expense Analyzer", layout="wide")
st.title("AI Bank Statement Analyzer")

# =========================================================
# CONFIG
# =========================================================

REQUIRED_COLUMNS = ["Date", "Description", "Amount", "Type", "Category"]

CATEGORY_RULES = {
    "Food": ["restaurant", "uber eats", "doordash", "pizza", "tim hortons", "mcdonald"],
    "Groceries": ["walmart", "costco", "metro", "nofrills", "freshco", "loblaws"],
    "Transport": ["uber", "lyft", "petro", "esso", "shell", "gas"],
    "Shopping": ["amazon", "canadian tire", "best buy"],
    "Bills": ["bell", "rogers", "telus", "insurance", "hydro"],
    "Entertainment": ["netflix", "spotify", "prime video"],
}

DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %b",
    "%d-%m-%Y"
]

# =========================================================
# UTILITIES
# =========================================================

def clean_amount(value):
    if value is None:
        return 0.0
    value = str(value)
    value = value.replace(",", "").replace("$", "").strip()
    try:
        return float(value)
    except:
        return 0.0


def normalize_date(date_str):
    if date_str is None:
        return None

    date_str = str(date_str).strip()

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt).date()
        except:
            continue

    try:
        return pd.to_datetime(date_str, errors="coerce").date()
    except:
        return None


def ensure_schema(df):
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[REQUIRED_COLUMNS]


# =========================================================
# UNIVERSAL TRANSACTION PATTERN
# =========================================================

transaction_pattern = re.compile(
    r"(\d{2}[/-]\d{2}|\d{2}\s[A-Za-z]{3}|\d{4}-\d{2}-\d{2})\s+(.*?)\s+(-?\$?\d+\.\d{2})"
)

# =========================================================
# PDF PARSING ENGINE
# =========================================================

def extract_transactions_from_page(page):
    rows = []

    tables = page.extract_tables()

    if tables:
        for table in tables:
            for row in table:
                if not row:
                    continue

                text = " ".join([str(x) for x in row if x])

                match = transaction_pattern.search(text)
                if match:
                    date, desc, amount = match.groups()

                    rows.append({
                        "Date": normalize_date(date),
                        "Description": desc.strip(),
                        "Amount": clean_amount(amount)
                    })

    text = page.extract_text()

    if text:
        for line in text.split("\n"):
            match = transaction_pattern.search(line)
            if match:
                date, desc, amount = match.groups()

                rows.append({
                    "Date": normalize_date(date),
                    "Description": desc.strip(),
                    "Amount": clean_amount(amount)
                })

    return rows


def parse_pdf(uploaded_file):
    rows = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            rows.extend(extract_transactions_from_page(page))

    df = pd.DataFrame(rows)

    return df


# =========================================================
# AI CLASSIFIER
# =========================================================

def ai_classify(description):

    if not AI_AVAILABLE:
        return None

    try:
        prompt = f"""
Categorize this bank transaction into one of these:
Food, Groceries, Transport, Shopping, Bills, Entertainment, Income, Other

Transaction:
{description}

Return only the category name.
"""

        model = Model(
            model_id="ibm/granite-13b-chat-v2",
            params={"temperature": 0}
        )

        result = model.generate(prompt)
        category = result["results"][0]["generated_text"].strip()

        return category

    except:
        return None


# =========================================================
# RULE CLASSIFIER
# =========================================================

def rule_classify(desc):
    if pd.isna(desc):
        return "Other"

    desc = desc.lower()

    for category, keywords in CATEGORY_RULES.items():
        for k in keywords:
            if k in desc:
                return category

    return "Other"


def classify_transactions(df):

    categories = []

    for desc in df["Description"]:

        category = ai_classify(desc)

        if category is None:
            category = rule_classify(desc)

        categories.append(category)

    df["Category"] = categories
    return df


# =========================================================
# INCOME / EXPENSE
# =========================================================

def detect_income_expense(df):
    df["Type"] = df["Amount"].apply(
        lambda x: "Income" if x > 0 else "Expense"
    )
    return df


# =========================================================
# DUPLICATE REMOVAL
# =========================================================

def remove_duplicates(df):
    return df.drop_duplicates(
        subset=["Date", "Description", "Amount"]
    )


# =========================================================
# MASTER PROCESSOR
# =========================================================

def process_statement(file):

    df = parse_pdf(file)

    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = remove_duplicates(df)
    df = detect_income_expense(df)
    df = classify_transactions(df)

    return ensure_schema(df)


# =========================================================
# UI
# =========================================================

uploaded_files = st.file_uploader(
    "Upload Bank Statements",
    type=["pdf"],
    accept_multiple_files=True
)

all_data = []

if uploaded_files:

    with st.spinner("Analyzing statements with AI..."):

        for file in uploaded_files:
            df = process_statement(file)

            if not df.empty:
                df["Source"] = file.name
                all_data.append(df)

    if all_data:

        final_df = pd.concat(all_data)

        st.subheader("Transactions")
        st.dataframe(final_df, use_container_width=True)

        st.subheader("Category Summary")

        summary = (
            final_df.groupby("Category")["Amount"]
            .sum()
            .reset_index()
        )

        st.dataframe(summary)

        final_df["Month"] = pd.to_datetime(final_df["Date"]).dt.to_period("M")

        st.subheader("Monthly Cashflow")
        monthly = final_df.groupby("Month")["Amount"].sum().reset_index()
        st.dataframe(monthly)

        # Export
        output_file = "expense_report.xlsx"
        final_df.to_excel(output_file, index=False)

        with open(output_file, "rb") as f:
            st.download_button(
                "Download Excel Report",
                f,
                file_name=output_file
            )

    else:
        st.warning("No transactions detected.")
