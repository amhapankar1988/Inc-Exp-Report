import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

        model = ModelInference(
            model_id="ibm/granite-4-h-small",
            credentials={
                "apikey": api_key,
                "url": "https://ca-tor.ml.cloud.ibm.com",
            },
            project_id=project_id,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 500,
                GenParams.TEMPERATURE: 0.2,
            }
        )
        return model
    except Exception as e:
        st.error(f"AI setup failed: {e}")
        return None


# =========================================================
# MERCHANT MEMORY SYSTEM
# =========================================================
MEMORY_FILE = "merchant_memory.pkl"


def load_memory():
    if os.path.exists(MEMORY_FILE):
        return joblib.load(MEMORY_FILE)
    return {}


def save_memory(memory):
    joblib.dump(memory, MEMORY_FILE)


merchant_memory = load_memory()


def memory_categorize(description):
    if not merchant_memory:
        return None

    merchants = list(merchant_memory.keys())

    vectorizer = TfidfVectorizer().fit(merchants + [description])
    vectors = vectorizer.transform(merchants + [description])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    best_match_index = similarity.argmax()
    score = similarity[0][best_match_index]

    if score > 0.75:
        merchant = merchants[best_match_index]
        return merchant_memory[merchant]

    return None


# =========================================================
# RULE ENGINE
# =========================================================
rules = {
    "uber": "Transportation",
    "lyft": "Transportation",
    "amazon": "Shopping",
    "walmart": "Shopping",
    "costco": "Groceries",
    "metro": "Groceries",
    "tim hortons": "Food and Dining",
    "starbucks": "Food and Dining",
    "netflix": "Entertainment",
    "spotify": "Entertainment",
}


def categorize(desc):
    desc_lower = desc.lower()

    memory_cat = memory_categorize(desc_lower)
    if memory_cat:
        return memory_cat

    for key in rules:
        if key in desc_lower:
            return rules[key]

    if "interest" in desc_lower:
        return "Interest Charge"

    if "fee" in desc_lower:
        return "Fees and Charges"

    if "deposit" in desc_lower:
        return "Deposits"

    return "Other"


# =========================================================
# BANK DETECTION
# =========================================================
def detect_bank(text):
    text = text.lower()

    if "royal bank" in text or "rbc" in text:
        return "RBC"
    if "td canada trust" in text or "toronto dominion" in text:
        return "TD"
    if "scotiabank" in text:
        return "Scotiabank"
    if "cibc" in text:
        return "CIBC"
    if "bank of montreal" in text or "bmo" in text:
        return "BMO"
    if "american express" in text or "amex" in text:
        return "AMEX"

    return "Generic"


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


# =========================================================
# BANK PARSER
# =========================================================
def parse_transactions(text, bank):

    patterns = {
        "RBC": r'(\d{2}/\d{2})\s+(.*?)\s+(-?\$?\d+\.\d{2})',
        "TD": r'(\d{2}-\d{2})\s+(.*?)\s+(-?\$?\d+\.\d{2})',
        "Scotiabank": r'(\d{2}\s\w{3})\s+(.*?)\s+(-?\$?\d+\.\d{2})',
        "CIBC": r'(\d{2}/\d{2})\s+(.*?)\s+(-?\$?\d+\.\d{2})',
        "BMO": r'(\d{2}/\d{2})\s+(.*?)\s+(-?\$?\d+\.\d{2})',
        "AMEX": r'(\w{3}\s\d{2})\s+(.*?)\s+(-?\$?\d+\.\d{2})',
        "Generic": r'(\d{2}[-/]\d{2})\s+(.*?)\s+(-?\$?\d+\.\d{2})'
    }

    pattern = patterns.get(bank, patterns["Generic"])
    matches = re.findall(pattern, text)

    data = []
    for date, desc, amount in matches:
        data.append([date, desc, amount, "Other"])

    return pd.DataFrame(data, columns=["Date", "Description", "Amount", "Category"])


# =========================================================
# AI CATEGORY IMPROVER
# =========================================================
def ai_refine_categories(df, model):

    unknown = df[df["Category"] == "Other"]

    if unknown.empty:
        return df

    text = "\n".join(unknown["Description"].tolist())

    prompt = f"""
Classify bank transactions into categories:
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

Transactions:
{text}

Return:
Description | Category
"""

    try:
        response = model.generate(prompt)
        result = response["results"][0]["generated_text"]

        mapping = {}
        for line in result.split("\n"):
            if "|" in line:
                parts = line.split("|")
                mapping[parts[0].strip()] = parts[1].strip()

        df["Category"] = df.apply(
            lambda row: mapping.get(row["Description"], row["Category"]),
            axis=1
        )
    except:
        pass

    return df


# =========================================================
# INTELLIGENCE FEATURES
# =========================================================
subscription_keywords = [
    "netflix", "spotify", "apple", "google",
    "prime", "icloud", "adobe", "chatgpt"
]


def detect_subscriptions(df):
    df["Subscription"] = df["Description"].str.lower().apply(
        lambda x: any(word in x for word in subscription_keywords)
    )
    return df


salary_keywords = ["payroll", "salary", "deposit", "income"]


def detect_salary(df):
    df["Salary"] = df["Description"].str.lower().apply(
        lambda x: any(word in x for word in salary_keywords)
    )
    return df


def detect_recurring(df):
    recurring = df.groupby("Description").size()
    recurring = recurring[recurring > 2].index
    df["Recurring"] = df["Description"].isin(recurring)
    return df


def monthly_spending_trend(df):
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M")

        trend = (
            df[df["Amount"] < 0]
            .groupby("Month")["Amount"]
            .sum()
            .abs()
        )
        return trend
    except:
        return None


# =========================================================
# LEARNING SYSTEM
# =========================================================
def learn_merchants(df):
    global merchant_memory

    for _, row in df.iterrows():
        desc = row["Description"].lower()
        cat = row["Category"]

        if cat != "Other":
            merchant_memory[desc] = cat

    save_memory(merchant_memory)


# =========================================================
# MAIN PROCESSOR
# =========================================================
def process_file(file, model):

    name = file.name.lower()

    if name.endswith("pdf"):
        text = extract_text(file)
        bank = detect_bank(text)

        st.sidebar.info(f"Detected Bank: {bank}")

        df = parse_transactions(text, bank)

    elif name.endswith("csv"):
        df = pd.read_csv(file)

    else:
        df = pd.read_excel(file)

    df["Amount"] = (
        df["Amount"]
        .astype(str)
        .str.replace(r"[^\d.-]", "", regex=True)
    )

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    df["Category"] = df["Description"].apply(categorize)

    df = ai_refine_categories(df, model)

    return df


# =========================================================
# UI
# =========================================================
st.title("🏦 Universal Bank AI Analyzer — Enterprise AI Version")

files = st.file_uploader(
    "Upload bank statements",
    type=["pdf", "csv", "xlsx"],
    accept_multiple_files=True
)

if files:
    model = get_ai_model()

    if not model:
        st.stop()

    results = []
    progress = st.progress(0)

    for i, file in enumerate(files):
        with st.spinner(f"Processing {file.name}"):
            df = process_file(file, model)
            results.append(df)

        progress.progress((i + 1) / len(files))

    final_df = pd.concat(results)

    if final_df.empty:
        st.warning("No transactions detected.")
        st.stop()

    # Intelligence
    final_df = detect_subscriptions(final_df)
    final_df = detect_salary(final_df)
    final_df = detect_recurring(final_df)

    # Learning
    learn_merchants(final_df)

    st.subheader("Transactions")
    st.dataframe(final_df, width="stretch")

    # Metrics
    col1, col2, col3 = st.columns(3)

    expenses = final_df[final_df["Amount"] < 0]["Amount"].sum()
    income = final_df[final_df["Amount"] > 0]["Amount"].sum()

    col1.metric("Total Spending", f"${abs(expenses):,.2f}")
    col2.metric("Total Income", f"${income:,.2f}")
    col3.metric("Net", f"${income + expenses:,.2f}")

    # Spending chart
    spending = (
        final_df[final_df["Amount"] < 0]
        .groupby("Category")["Amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
    )

    st.subheader("Spending by Category")

    if not spending.empty:
        st.bar_chart(spending)
    else:
        st.info("No spending data available.")

    # Monthly trend
    trend = monthly_spending_trend(final_df)

    if trend is not None and not trend.empty:
        st.subheader("Monthly Spending Trend")
        st.line_chart(trend)

    # Subscriptions
    st.subheader("Subscriptions Detected")
    subs = final_df[final_df["Subscription"]]
    st.dataframe(subs[["Date", "Description", "Amount"]], width="stretch")

    # Recurring
    st.subheader("Recurring Payments")
    rec = final_df[final_df["Recurring"]]
    st.dataframe(rec[["Date", "Description", "Amount"]], width="stretch")

    # AI Advisor
    st.subheader("AI Financial Advisor")

    if st.button("Generate Financial Insights"):
        sample = final_df.head(50).to_string()

        prompt = f"""
Analyze these bank transactions and provide financial advice:

{sample}
"""

        try:
            response = model.generate(prompt)
            insights = response["results"][0]["generated_text"]
            st.write(insights)
        except:
            st.warning("AI insights unavailable.")
