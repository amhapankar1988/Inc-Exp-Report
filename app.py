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
            model_id="ibm/granite-3-8b-instruct",
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

ai_model = get_ai_model()

# =========================================================
# MERCHANT MEMORY
# =========================================================
MEMORY_FILE = "merchant_memory.pkl"

def load_memory():
    return joblib.load(MEMORY_FILE) if os.path.exists(MEMORY_FILE) else {}

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
        return merchant_memory[merchants[best_match_index]]
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
# AI PDF PARSER
# =========================================================
def ai_parse_pdf(file, model, bank="Generic"):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

    prompt = f"""
You are an expert Canadian banking assistant.

Extract all transactions from the following bank statement into a table.
Columns: Date | Description | Amount

Rules:
- Dates may be in formats: dd/mm, dd-mm, dd/mm/yyyy
- Amounts may include commas or $ signs
- Spending is negative, deposits positive
- Keep original text for Description
- Bank: {bank}

Statement text:
{text[:5000]}
Return ONLY a table in format:

Date | Description | Amount
"""
    try:
        response = model.generate(prompt)
        raw_table = response["results"][0]["generated_text"]
        rows = []
        for line in raw_table.split("\n"):
            if "|" in line and "Date" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    date, desc, amount = parts[:3]
                    amount = amount.replace("$","").replace(",","")
                    try:
                        amount = float(amount)
                    except:
                        amount = 0.0
                    rows.append([date, desc.strip(), amount, "Other"])
        return pd.DataFrame(rows, columns=["Date", "Description", "Amount", "Category"])
    except Exception as e:
        st.warning(f"AI PDF parsing failed: {e}")
        return pd.DataFrame()

# =========================================================
# AI CATEGORY IMPROVER
# =========================================================
def ai_refine_categories(df, model):
    unknown = df[df["Category"]=="Other"]
    if unknown.empty: return df
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
        df["Category"] = df.apply(lambda row: mapping.get(row["Description"], row["Category"]), axis=1)
    except:
        pass
    return df

# =========================================================
# INTELLIGENCE FEATURES
# =========================================================
subscription_keywords = ["netflix","spotify","apple","google","prime","icloud","adobe","chatgpt"]
salary_keywords = ["payroll","salary","deposit","income"]

def detect_subscriptions(df):
    df["Subscription"] = df["Description"].str.lower().apply(lambda x:any(word in x for word in subscription_keywords))
    return df

def detect_salary(df):
    df["Salary"] = df["Description"].str.lower().apply(lambda x:any(word in x for word in salary_keywords))
    return df

def detect_recurring(df):
    recurring = df.groupby("Description").size()
    recurring = recurring[recurring>2].index
    df["Recurring"] = df["Description"].isin(recurring)
    return df

def monthly_spending_trend(df):
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M")
        trend = df[df["Amount"]<0].groupby("Month")["Amount"].sum().abs()
        return trend
    except:
        return None

# =========================================================
# MERCHANT LEARNING
# =========================================================
def learn_merchants(df):
    global merchant_memory
    for _, row in df.iterrows():
        desc = row["Description"].lower()
        cat = row["Category"]
        if cat!="Other":
            merchant_memory[desc]=cat
    save_memory(merchant_memory)

# =========================================================
# PROCESS FILE
# =========================================================
def process_file(file, model):
    name = file.name.lower()
    if name.endswith("pdf"):
        text_preview=""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages[:2]:
                t = page.extract_text()
                if t: text_preview+=t
        bank = detect_bank(text_preview)
        st.sidebar.info(f"Detected Bank: {bank}")
        df = ai_parse_pdf(file, model, bank)
    elif name.endswith("csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    if df.empty: return df
    df["Amount"] = df["Amount"].astype(str).str.replace(r"[^\d.-]","",regex=True)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Category"] = df["Description"].apply(categorize)
    df = ai_refine_categories(df, model)
    return df

# =========================================================
# STREAMLIT UI
# =========================================================
st.title("🏦 Universal Bank AI Analyzer — AI PDF Parser Version")

files = st.file_uploader("Upload bank statements", type=["pdf","csv","xlsx"], accept_multiple_files=True)

if files:
    results=[]
    progress = st.progress(0)
    for i,file in enumerate(files):
        with st.spinner(f"Processing {file.name}"):
            df = process_file(file, ai_model)
            results.append(df)
        progress.progress((i+1)/len(files))
    final_df = pd.concat(results)
    if final_df.empty:
        st.warning("No transactions detected")
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
    col1,col2,col3=st.columns(3)
    expenses = final_df[final_df["Amount"]<0]["Amount"].sum()
    income = final_df[final_df["Amount"]>0]["Amount"].sum()
    col1.metric("Total Spending", f"${abs(expenses):,.2f}")
    col2.metric("Total Income", f"${income:,.2f}")
    col3.metric("Net", f"${income+expenses:,.2f}")
    # Spending chart
    spending = final_df[final_df["Amount"]<0].groupby("Category")["Amount"].sum().abs().sort_values(ascending=False)
    st.subheader("Spending by Category")
    if not spending.empty: st.bar_chart(spending)
    # Monthly trend
    trend = monthly_spending_trend(final_df)
    if trend is not None and not trend.empty:
        st.subheader("Monthly Spending Trend")
        st.line_chart(trend)
    # Subscriptions
    st.subheader("Subscriptions Detected")
    subs = final_df[final_df["Subscription"]]
    st.dataframe(subs[["Date","Description","Amount"]], width="stretch")
    # Recurring
    st.subheader("Recurring Payments")
    rec = final_df[final_df["Recurring"]]
    st.dataframe(rec[["Date","Description","Amount"]], width="stretch")
    # AI Advisor
    st.subheader("AI Financial Advisor")
    if st.button("Generate Financial Insights"):
        sample = final_df.head(50).to_string()
        prompt = f"Analyze these bank transactions and provide financial advice:\n{sample}"
        try:
            response = ai_model.generate(prompt)
            insights = response["results"][0]["generated_text"]
            st.write(insights)
        except:
            st.warning("AI insights unavailable")
