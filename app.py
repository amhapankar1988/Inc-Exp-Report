import streamlit as st
import pdfplumber
import pandas as pd
import io
import re
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =========================================================
# 1. AI CONFIGURATION (Base Model Approach)
# =========================================================
def get_ai_model():
    try:
        api_key = st.secrets["WATSONX_APIKEY"].strip()
        project_id = st.secrets["WATSONX_PROJECT_ID"].strip()
        url = "https://ca-tor.ml.cloud.ibm.com"

        # Use WatsonxLLM (Text-to-Text) instead of ChatWatsonx
        return WatsonxLLM(
            model_id="meta-llama/llama-3-1-8b", 
            url=url,
            project_id=project_id,
            apikey=api_key,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 1500,
                GenParams.TEMPERATURE: 0
            },
        )
    except Exception as e:
        st.error(f"Watsonx Config Error: {e}. Please check your Streamlit Secrets.")
        return None

# =========================================================
# 2. DATA EXTRACTION LOGIC
# =========================================================

def extract_raw_text(file):
    """Universal text extraction for PDF, CSV, and Excel."""
    text_content = ""
    filename = file.name.lower()
    
    if filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
    elif filename.endswith('.csv'):
        df = pd.read_csv(file)
        text_content = df.to_string()
    else: # Excel
        df = pd.read_excel(file)
        text_content = df.to_string()
    return text_content

def ai_classify_transactions(raw_text, model, user_context=""):
    """
    Directly extracts and classifies from raw text using the AI model.
    """
    categories = [
        "Utilities", "Shopping", "Health and Fitness", "Interest Charge", 
        "Overdraft Fee", "NSF", "Transportation", "Fees and Charges", 
        "Food and Dining", "Groceries", "Entertainment", "Mortgage", 
        "Withdrawal", "Deposits", "Other"
    ]
    
    # We use a completion prompt structure for Llama base models
    prompt = f"""[INST] Analyze the Canadian bank statement text below.
Extract all transactions and categorize them.

Categories to use: {", ".join(categories)}

Rules:
1. Format as a table using | as separator.
2. Debits/Payments must be negative numbers.
3. Deposits must be positive numbers.
4. If a description looks like a transfer or fee, use specific categories like "Withdrawal" or "Fees and Charges".

User Specific Rules: {user_context}

Output format:
Date | Description | Amount | Category

Statement Text:
{raw_text[:4000]}
[/INST]

Date | Description | Amount | Category
"""
    
    try:
        # WatsonxLLM returns a string directly
        response = model.invoke(prompt)
        
        # Parse the pipe-delimited output
        lines = response.strip().split('\n')
        data = []
        for line in lines:
            if '|' in line and "Date" not in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    data.append(parts[:4])
        return pd.DataFrame(data, columns=["Date", "Description", "Amount", "Category"])
    except Exception as e:
        st.error(f"Classification failed: {e}")
        return pd.DataFrame()

# =========================================================
# 3. STREAMLIT UI & WORKFLOW
# =========================================================

st.set_page_config(page_title="AI Bank Intelligence", layout="wide")
st.title("🏦 Universal Canadian Bank AI Analyzer")
st.info("Directly classifying RBC, TD, BMO, Scotiabank, CIBC, and Canadian Tire statements using IBM Watsonx.")

# AI Memory / Learning Section
if "context" not in st.session_state:
    st.session_state.context = ""

with st.sidebar:
    st.header("🧠 AI Training & Fine-Tuning")
    st.write("Add specific rules to help the AI learn your spending patterns.")
    new_rule = st.text_area("Example: 'Transactions to Lucky Mart are Groceries'", height=100)
    if st.button("Train AI"):
        st.session_state.context += f" {new_rule}."
        st.success("AI Knowledge Updated!")

# 1. Upload All Formats
files = st.file_uploader("Upload bank statements (PDF, CSV, Excel)", accept_multiple_files=True)

if files:
    model = get_ai_model()
    if model:
        full_df_list = []
        
        for f in files:
            with st.spinner(f"AI Analyzing {f.name}..."):
                # 2. Embed/Learn context and extract raw text
                raw_text = extract_raw_text(f)
                
                # 3. Analyze and Categorize
                df = ai_classify_transactions(raw_text, model, st.session_state.context)
                if not df.empty:
                    full_df_list.append(df)
        
        if full_df_list:
            final_df = pd.concat(full_df_list, ignore_index=True)
            
            # Numeric cleanup for calculations
            final_df["Amount"] = final_df["Amount"].str.replace(r'[^\d.-]', '', regex=True)
            final_df["Amount"] = pd.to_numeric(final_df["Amount"], errors='coerce').fillna(0.0)

            st.subheader("📋 Categorized Transactions")
            st.dataframe(final_df, use_container_width=True)

            # Industry-Standard Insights
            c1, c2 = st.columns(2)
            with c1:
                st.write("### Spending by Category")
                spending = final_df[final_df["Amount"] < 0].groupby("Category")["Amount"].sum().abs()
                st.bar_chart(spending)
            
            with c2:
                st.write("### Key Financial Metrics")
                total_spent = final_df[final_df["Amount"] < 0]["Amount"].sum()
                total_in = final_df[final_df["Amount"] > 0]["Amount"].sum()
                
                st.metric("Total Expenses", f"${abs(total_spent):,.2f}")
                st.metric("Total Income", f"${total_in:,.2f}")
                
                # Highlight Fees specifically as requested
                fees = final_df[final_df["Category"].isin(["NSF", "Overdraft Fee", "Fees and Charges"])]["Amount"].sum()
                st.metric("Bank Fees Paid", f"${abs(fees):,.2f}", delta_color="inverse")

            st.download_button("Export to Excel/CSV", final_df.to_csv(index=False), "categorized_finances.csv")
