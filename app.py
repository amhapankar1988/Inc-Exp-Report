import streamlit as st
import pdfplumber
import pandas as pd
import io
import re
from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =========================================================
# 1. AI CONFIGURATION (Using Llama-3.1-8b for Speed/Free Tier)
# =========================================================
def get_ai_model():
    try:
        # These must be set in your Streamlit Secrets
        api_key = st.secrets["WATSONX_APIKEY"].strip()
        project_id = st.secrets["WATSONX_PROJECT_ID"].strip()
        url = "https://ca-tor.ml.cloud.ibm.com"

        return ChatWatsonx(
            model_id="meta-llama/llama-3-1-8b", 
            url=url,
            project_id=project_id,
            apikey=api_key,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 1000,
                GenParams.TEMPERATURE: 0
            },
        )
    except Exception as e:
        st.error(f"Watsonx Connection Error: {e}")
        return None

# =========================================================
# 2. THE NEW "TEXT-TO-DATA" EXTRACTION LOGIC
# =========================================================

def extract_raw_text(file):
    """Extracts all text regardless of format to be analyzed by AI."""
    text_content = ""
    if file.name.lower().endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
    elif file.name.lower().endswith('.csv'):
        df = pd.read_csv(file)
        text_content = df.to_string()
    else:
        df = pd.read_excel(file)
        text_content = df.to_string()
    return text_content

def ai_classify_transactions(raw_text, model):
    """The AI directly extracts and categorizes from the raw text block."""
    categories = [
        "Utilities", "Shopping", "Health and Fitness", "Interest Charge", 
        "Overdraft Fee", "NSF", "Transportation", "Fees and Charges", 
        "Food and Dining", "Groceries", "Entertainment", "Mortgage", 
        "Withdrawal", "Deposits", "Other"
    ]
    
    prompt = f"""
    [INST] You are a Canadian Banking Expert. Analyze the following bank statement text.
    Extract every transaction and return it in a table format using the pipe symbol (|).
    
    Categories to use: {", ".join(categories)}
    
    Rules:
    1. Identify Date, Description, Amount, and Category.
    2. Debits/Spending must be negative numbers.
    3. Deposits/Income must be positive numbers.
    4. Ignore headers, footers, and balances.
    
    Format: Date | Description | Amount | Category
    
    Statement Text:
    {raw_text[:4000]} 
    [/INST]
    """
    
    try:
        response = model.invoke(prompt)
        lines = response.content.strip().split('\n')
        data = []
        for line in lines:
            if '|' in line and "Date" not in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    data.append(parts[:4])
        return pd.DataFrame(data, columns=["Date", "Description", "Amount", "Category"])
    except Exception as e:
        st.error(f"AI Classification Error: {e}")
        return pd.DataFrame()

# =========================================================
# 3. UI WORKFLOW
# =========================================================

st.set_page_config(page_title="AI Bank Intelligence", layout="wide")
st.title("🏦 Universal Canadian Bank Analyzer")
st.markdown("Upload any PDF/CSV/Excel from RBC, TD, Scotiabank, BMO, CIBC, or Canadian Tire.")

# Sidebar for Learned Rules
if "learned_context" not in st.session_state:
    st.session_state.learned_context = []

with st.sidebar:
    st.header("🧠 AI Training Context")
    st.info("The AI uses industry-standard categories, but you can add specific context below.")
    context_input = st.text_area("Add mapping (e.g., 'Internal transfer 5512 is Mortgage')")
    if st.button("Update AI Memory"):
        st.session_state.learned_context.append(context_input)
        st.success("Context Embedded!")

# File Upload
uploaded_files = st.file_uploader("Upload Statements", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    model = get_ai_model()
    if model:
        all_data = []
        for f in uploaded_files:
            with st.spinner(f"AI Analyzing {f.name}..."):
                raw_text = extract_raw_text(f)
                # Adding user context to the prompt
                if st.session_state.learned_context:
                    raw_text = "User Notes: " + " ".join(st.session_state.learned_context) + "\n" + raw_text
                
                df = ai_classify_transactions(raw_text, model)
                if not df.empty:
                    all_data.append(df)
        
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Clean numeric data
            final_df["Amount"] = final_df["Amount"].str.replace(r'[^\d.-]', '', regex=True)
            final_df["Amount"] = pd.to_numeric(final_df["Amount"], errors='coerce').fillna(0.0)

            st.subheader("📊 Classified Transactions")
            st.dataframe(final_df, use_container_width=True)

            # Analytics
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Spending by Category")
                summary = final_df[final_df["Amount"] < 0].groupby("Category")["Amount"].sum().abs()
                st.bar_chart(summary)
            
            with col2:
                st.write("### Total Summary")
                total_spent = final_df[final_df["Amount"] < 0]["Amount"].sum()
                total_deposit = final_df[final_df["Amount"] > 0]["Amount"].sum()
                st.metric("Total Spending", f"${abs(total_spent):,.2f}")
                st.metric("Total Deposits", f"${total_deposit:,.2f}")

            st.download_button("Download CSV", final_df.to_csv(index=False), "classified_bank_data.csv")
