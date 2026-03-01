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

        # Correcting the class: Use WatsonxLLM for base models like llama-3-1-8b
    parameters={
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 1000,
        GenParams.TEMPERATURE: 0,
    }
            # IMPORTANT: Use WatsonxLLM for base models like Llama 3.1 8B
    return WatsonxLLM(
        model_id="meta-llama/llama-3-1-8b",
        url="https://ca-tor.ml.cloud.ibm.com",
        project_id=st.secrets["WATSONX_PROJECT_ID"],
        apikey=st.secrets["WATSONX_APIKEY"],
        params=parameters
    )
        )
    except Exception as e:
        st.error(f"Watsonx Config Error: {e}")
        return None

# =========================================================
# 2. DATA EXTRACTION LOGIC
# =========================================================

def extract_raw_text(file):
    text_content = ""
    filename = file.name.lower()
    if filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text_content += (page.extract_text() or "") + "\n"
    elif filename.endswith('.csv'):
        text_content = pd.read_csv(file).to_string()
    else:
        text_content = pd.read_excel(file).to_string()
    return text_content

def ai_classify_transactions(raw_text, model, user_context=""):
    categories = [
        "Utilities", "Shopping", "Health and Fitness", "Interest Charge", 
        "Overdraft Fee", "NSF", "Transportation", "Fees and Charges", 
        "Food and Dining", "Groceries", "Entertainment", "Mortgage", 
        "Withdrawal", "Deposits", "Other"
    ]
    
    # Llama 3.1 Specific Prompt Template
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a Canadian banking expert. Extract all transactions from the text into a table.
    Columns: Date | Description | Amount | Category
    Rules:
    - Categories: {", ".join(categories)}
    - Debits (spending) are negative numbers.
    - Credits (income/deposits) are positive numbers.
    - {user_context}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Statement Text:
    {raw_text[:4000]}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Date | Description | Amount | Category
    """
    
    try:
        # This calls /text/generation directly
        response = model.invoke(prompt) 
        return response
    except Exception as e:
        st.error(f"Classification failed: {e}")

# =========================================================
# 3. STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Universal Bank AI", layout="wide")
st.title("🏦 Universal Bank AI Analyzer")

if "context" not in st.session_state: st.session_state.context = ""

with st.sidebar:
    st.header("🧠 AI Rules")
    new_rule = st.text_area("Add custom categorization rule:")
    if st.button("Update AI"):
        st.session_state.context += f" {new_rule}."
        st.success("Rule Saved")

files = st.file_uploader("Upload Statements", accept_multiple_files=True)

if files:
    model = get_ai_model()
    if model:
        all_dfs = []
        for f in files:
            with st.spinner(f"Analyzing {f.name}..."):
                text = extract_raw_text(f)
                df = ai_classify_transactions(text, model, st.session_state.context)
                if not df.empty: all_dfs.append(df)
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            # Basic cleanup: remove currency symbols and convert to float
            final_df["Amount"] = final_df["Amount"].str.replace(r'[^\d.-]', '', regex=True)
            final_df["Amount"] = pd.to_numeric(final_df["Amount"], errors='coerce').fillna(0.0)
            
            st.dataframe(final_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Spending by Category")
                st.bar_chart(final_df[final_df["Amount"] < 0].groupby("Category")["Amount"].sum().abs())
            with col2:
                fees = final_df[final_df["Category"].str.contains("Fee|NSF", case=False, na=False)]["Amount"].sum()
                st.metric("Total Bank Fees", f"${abs(fees):,.2f}")
