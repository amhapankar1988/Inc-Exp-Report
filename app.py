import streamlit as st
import pdfplumber
import pandas as pd
import io
import re
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =========================================================
# 1. AI CONFIGURATION (Fixed Indentation & API Logic)
# =========================================================
def get_ai_model():
    try:
        api_key = st.secrets["WATSONX_APIKEY"].strip()
        project_id = st.secrets["WATSONX_PROJECT_ID"].strip()
        url = "https://ca-tor.ml.cloud.ibm.com"

        # Define parameters correctly
        parameters = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 1000,
            GenParams.TEMPERATURE: 0,
        }

        # Use WatsonxLLM for base models to avoid 'function' errors
        return WatsonxLLM(
            model_id="meta-llama/llama-3-1-8b",
            url=url,
            project_id=project_id,
            apikey=api_key,
            params=parameters
        )
    except Exception as e:
        st.error(f"Watsonx Config Error: {e}")
        return None

# =========================================================
# 2. DATA EXTRACTION & CLASSIFICATION
# =========================================================

def extract_raw_text(file):
    text_content = ""
    filename = file.name.lower()
    try:
        if filename.endswith('.pdf'):
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text_content += (page.extract_text() or "") + "\n"
        elif filename.endswith('.csv'):
            text_content = pd.read_csv(file).to_string()
        else:
            text_content = pd.read_excel(file).to_string()
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
    return text_content

def ai_classify_transactions(raw_text, model, user_context=""):
    categories = [
        "Utilities", "Shopping", "Health and Fitness", "Interest Charge", 
        "Overdraft Fee", "NSF", "Transportation", "Fees and Charges", 
        "Food and Dining", "Groceries", "Entertainment", "Mortgage", 
        "Withdrawal", "Deposits", "Other"
    ]
    
    # Prompt structured for Llama-3-1 Base
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Canadian banking expert. Extract transactions from the text into a table.
Format: Date | Description | Amount | Category
Rules:
- Categories: {", ".join(categories)}
- Spending is negative (e.g., -50.00). Deposits are positive.
- {user_context}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Statement Text:
{raw_text[:4000]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Date | Description | Amount | Category
"""
    
    try:
        # Get raw response string
        response = model.invoke(prompt) 
        
        # Parse text into DataFrame
        lines = response.strip().split('\n')
        data = []
        for line in lines:
            if '|' in line and "Date" not in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    data.append(parts[:4])
        
        return pd.DataFrame(data, columns=["Date", "Description", "Amount", "Category"])
    except Exception as e:
        st.error(f"AI Error: {e}")
        return pd.DataFrame()

# =========================================================
# 3. STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Universal Bank AI", layout="wide")
st.title("🏦 Universal Bank AI Analyzer")

if "context" not in st.session_state: 
    st.session_state.context = ""

with st.sidebar:
    st.header("🧠 AI Rules")
    new_rule = st.text_area("Example: 'Starbucks is Food and Dining'")
    if st.button("Update Rules"):
        st.session_state.context += f" {new_rule}."
        st.success("Rule Saved")

files = st.file_uploader("Upload Bank Statements", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

if files:
    ai_model = get_ai_model()
    if ai_model:
        all_results = []
        for f in files:
            with st.spinner(f"AI Analyzing {f.name}..."):
                raw_text = extract_raw_text(f)
                df = ai_classify_transactions(raw_text, ai_model, st.session_state.context)
                if not df.empty: 
                    all_results.append(df)
        
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            
            # Clean numeric data
            final_df["Amount"] = final_df["Amount"].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            final_df["Amount"] = pd.to_numeric(final_df["Amount"], errors='coerce').fillna(0.0)
            
            st.subheader("Classified Transactions")
            st.dataframe(final_df, use_container_width=True)
            
            # Analytics
            c1, c2 = st.columns(2)
            with c1:
                st.write("### Spending by Category")
                spent = final_df[final_df["Amount"] < 0].groupby("Category")["Amount"].sum().abs()
                st.bar_chart(spent)
            with c2:
                st.write("### Quick Stats")
                total_spent = final_df[final_df["Amount"] < 0]["Amount"].sum()
                st.metric("Total Expenses", f"${abs(total_spent):,.2f}")
                fees = final_df[final_df["Category"].str.contains("Fee|NSF|Interest", case=False, na=False)]["Amount"].sum()
                st.metric("Bank Fees", f"${abs(fees):,.2f}")
