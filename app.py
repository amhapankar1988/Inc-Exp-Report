import streamlit as st
import pandas as pd
import pdfplumber
import os
import io
from langchain_ibm import ChatWatsonx

# --- 1. SECURE INITIALIZATION ---
def get_llm():
    api_key = os.getenv("WATSONX_APIKEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    # Try the explicit regional endpoint
    url = "https://ca-tor.ml.cloud.ibm.com"

    if not api_key or not project_id:
        st.error(f"Missing Credentials: APIKEY={bool(api_key)}, PID={bool(project_id)}")
        st.stop()

    try:
        # We explicitly pass the credentials to ChatWatsonx
        return ChatWatsonx(
            model_id="meta-llama/llama-3-3-70b-instruct",
            url=url,
            apikey=api_key, # Use 'apikey' (lowercase) for ChatWatsonx
            project_id=project_id,
            params={
                "decoding_method": "greedy",
                "max_new_tokens": 1000,
            }
        )
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        st.stop()

# --- 2. CANADIAN BANK DATA NORMALIZATION ---
def standardize_df(df):
    """Standardizes headers for BMO, RBC, Scotia, and Canadian Tire."""
    df.columns = [str(c).strip().lower() for c in df.columns]
    mapping = {
        'date': ['date', 'transaction date', 'posted date', 'txn date', 'trans date'],
        'description': ['description', 'desc', 'transaction', 'memo', 'merchant', 'details'],
        'amount': ['amount', 'debit', 'credit', 'value', 'amount ($)']
    }
    
    new_cols = {}
    for standard, synonyms in mapping.items():
        for col in df.columns:
            if col in synonyms:
                new_cols[col] = standard.capitalize()
    
    df = df.rename(columns=new_cols)
    
    if 'Amount' in df.columns:
        # Handle Canadian currency formatting: $1,200.00 or (50.00)
        df['Amount'] = df['Amount'].astype(str).replace(r'[\$,]', '', regex=True)
        df['Amount'] = df['Amount'].replace(r'\((.*)\)', r'-\1', regex=True)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
    return df

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="Canada AI Finance", layout="wide")
st.title("🇨🇦 Canadian Bank Multi-Statement Analyzer")
st.info("Connected to Watsonx Toronto Region (ca-tor)")

files = st.file_uploader("Upload BMO, RBC, Scotia, or Canadian Tire statements", 
                         type=["pdf", "csv", "xlsx"], 
                         accept_multiple_files=True)

if files:
    if st.button("🚀 Run AI Analysis"):
        llm = get_llm()
        master_df = pd.DataFrame()
        
        for file in files:
            with st.spinner(f"Brain is reading {file.name}..."):
                ext = file.name.split('.')[-1].lower()
                
                if ext in ['csv', 'xlsx', 'xls']:
                    df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
                    df = standardize_df(df)
                else:
                    # PDF Processing (BMO/RBC/Scotiabank Layouts)
                    with pdfplumber.open(file) as pdf:
                        text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
                        pdf_prompt = f"""Extract transactions from this text: {text[:5000]}
                        Return ONLY a Markdown table: Date | Description | Amount
                        Use YYYY-MM-DD for dates. Use negative numbers for expenses."""
                        res = llm.invoke(pdf_prompt)
                        # Minimal table parser for LLM response
                        rows = [line.split('|') for line in res.content.split('\n') if '|' in line and '---' not in line]
                        df = pd.DataFrame([r[1:4] for r in rows[1:]], columns=['Date','Description','Amount'])
                        df = standardize_df(df)
                
                # AI Categorization
                if not df.empty and 'Description' in df.columns:
                    unique_merchants = df['Description'].dropna().unique()[:40]
                    cat_prompt = f"Categorize these Canadian merchants: {unique_merchants}. Return 'Merchant | Category' only."
                    cat_res = llm.invoke(cat_prompt)
                    
                    cat_map = {}
                    for line in cat_res.content.split('\n'):
                        if '|' in line:
                            parts = line.split('|')
                            cat_map[parts[0].strip()] = parts[1].strip()
                    
                    df['Category'] = df['Description'].map(cat_map).fillna('Other')
                    df['Source'] = file.name
                    master_df = pd.concat([master_df, df], ignore_index=True)

        if not master_df.empty:
            master_df = master_df.dropna(subset=['Amount', 'Date'])
            master_df['Month'] = master_df['Date'].dt.strftime('%Y-%m')

            # --- VISUAL SUMMARY ---
            st.header("📈 Income vs. Expense Trends")
            summary = master_df.groupby('Month').agg(
                Income=('Amount', lambda x: x[x > 0].sum()),
                Expenses=('Amount', lambda x: abs(x[x < 0].sum()))
            )
            st.bar_chart(summary)

            # --- DRILL DOWN ---
            st.header("🔍 Monthly Details")
            for month in sorted(master_df['Month'].unique(), reverse=True):
                with st.expander(f"View {month}"):
                    m_data = master_df[master_df['Month'] == month]
                    st.dataframe(m_data[['Date', 'Description', 'Amount', 'Category', 'Source']], use_container_width=True)

            # --- EXPORT ---
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                master_df.to_excel(writer, index=False, sheet_name='All_Transactions')
                summary.to_excel(writer, sheet_name='Monthly_Summary')
            
            st.download_button("📥 Download Excel Report", output.getvalue(), "Canadian_Finance_Report.xlsx")
