import streamlit as st
import pandas as pd
import pdfplumber
import os
import io
import re
from langchain_ibm import ChatWatsonx

# --- INITIALIZATION ---
def init_ai():
    api_key = os.getenv("WATSONX_APIKEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL", "https://ca-tor.ml.cloud.ibm.com")
    return ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url=url,
        apikey=api_key,
        project_id=project_id,
        params={"decoding_method": "greedy", "max_new_tokens": 1500}
    )

# --- CANADIAN BANK COLUMN MAPPER ---
def standardize_columns(df):
    """Maps common Canadian bank headers to standard internal names."""
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Synonyms for mapping
    mapping = {
        'date': ['date', 'transaction date', 'txn date', 'posted date', 'trans date'],
        'description': ['description', 'desc', 'transaction', 'memo', 'details', 'name'],
        'amount': ['amount', 'value', 'debit', 'credit', 'transaction amount', 'amount ($)']
    }
    
    final_map = {}
    for standard, synonyms in mapping.items():
        for col in df.columns:
            if col in synonyms:
                final_map[col] = standard.capitalize()
    
    return df.rename(columns=final_map)

# --- DOCUMENT EXTRACTION ---
def load_data(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'csv':
        return standardize_columns(pd.read_csv(file))
    elif ext in ['xlsx', 'xls']:
        return standardize_columns(pd.read_excel(file))
    elif ext == 'pdf':
        with pdfplumber.open(file) as pdf:
            # Extracting tables directly from PDF often works better for BMO/RBC
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text
    return None

# --- UI CONFIG ---
st.set_page_config(page_title="Canadian Finance AI", layout="wide")
st.title("🇨🇦 Canadian Bank Statement Analyzer")
st.markdown("Optimized for **BMO, Scotiabank, RBC, and Canadian Tire**.")

files = st.file_uploader("Upload Statements", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

if files:
    if st.button("🔍 Process and Categorize"):
        llm = init_ai()
        master_df = pd.DataFrame()
        
        for file in files:
            with st.status(f"Analyzing {file.name}...", expanded=False) as status:
                data = load_data(file)
                
                # Handling CSV/Excel (Structured)
                if isinstance(data, pd.DataFrame):
                    # Filter only required columns
                    needed = ['Date', 'Description', 'Amount']
                    available = [c for c in needed if c in data.columns]
                    df_subset = data[available].copy()
                    
                    # AI Categorization of unique descriptions (to save cost/time)
                    unique_desc = df_subset['Description'].dropna().unique().tolist()
                    
                    prompt = f"""Categorize these Canadian merchant names into: [Groceries, Utilities, Transport, Entertainment, Shopping, Income, Other].
                    Return ONLY a list: Merchant | Category
                    List: {unique_desc[:40]}"""
                    
                    response = llm.invoke(prompt)
                    cat_map = {}
                    for line in response.content.split('\n'):
                        if '|' in line:
                            parts = line.split('|')
                            cat_map[parts[0].strip()] = parts[1].strip()
                    
                    df_subset['Category'] = df_subset['Description'].map(cat_map).fillna('Other')
                    df_subset['Source'] = file.name
                    master_df = pd.concat([master_df, df_subset], ignore_index=True)

                # Handling PDF (Unstructured)
                else:
                    prompt = f"""Extract a transaction table from this Canadian bank text:
                    {data[:5000]}
                    Return ONLY a Markdown table: Date | Description | Amount | Category
                    Use negative for spending, positive for income."""
                    
                    response = llm.invoke(prompt)
                    # Convert AI text to table (using your existing parse_ai_table logic)
                    # ... [Insert parse_ai_table call here] ...
                
                status.update(label=f"Completed {file.name}!", state="complete")

        if not master_df.empty:
            # Clean Amount: RBC/BMO sometimes use (10.00) for negative or have currency symbols
            master_df['Amount'] = pd.to_numeric(
                master_df['Amount'].astype(str)
                .replace(r'[\$,]', '', regex=True)
                .replace(r'\((.*)\)', r'-\1', regex=True), # Converts (10.00) to -10.00
                errors='coerce'
            )
            
            st.header("📊 Total Monthly Summary")
            master_df['Date'] = pd.to_datetime(master_df['Date'], errors='coerce')
            master_df['Month'] = master_df['Date'].dt.to_period('M').astype(str)
            
            summary = master_df.groupby('Month').agg(
                Income=('Amount', lambda x: x[x > 0].sum()),
                Expenses=('Amount', lambda x: x[x < 0].sum())
            )
            st.bar_chart(summary)
            st.dataframe(master_df)
