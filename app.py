import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
from datetime import datetime
from thefuzz import process

# Correct Import for Langchain IBM
try:
    from langchain_ibm import ChatWatsonx
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

st.set_page_config(page_title="AI Expense Analyzer Pro", layout="wide")

# =========================================================
# 1. PARSING ENGINE (RBC PDF & Scotiabank CSV)
# =========================================================

def process_scotiabank_csv(uploaded_file):
    """Parses Scotiabank CSVs by skipping metadata and identifying columns."""
    content = uploaded_file.getvalue().decode('utf-8').splitlines()
    data_start_idx = 0
    for i, line in enumerate(content):
        # The data header starts with 'Filter,Date,Description' or contains date-like strings
        if 'Date,Description' in line:
            data_start_idx = i
            break
    
    df = pd.read_csv(io.StringIO("\n".join(content[data_start_idx:])))
    
    # Standardize column names
    df = df.rename(columns={'Date': 'Date', 'Description': 'Description', 'Amount': 'Amount'})
    
    # Clean Amount: Scotiabank Credit uses Debit/Credit columns sometimes or string amounts
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        
    return df[['Date', 'Description', 'Amount']]

def process_rbc_pdf(uploaded_file):
    """Parses RBC Business PDFs (Chequing & Operating Loans)."""
    rows = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2: continue
                
                headers = [str(h).replace('\n', ' ') for h in table[0]]
                
                for row in table[1:]:
                    if not row[0] or "Opening balance" in str(row[1]): continue
                    
                    entry = {"Date": row[0], "Description": str(row[1]).replace('\n', ' ')}
                    
                    # Logic A: RBC Business Chequing (Debits/Credits columns)
                    if "Cheques & Debits" in str(headers):
                        debit = str(row[2]).replace(',', '').replace('$', '').strip() if row[2] else "0"
                        credit = str(row[3]).replace(',', '').replace('$', '').strip() if row[3] else "0"
                        try:
                            entry["Amount"] = float(credit or 0) - float(debit or 0)
                        except: entry["Amount"] = 0
                    
                    # Logic B: RBC Operating Loan (Transaction Amount column)
                    elif "Transaction Amount" in str(headers):
                        amt_str = str(row[3]).replace(',', '').replace('$', '').strip()
                        try:
                            # Handle negative sign if present (e.g., "-$1,750.00")
                            is_negative = "-" in amt_str
                            val = float(re.sub(r'[^\d.]', '', amt_str))
                            entry["Amount"] = -val if is_negative else val
                        except: entry["Amount"] = 0
                    
                    if "Amount" in entry:
                        rows.append(entry)
                        
    return pd.DataFrame(rows)

# =========================================================
# 2. ENRICHMENT & UI
# =========================================================

st.title("🚀 Financial Statement AI")
st.write("Upload your RBC PDFs and Scotiabank CSVs below.")

files = st.file_uploader("Upload Statements", type=["pdf", "csv"], accept_multiple_files=True)

if files:
    all_data = []
    for f in files:
        try:
            if f.name.lower().endswith('.csv'):
                df = process_scotiabank_csv(f)
            else:
                df = process_rbc_pdf(f)
            
            if not df.empty:
                df['Source'] = f.name
                all_data.append(df)
                st.success(f"Loaded {f.name}")
        except Exception as e:
            st.error(f"Error processing {f.name}: {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Dashboard
        st.subheader("Unified Transaction Feed")
        st.dataframe(final_df, use_container_width=True)
        
        # Monthly Summary
        final_df['Date'] = pd.to_datetime(final_df['Date'], errors='coerce')
        summary = final_df.groupby(final_df['Date'].dt.strftime('%Y-%m'))['Amount'].sum()
        st.subheader("Monthly Cashflow")
        st.line_chart(summary)

        # Export
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clean Data (CSV)", csv, "fin_report.csv", "text/csv")
