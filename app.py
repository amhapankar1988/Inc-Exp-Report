import streamlit as st
import pandas as pd
import pdfplumber
import os
import io
import re
from langchain_ibm import ChatWatsonx

# --- WATSONX SETUP ---
def init_ai():
    api_key = os.getenv("WATSONX_APIKEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL", "https://ca-tor.ml.cloud.ibm.com")

    if not api_key or not project_id:
        st.error("Credentials missing! Please check HF Secrets.")
        st.stop()

    return ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url=url,
        apikey=api_key,
        project_id=project_id,
        params={"decoding_method": "greedy", "max_new_tokens": 2000}
    )

# --- DATA EXTRACTION ---
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif ext in ['csv', 'xlsx', 'xls']:
        df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
        return df.to_string()
    return ""

def parse_ai_table(ai_text):
    lines = ai_text.strip().split('\n')
    data = []
    for line in lines:
        if '|' in line and '---' not in line:
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
            if len(row) >= 4:
                data.append(row)
    if len(data) > 1:
        df = pd.DataFrame(data[1:], columns=data[0][:len(data[1])])
        # Force numeric conversion and clean currency symbols
        df['Amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    return pd.DataFrame()

# --- UI CONFIG ---
st.set_page_config(page_title="AI Finance Dashboard", layout="wide")
st.title("📊 Financial Intelligence Dashboard")

uploaded_files = st.file_uploader("Upload Statements", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    llm = init_ai()
    
    if st.button("🚀 Analyze and Visualize"):
        master_df = pd.DataFrame()
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            content = extract_text(file)
            prompt = f"""
            Audit this statement: {file.name}. 
            Return a Markdown table: Date | Description | Amount | Category
            RULES:
            1. Use NEGATIVE numbers for expenses (e.g., -50.00).
            2. Use POSITIVE numbers for income/deposits (e.g., 2000.00).
            3. Categories: [Housing, Food, Transport, Utilities, Entertainment, Salary, Income, Other].
            Data: {content[:5000]}
            """
            response = llm.invoke(prompt)
            temp_df = parse_ai_table(response.content)
            master_df = pd.concat([master_df, temp_df], ignore_index=True)
            progress_bar.progress((i + 1) / len(uploaded_files))

        # Cleanup Data
        master_df = master_df.dropna(subset=['Amount', 'Date'])
        master_df['Month'] = master_df['Date'].dt.strftime('%Y-%m')

        # --- VISUALIZATION SECTION ---
        st.header("📈 Financial Trends")
        
        # Monthly Income vs Expense Chart
        chart_data = master_df.groupby('Month').agg(
            Income=('Amount', lambda x: x[x > 0].sum()),
            Expenses=('Amount', lambda x: abs(x[x < 0].sum()))
        )
        
        st.bar_chart(chart_data)

        # Drill Down
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Category Breakdown (All Time)")
            category_data = master_df[master_df['Amount'] < 0].groupby('Category')['Amount'].sum().abs()
            st.write(category_data)
            
        with col2:
            st.subheader("Monthly Totals")
            summary_table = chart_data.copy()
            summary_table['Net'] = summary_table['Income'] - summary_table['Expenses']
            st.dataframe(summary_table.style.highlight_max(axis=0, color='#90ee90'))

        # --- DETAILED DRILL DOWN ---
        st.header("🔍 Monthly Details")
        for month in sorted(master_df['Month'].unique(), reverse=True):
            with st.expander(f"View transactions for {month}"):
                st.dataframe(master_df[master_df['Month'] == month][['Date', 'Description', 'Amount', 'Category']], use_container_width=True)

        # --- EXCEL EXPORT ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            summary_table.to_excel(writer, sheet_name='Summary')
            master_df.to_excel(writer, sheet_name='All_Transactions', index=False)
        
        st.download_button("📥 Download Excel Report", output.getvalue(), "Finance_Report.xlsx")
