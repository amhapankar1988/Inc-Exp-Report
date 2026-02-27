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
        st.error("Credentials missing in HF Secrets!")
        st.stop()

    return ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url=url,
        apikey=api_key,
        project_id=project_id,
        params={
            "decoding_method": "greedy", 
            "max_new_tokens": 2000,
            "temperature": 0.0
        }
    )

# --- DATA EXTRACTION ---
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    try:
        if ext == 'pdf':
            with pdfplumber.open(file) as pdf:
                return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif ext in ['csv', 'xlsx', 'xls']:
            df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
            return df.to_string()
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
    return ""

# --- PARSING AI TABLE TO DATAFRAME ---
def parse_ai_table(ai_text):
    """Parses Markdown tables into a Pandas DataFrame."""
    lines = ai_text.strip().split('\n')
    data = []
    for line in lines:
        if '|' in line and '---' not in line:
            # Split by pipe, strip whitespace, remove empty edge strings
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
            if row:
                data.append(row)
    
    if len(data) > 1:
        return pd.DataFrame(data[1:], columns=data[0])
    return pd.DataFrame(columns=["Analysis"], data=[[ai_text]])

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Finance Multi-Analyzer", layout="wide")
st.title("🏦 Multi-Statement Transaction Classifier")
st.markdown("Upload multiple statements. The 'Brain' (watsonx.ai) will categorize them into a single Excel file.")

uploaded_files = st.file_uploader(
    "Upload Bank Statements (PDF, CSV, XLSX)", 
    type=["pdf", "csv", "xlsx"], 
    accept_multiple_files=True
)

if uploaded_files:
    llm = init_ai()
    
    if st.button("🚀 Run AI Analysis"):
        master_df = pd.DataFrame()
        progress_text = st.empty()
        bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            progress_text.text(f"Analyzing {file.name}...")
            content = extract_text(file)
            
            # Refined Prompt for Strict Table Output
            prompt = f"""
            You are a financial auditor. Extract transactions from this text.
            Classify into: [Housing, Food, Transport, Utilities, Entertainment, Salary, Other].
            
            Return ONLY a Markdown table with exactly these columns:
            Date | Description | Amount | Category | Source_File

            Use the file name "{file.name}" for the Source_File column.
            
            Data:
            {content[:5000]}
            """
            
            response = llm.invoke(prompt)
            temp_df = parse_ai_table(response.content)
            master_df = pd.concat([master_df, temp_df], ignore_index=True)
            
            bar.progress((i + 1) / len(uploaded_files))

        progress_text.text("Analysis Complete!")

        # --- RESULTS VIEW ---
        st.subheader("Categorized Transactions")
        st.dataframe(master_df, use_container_width=True)

        # --- EXCEL DOWNLOAD ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            master_df.to_excel(writer, index=False, sheet_name='Categorized')
            
            # Formatting the Excel (Optional)
            workbook = writer.book
            worksheet = writer.sheets['Categorized']
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            for col_num, value in enumerate(master_df.columns.values):
                worksheet.write(0, col_num, value, header_format)

        st.download_button(
            label="📥 Download Master Excel Report",
            data=output.getvalue(),
            file_name="Bank_Analysis_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )