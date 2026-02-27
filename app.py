import streamlit as st
import pandas as pd
import pdfplumber
import os
import io
from langchain_ibm import ChatWatsonx

# --- SAFE INITIALIZATION ---
def init_ai():
    """Initializes Watsonx only when called to prevent build-time crashes."""
    api_key = os.getenv("WATSONX_APIKEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL", "https://ca-tor.ml.cloud.ibm.com")

    if not api_key or not project_id:
        st.error("Missing Credentials. Please add WATSONX_APIKEY and PROJECT_ID to Space Secrets.")
        st.stop()

    try:
        return ChatWatsonx(
            model_id="meta-llama/llama-3-3-70b-instruct",
            url=url,
            apikey=api_key,
            project_id=project_id,
            params={
                "decoding_method": "greedy", 
                "max_new_tokens": 2000,
                "temperature": 0
            }
        )
    except Exception as e:
        st.error(f"Failed to connect to Watsonx: {str(e)}")
        st.stop()

# --- DOCUMENT PROCESSING ---
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

def parse_ai_table(ai_text):
    """Converts AI Markdown table back to a Pandas DataFrame."""
    lines = ai_text.strip().split('\n')
    data = []
    for line in lines:
        if '|' in line and '---' not in line:
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
            if len(row) >= 4:
                data.append(row)
    if len(data) > 1:
        # Create DF and handle headers
        cols = [c for c in data[0]]
        df = pd.DataFrame(data[1:], columns=cols[:len(data[1])])
        # Data Cleaning
        df['Amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    return pd.DataFrame()

# --- DASHBOARD UI ---
st.set_page_config(page_title="AI Finance Dashboard", layout="wide")
st.title("📊 Financial Statement Brain")
st.markdown("Upload multiple statements to see a monthly breakdown of income vs. expenses.")

files = st.file_uploader("Upload Bank Statements", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

if files:
    if st.button("🚀 Analyze All Statements"):
        llm = init_ai()
        master_df = pd.DataFrame()
        
        with st.spinner("The 'Brain' is processing your data..."):
            for file in files:
                text_content = extract_text(file)
                prompt = f"""
                Act as a financial analyzer. Extract transactions from this data:
                {text_content[:6000]}
                
                Return a Markdown table with these columns:
                Date | Description | Amount | Category
                
                RULES:
                1. Amounts: Negative for spending (e.g., -15.50), Positive for income (e.g., 500.00).
                2. Category: [Food, Housing, Transport, Utilities, Entertainment, Salary, Other].
                """
                response = llm.invoke(prompt)
                master_df = pd.concat([master_df, parse_ai_table(response.content)], ignore_index=True)

        if not master_df.empty:
            master_df = master_df.dropna(subset=['Amount', 'Date'])
            master_df['Month'] = master_df['Date'].dt.strftime('%Y-%m')

            # --- VISUALIZATION ---
            st.header("📈 Income vs. Expenses Trend")
            chart_data = master_df.groupby('Month').agg(
                Income=('Amount', lambda x: x[x > 0].sum()),
                Expenses=('Amount', lambda x: abs(x[x < 0].sum()))
            )
            st.bar_chart(chart_data)

            # --- DRILL DOWN ---
            st.header("🔍 Monthly Drill-Down")
            for month in sorted(master_df['Month'].unique(), reverse=True):
                with st.expander(f"Details for {month}"):
                    m_data = master_df[master_df['Month'] == month]
                    st.write(f"**Total Income:** ${m_data[m_data['Amount'] > 0]['Amount'].sum():.2f}")
                    st.write(f"**Total Expenses:** ${abs(m_data[m_data['Amount'] < 0]['Amount'].sum()):.2f}")
                    st.dataframe(m_data[['Date', 'Description', 'Amount', 'Category']], use_container_width=True)

            # --- EXPORT ---
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                master_df.to_excel(writer, index=False, sheet_name='All_Transactions')
                chart_data.to_excel(writer, sheet_name='Monthly_Summary')
            
            st.download_button("📥 Download Excel Report", output.getvalue(), "Financial_Report.xlsx")
        else:
            st.warning("Could not extract any valid transactions. Check the file format.")
