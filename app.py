import streamlit as st
import pandas as pd
import pdfplumber
import io
from ibm_watsonx_ai import APIClient, Credentials
from langchain_ibm import ChatWatsonx

# --- 1. SECURE INITIALIZATION ---
def get_llm():
    api_key = st.secrets["WATSONX_APIKEY"]
    project_id = st.secrets["WATSONX_PROJECT_ID"]
    
    creds = Credentials(url="https://ca-tor.ml.cloud.ibm.com", api_key=api_key)

    try:
        client = APIClient(creds)
        return ChatWatsonx(
            model_id="meta-llama/llama-3-3-70b-instruct",
            watsonx_client=client,
            project_id=project_id,
            params={"decoding_method": "greedy", "max_new_tokens": 1500, "temperature": 0}
        )
    except Exception as e:
        st.error(f"Brain Connection Error: {str(e)}")
        st.stop()

# --- 2. ENHANCED CLASSIFICATION ENGINE ---
def classify_transactions(df, llm):
    """Classifies transactions using Canadian banking rules and LLM fallback."""
    
    # Precise Keyword Mapping based on your requirements
    mapping = {
        'Salary/Wages': ['payroll', 'salary', 'direct deposit', 'work inc'],
        'Government Benefits': ['gst', 'canada pro', 'ei benefit', 'tax refund', 'social security'],
        'Transfers In': ['transfer from', 'ebt from', 'savings to checking'],
        'Interests & Dividends': ['interest earned', 'dividend payment', 'investment credit'],
        'Housing/Rent': ['mortgage', 'rent payment', 'hoa', 'condo fee'],
        'Utilities': ['hydro', 'enbridge', 'rogers', 'bell', 'water bill', 'reliance'],
        'Transportation': ['uber', 'lyft', 'presto', 'shell', 'esso', 'petro', 'parking'],
        'Groceries': ['walmart', 'loblaws', 'no frills', 'sobeys', 'metro', 'costco', 'superstore'],
        'Dining & Entertainment': ['starbucks', 'mcdonalds', 'netflix', 'spotify', 'restaurant', 'pub', 'tim hortons'],
        'Health & Wellness': ['shoppers', 'pharmacy', 'gym', 'goodlife', 'dentist'],
        'Shopping': ['amazon', 'apple', 'best buy', 'indigo', 'retail'],
        'Fees': ['monthly fee', 'overdraft', 'atm withdrawal', 'service charge']
    }

    def get_category(description):
        desc = str(description).lower()
        for category, keywords in mapping.items():
            if any(k in desc for k in keywords):
                return category
        return "Uncategorized"

    df['Category'] = df['Description'].apply(get_category)
    
    # AI Fallback for "Uncategorized" items
    uncategorized = df[df['Category'] == "Uncategorized"]['Description'].unique()
    if len(uncategorized) > 0:
        cat_prompt = f"""Categorize these Canadian bank transactions into: 
        {list(mapping.keys())} or 'Misc. Expenses'.
        List: {uncategorized[:30]}
        Return ONLY: Merchant | Category"""
        
        try:
            res = llm.invoke(cat_prompt)
            ai_map = {line.split('|')[0].strip(): line.split('|')[1].strip() 
                      for line in res.content.split('\n') if '|' in line}
            df.loc[df['Category'] == "Uncategorized", 'Category'] = df['Description'].map(ai_map).fillna('Misc. Expenses')
        except:
            pass # Fallback to 'Misc. Expenses' if LLM fails
            
    return df

# --- 3. DATA NORMALIZATION ---
def standardize_df(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    mapping = {
        'date': ['date', 'transaction date', 'posted date', 'txn date'],
        'description': ['description', 'desc', 'transaction', 'memo', 'details'],
        'amount': ['amount', 'debit', 'credit', 'value']
    }
    new_cols = {}
    for standard, synonyms in mapping.items():
        for col in df.columns:
            if col in synonyms: new_cols[col] = standard.capitalize()
    
    df = df.rename(columns=new_cols)
    if 'Amount' in df.columns:
        df['Amount'] = df['Amount'].astype(str).replace(r'[\$,]', '', regex=True)
        df['Amount'] = df['Amount'].replace(r'\((.*)\)', r'-\1', regex=True)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

# --- 4. UI DASHBOARD ---
st.set_page_config(page_title="Canada AI Finance", layout="wide")
st.title("🇨🇦 Canadian Bank Intelligence Report")

files = st.file_uploader("Upload Statements (PDF/CSV)", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

if files:
    if st.button("🚀 Run AI Analysis"):
        llm = get_llm()
        master_df = pd.DataFrame()
        
        for file in files:
            with st.spinner(f"Analyzing {file.name}..."):
                ext = file.name.split('.')[-1].lower()
                if ext in ['csv', 'xlsx', 'xls']:
                    df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
                else:
                    with pdfplumber.open(file) as pdf:
                        text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
                        res = llm.invoke(f"Extract transactions from: {text[:4000]}. Format: Date | Description | Amount")
                        rows = [l.split('|') for l in res.content.split('\n') if '|' in l and '---' not in l]
                        df = pd.DataFrame([r[1:4] for r in rows[1:]], columns=['Date','Description','Amount'])
                
                df = standardize_df(df)
                df = classify_transactions(df, llm)
                df['Source'] = file.name
                master_df = pd.concat([master_df, df], ignore_index=True)

        if not master_df.empty:
            master_df = master_df.dropna(subset=['Amount', 'Date'])
            
            # --- TABS FOR OUTPUT ---
            tab1, tab2, tab3 = st.tabs(["📊 Summary Report", "📝 All Transactions", "📥 Export"])

            with tab1:
                st.subheader("Financial Overview")
                
                # Metrics
                total_income = master_df[master_df['Amount'] > 0]['Amount'].sum()
                total_expense = master_df[master_df['Amount'] < 0]['Amount'].sum()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Income", f"${total_income:,.2f}")
                m2.metric("Total Expenses", f"${abs(total_expense):,.2f}")
                m3.metric("Net Cash Flow", f"${(total_income + total_expense):,.2f}")

                # Summary Table by Category
                st.write("### Spending by Category")
                cat_summary = master_df.groupby('Category')['Amount'].agg(['sum', 'count']).reset_index()
                cat_summary.columns = ['Category', 'Total Amount', 'Transactions']
                cat_summary['Total Amount'] = cat_summary['Total Amount'].map('${:,.2f}'.format)
                st.table(cat_summary)
                
                # Visuals
                st.bar_chart(master_df.groupby('Category')['Amount'].sum().abs())

            with tab2:
                st.subheader("Transaction Log")
                st.dataframe(master_df[['Date', 'Description', 'Category', 'Amount', 'Source']], use_container_width=True)

            with tab3:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    master_df.to_excel(writer, index=False, sheet_name='Data')
                st.download_button("Download Excel Report", output.getvalue(), "Finance_Report.xlsx")
