import streamlit as st
import pandas as pd
import pdfplumber
import io
import datetime
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
    # Rule-based mapping for Canadian Merchants & Scotiabank specific codes
    mapping = {
        'Salary/Wages': ['payroll', 'salary', 'direct deposit', 'work inc'],
        'Government Benefits': ['gst', 'canada pro', 'ei benefit', 'tax refund', 'social security', 'provincial payment'],
        'Transfers In': ['customer transfer cr', 'ebt from', 'savings to checking', 'interac e-transfer'],
        'Interests & Dividends': ['interest earned', 'dividend payment', 'investment credit', 'overdraft interest'],
        'Housing/Rent': ['mortgage', 'rent payment', 'hoa', 'condo fee', 'landlord'],
        'Utilities': ['hydro', 'enbridge', 'rogers', 'bell', 'water bill', 'reliance', 'hp *instant ink'],
        'Transportation': ['uber', 'lyft', 'presto', 'shell', 'esso', 'petro', 'parking', 'cn tower food'],
        'Groceries': ['walmart', 'loblaws', 'no frills', 'sobeys', 'metro', 'costco', 'superstore', 'freshco', 'fortinos'],
        'Dining & Entertainment': ['starbucks', 'mcdonalds', 'netflix', 'spotify', 'tim hortons', 'lcbo', 'barburrito', 'osmow', 'harvey', 'second cup', 'popeyes', 'playstation', 'vimeo'],
        'Health & Wellness': ['shoppers', 'pharmacy', 'gym', 'goodlife', 'dentist', 'veterinary', 'trupanion', 'anytime fit'],
        'Shopping': ['amazon', 'apple.com', 'best buy', 'indigo', 'retail', 'sephora', 'winners'],
        'Insurance': ['insurance', 'sun life', 'manulife', 'geico', 'aviva'],
        'Fees': ['monthly fee', 'overdraft', 'atm withdrawal', 'service charge', 'nsf fee', 'interest charges']
    }

    def get_category(row):
        # Scan both Description and Sub-description
        text = (str(row.get('Description', '')) + " " + str(row.get('Sub-description', ''))).lower()
        for category, keywords in mapping.items():
            if any(k in text for k in keywords):
                return category
        return "Uncategorized"

    df['Category'] = df.apply(get_category, axis=1)
    
    # Optional: Send truly unknown items to AI
    uncat_mask = df['Category'] == "Uncategorized"
    if uncat_mask.any():
        unique_uncat = df[uncat_mask]['Description'].unique()[:20]
        try:
            prompt = f"Categorize these merchants into: {list(mapping.keys())}. List: {unique_uncat}. Format: Merchant | Category"
            res = llm.invoke(prompt)
            ai_map = {l.split('|')[0].strip(): l.split('|')[1].strip() for l in res.content.split('\n') if '|' in l}
            df.loc[uncat_mask, 'Category'] = df['Description'].map(ai_map).fillna('Misc. Expenses')
        except:
            df.loc[uncat_mask, 'Category'] = 'Misc. Expenses'
            
    return df

# --- 3. DATA NORMALIZATION ---
def standardize_df(df, filename):
    df.columns = [str(c).strip() for c in df.columns]
    low_cols = [c.lower() for c in df.columns]
    
    # Identify Scotiabank Visa (Signs are flipped in CSVs)
    is_visa = "visa" in filename.lower() or "scene" in filename.lower()

    mapping = {
        'date': ['Date'],
        'description': ['Description'],
        'sub-description': ['Sub-description'],
        'amount': ['Amount']
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
        # Normalize Visa: Spending (+) becomes Outflow (-), Payments (-) become Inflow (+)
        if is_visa:
            df['Amount'] = df['Amount'] * -1
            
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    return df

# --- 4. UI DASHBOARD ---
st.set_page_config(page_title="Canada AI Finance", layout="wide")
st.title("🇨🇦 Enhanced Canadian Bank Multi-Statement Analyzer")

files = st.file_uploader("Upload Statements (Scotiabank, BMO, RBC, etc.)", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

if files:
    if st.button("🚀 Run AI Analysis"):
        llm = get_llm()
        master_df = pd.DataFrame()
        
        for file in files:
            with st.spinner(f"Processing {file.name}..."):
                ext = file.name.split('.')[-1].lower()
                if ext in ['csv', 'xlsx', 'xls']:
                    df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
                else:
                    with pdfplumber.open(file) as pdf:
                        text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
                        res = llm.invoke(f"Extract transactions from: {text[:4000]}. Format: Date | Description | Amount")
                        rows = [l.split('|') for l in res.content.split('\n') if '|' in l and '---' not in l]
                        df = pd.DataFrame([r[1:4] for r in rows[1:]], columns=['Date','Description','Amount'])
                
                df = standardize_df(df, file.name)
                df = classify_transactions(df, llm)
                df['Source'] = file.name
                master_df = pd.concat([master_df, df], ignore_index=True)

        if not master_df.empty:
            master_df = master_df.dropna(subset=['Amount', 'Date'])
            master_df['Month-Year'] = master_df['Date'].dt.strftime('%Y-%m')
            
            # --- TABS FOR OUTPUT ---
            tab1, tab2, tab3 = st.tabs(["📊 Summary Report", "📝 All Transactions", "📥 Export"])

            with tab1:
                st.subheader("Monthly Category Breakdown")
                pivot_summary = master_df.pivot_table(
                    index='Category', 
                    columns='Month-Year', 
                    values='Amount', 
                    aggfunc='sum', 
                    fill_value=0
                )
                st.dataframe(pivot_summary.style.format("${:,.2f}"), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Total Inflow/Outflow")
                    summary = master_df.groupby('Month-Year').agg(
                        Income=('Amount', lambda x: x[x > 0].sum()),
                        Expenses=('Amount', lambda x: abs(x[x < 0].sum()))
                    )
                    st.bar_chart(summary)

            with tab2:
                st.subheader("Unified Transaction Log")
                st.dataframe(master_df[['Date', 'Description', 'Category', 'Amount', 'Source']], use_container_width=True)

            with tab3:
                # Prepare Excel with Multiple Sheets
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    master_df.to_excel(writer, index=False, sheet_name='All_Transactions')
                    pivot_summary.to_excel(writer, sheet_name='Monthly_Category_Summary')
                
                st.success("Report generated with Category Summary tab!")
                st.download_button("📥 Download Excel Report", output.getvalue(), "Scotiabank_Enhanced_Report.xlsx")
