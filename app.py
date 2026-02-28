import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =========================================================
# 1. INITIALIZATION & SESSION STATE
# =========================================================

st.set_page_config(page_title="AI Expense Intelligence", layout="wide")

if "custom_rules" not in st.session_state:
    st.session_state.custom_rules = {}

REQUIRED_CATEGORIES = [
    "Utilities", "Interest Charge", "Shopping", "Food & Dining", 
    "Transportation", "Entertainment", "Health & Fitness", "Mortgage", 
    "Housing", "Deposits", "Withdrawals", "Overdraft Fee", "NSF", "Monthly Account Fee", "Other"
]

def get_ai_model():
    try:
        api_key = st.secrets["WATSONX_APIKEY"].strip()
        project_id = st.secrets["WATSONX_PROJECT_ID"].strip()
        url = "https://ca-tor.ml.cloud.ibm.com"

        return ChatWatsonx(
            # Switch to a supported, high-performance model from your environment list
            model_id="meta-llama/llama-3-3-70b-instruct", 
            url=url,
            project_id=project_id,
            apikey=api_key,
            params={
                GenParams.DECODING_METHOD: "greedy", 
                GenParams.MAX_NEW_TOKENS: 500, 
                GenParams.TEMPERATURE: 0
            },
        )
    except Exception as e:
        st.error(f"AI Config Error: {e}")
        return None

# =========================================================
# 2. ENHANCED CATEGORIZATION ENGINE
# =========================================================

def categorize_data(df, model):
    base_rules = {
        "Food & Dining": ["starbucks", "mcdonald", "tim hortons", "uber eats", "restaurant", "subway", "wendy", "pizza", "popeyes", "osmow", "barburrito", "harvey"],
        "Transportation": ["uber", "lyft", "shell", "petro", "esso", "gas", "presto", "ttc", "go transit", "parking"],
        "Utilities": ["bell", "rogers", "fido", "hydro", "enbridge", "telus", "internet", "metergy"],
        "Health & Fitness": ["shoppers", "pharmacy", "gym", "dentist", "medical", "hospital", "lifelabs", "veterinary", "trupanion", "anytime fit"],
        "Mortgage": ["mortgage", "housing loan", "property tax"],
        "Interest Charge": ["interest charge", "finance charge", "monthly fee", "service fee", "overdraft"],
        "Shopping": ["amazon", "walmart", "costco", "best buy", "canadian tire", "dollarama", "fortinos", "freshco", "lcbo", "winners", "apple.com/bill"]
    }

    def apply_logic(desc):
        desc = str(desc).lower()
        for keyword, cat in st.session_state.custom_rules.items():
            if keyword.lower() in desc: return cat
        for cat, keywords in base_rules.items():
            if any(k in desc for k in keywords): return cat
        return None

    df["Category"] = df["Description"].apply(apply_logic)
    
    mask = df["Category"].isnull()
    unknowns = df[mask]["Description"].unique().tolist()

    if unknowns and model:
        with st.spinner(f"AI is classifying {len(unknowns)} unique items..."):
            prompt = f"[INST] Categorize these into: {', '.join(REQUIRED_CATEGORIES)}. Return ONLY format: Description | Category\n\n" + "\n".join([f"- {d}" for d in unknowns[:20]])
            try:
                res = model.invoke(prompt)
                for line in res.content.split('\n'):
                    if '|' in line:
                        for d in unknowns:
                            if d.lower() in line.lower():
                                for cat in REQUIRED_CATEGORIES:
                                    if cat.lower() in line.lower():
                                        df.loc[df["Description"] == d, "Category"] = cat
            except: pass

    df["Category"] = df["Category"].fillna("Other")
    return df

# =========================================================
# 3. ROBUST PARSERS
# =========================================================

def process_pdf(file):
    all_rows = []
    line_regex = re.compile(r"([A-Z][a-z]{2}\s\d{2})\s+([A-Z][a-z]{2}\s\d{2})\s+(.*?)\s+(-?[\d,]+\.\d{2})")

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split('\n'):
                    if "INTEREST CHARGES" in line.upper():
                        parts = line.split()
                        if len(parts) >= 3:
                            all_rows.append([parts[0], "INTEREST CHARGES", parts[-1]])
                    
                    match = line_regex.search(line)
                    if match:
                        all_rows.append([match.group(1), match.group(3), match.group(4)])
                        
    df = pd.DataFrame(all_rows, columns=["Date", "Description", "Amount"])
    
    def clean_currency(val):
        s = str(val).replace('$', '').replace(',', '').strip()
        if s.endswith('-'): s = '-' + s[:-1]
        try: return float(s)
        except: return 0.0

    df["Amount"] = df["Amount"].apply(clean_currency)
    noise = ["date", "description", "amount", "total", "posting", "page", "statement"]
    df = df[~df['Description'].str.lower().str.contains('|'.join(noise), na=False)]
    return df

def process_excel_csv(file):
    filename = file.name.lower()
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file, skiprows=1) # Skipping the filter/header row found in your samples 
        else:
            df = pd.read_excel(file, engine='openpyxl')
        
        # Standardizing Column Names based on your sample files 
        mapping = {
            'Date': 'Date',
            'Description': 'Description',
            'Sub-description': 'Sub_Description',
            'Amount': 'Amount'
        }
        df.rename(columns=mapping, inplace=True)
        
        # Merge Description and Sub-description for better AI context [cite: 201, 492]
        if 'Sub_Description' in df.columns:
            df['Description'] = df['Description'].fillna('') + ' ' + df['Sub_Description'].fillna('')
        
        return df[['Date', 'Description', 'Amount']]
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
    return pd.DataFrame()

# =========================================================
# 4. UI & WORKFLOW
# =========================================================

st.title("🏦 Smart AI Expense Analyzer")

with st.sidebar:
    st.header("🎓 Teach the AI")
    new_kw = st.text_input("Vendor Keyword (e.g. 'NETFLIX')")
    new_cat = st.selectbox("Assign to Category", REQUIRED_CATEGORIES)
    if st.button("Learn Keyword"):
        if new_kw:
            st.session_state.custom_rules[new_kw] = new_cat
            st.success(f"Added {new_kw} -> {new_cat}")
            st.rerun()

    if st.session_state.custom_rules:
        st.write("### Learned Rules")
        for k, v in st.session_state.custom_rules.items():
            st.caption(f"{k} ➔ {v}")

files = st.file_uploader("Upload Statements", type=["pdf", "csv", "xlsx", "xls"], accept_multiple_files=True)

if files:
    ai_model = get_ai_model()
    master_dfs = []

    for f in files:
        if f.name.endswith('pdf'):
            df = process_pdf(f)
        else:
            df = process_excel_csv(f)
            
        if not df.empty:
            df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce').fillna(0.0)
            df = categorize_data(df, ai_model)
            master_dfs.append(df)

    if master_dfs:
        final_df = pd.concat(master_dfs, ignore_index=True)
        st.subheader("📝 Transaction Review")
        edited_df = st.data_editor(final_df, use_container_width=True, hide_index=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Total Spending")
            summary = edited_df.groupby("Category")["Amount"].sum().abs()
            st.bar_chart(summary)
        with c2:
            st.write("### Key Metrics")
            st.metric("Total Expenses", f"${summary.sum():,.2f}")
            st.table(summary.map(lambda x: f"$ {x:,.2f}"))

        st.download_button("📥 Download Report", edited_df.to_csv(index=False), "expense_report.csv")
