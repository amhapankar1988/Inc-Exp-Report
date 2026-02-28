import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =========================================================
# 1. INITIALIZATION & SECRETS
# =========================================================

st.set_page_config(page_title="Universal AI Expense Analyzer", layout="wide")

REQUIRED_CATEGORIES = [
    "Utilities", "Interest Charge", "Shopping", "Dining", 
    "Transportation", "Health and Wellbeing", "Mortgage", "Other"
]

def get_ai_model():
    try:
        # Accessing Streamlit Secret Variables
        api_key = st.secrets["WATSONX_APIKEY"].strip()
        project_id = st.secrets["WATSONX_PROJECT_ID"].strip()
        
        # CORRECTED URL FOR TORONTO REGION
        url = "https://ca-tor.ml.cloud.ibm.com"

        parameters = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 500,
            GenParams.TEMPERATURE: 0
        }

        return ChatWatsonx(
            model_id="ibm/granite-3-8b-instruct",
            url=url,
            project_id=project_id,
            apikey=api_key,
            params=parameters,
        )
    except Exception as e:
        st.error(f"AI Config Error: {e}")
        return None

# =========================================================
# 2. EXTRACTION ENGINE (Hybrid PDF + Structured)
# =========================================================

def clean_amount(val):
    if val is None or pd.isna(val): return 0.0
    clean = re.sub(r'[^\d.-]', '', str(val))
    try:
        if clean.endswith('-'): clean = '-' + clean[:-1]
        return float(clean)
    except: return 0.0

def process_pdf(file):
    all_rows = []
    line_regex = re.compile(r"(\d{1,4}[/-]\d{1,2}[/-]?\d{0,4}|[A-Z][a-z]{2}\s\d{1,2})\s+(.*?)\s+(-?\$?[\d,]+\.\d{2})")

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    clean_row = [str(c).strip() for c in row if c and str(c).strip()]
                    if len(clean_row) >= 3: all_rows.append(clean_row[:3])
            
            if not all_rows:
                text = page.extract_text(layout=True)
                if text:
                    for line in text.split('\n'):
                        match = line_regex.search(line)
                        if match: all_rows.append(list(match.groups()))
                        
    df = pd.DataFrame(all_rows, columns=["Date", "Description", "Amount"])
    noise = ["balance", "transaction", "description", "date", "amount"]
    df = df[~df['Description'].str.lower().str.contains('|'.join(noise), na=False)]
    return df

def process_structured(file):
    ext = file.name.split('.')[-1].lower()
    df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
    cols = df.columns.tolist()
    d_col = next((c for c in cols if 'date' in c.lower()), None)
    desc_col = next((c for c in cols if any(x in c.lower() for x in ['desc', 'memo', 'trans', 'detail'])), None)
    a_col = next((c for c in cols if any(x in c.lower() for x in ['amt', 'amount', 'value', 'charge'])), None)
    
    if d_col and desc_col and a_col:
        return pd.DataFrame({"Date": df[d_col], "Description": df[desc_col], "Amount": df[a_col]})
    return pd.DataFrame()

# =========================================================
# 3. HYBRID CATEGORIZATION
# =========================================================

def categorize_data(df, model):
    rules = {
        "Dining": ["starbucks", "mcdonald", "pizza", "uber eats", "tim hortons", "dining", "restaurant"],
        "Transportation": ["uber", "lyft", "shell", "petro", "esso", "gas", "parking", "presto"],
        "Utilities": ["bell", "rogers", "hydro", "water", "electricity", "telus", "internet", "enbridge"],
        "Health and Wellbeing": ["pharmacy", "gym", "dentist", "shoppers", "hospital", "medical"],
        "Mortgage": ["mortgage", "housing loan", "home loan"],
        "Interest Charge": ["interest charge", "finance charge", "interest pd", "service fee"]
    }

    def apply_rules(desc):
        desc = str(desc).lower()
        for cat, keywords in rules.items():
            if any(k in desc for k in keywords): return cat
        return None

    df["Category"] = df["Description"].apply(apply_rules)
    mask = df["Category"].isnull()
    unknowns = df[mask]["Description"].unique().tolist()

    if unknowns and model:
        prompt = f"Categorize these into {', '.join(REQUIRED_CATEGORIES)}. Return: 'Description | Category'.\n\n" + "\n".join([f"- {d}" for d in unknowns[:30]])
        try:
            res = model.invoke(prompt)
            for line in res.content.split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    d_key, c_val = parts[0].strip("- "), parts[1].strip()
                    df.loc[df["Description"].str.contains(re.escape(d_key), case=False, na=False), "Category"] = c_val
        except: pass

    df["Category"] = df["Category"].fillna("Other")
    return df

# =========================================================
# 4. APP UI
# =========================================================

st.title("🏦 AI Financial Intelligence (Canada Region)")

# Sidebar Connection Test
if st.sidebar.button("Verify AI Connection"):
    with st.spinner("Pinging IBM Watsonx Toronto..."):
        model = get_ai_model()
        if model:
            try:
                test = model.invoke("Say 'Connected'")
                st.sidebar.success(f"✅ {test.content}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

files = st.file_uploader("Upload Files (PDF, CSV, Excel)", type=["pdf", "csv", "xlsx", "xls"], accept_multiple_files=True)

if files:
    ai_model = get_ai_model()
    master_list = []

    for f in files:
        with st.spinner(f"Extracting {f.name}..."):
            data = process_pdf(f) if f.name.endswith('pdf') else process_structured(f)
            if not data.empty:
                data["Amount"] = data["Amount"].apply(clean_amount)
                data = categorize_data(data, ai_model)
                master_list.append(data)

    if master_list:
        final_df = pd.concat(master_list, ignore_index=True)
        st.subheader("📝 Review Transactions")
        final_df = st.data_editor(final_df, use_container_width=True, hide_index=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Category Breakdown")
            summary = final_df.groupby("Category")["Amount"].sum().abs()
            st.bar_chart(summary)
        with c2:
            st.write("### Summary Metrics")
            st.metric("Total Expenses", f"${summary.sum():,.2f}")
            st.dataframe(summary.map(lambda x: f"$ {x:,.2f}"))

        st.download_button("📥 Download Report", final_df.to_csv(index=False), "expense_report.csv")
