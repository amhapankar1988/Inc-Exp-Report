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

# Persistent memory for "learned" categories
if "custom_rules" not in st.secrets:
    if "custom_rules" not in st.session_state:
        st.session_state.custom_rules = {}

REQUIRED_CATEGORIES = [
    "Utilities", "Interest Charge", "Shopping", "Dining", 
    "Transportation", "Health and Wellbeing", "Mortgage", "Other"
]

def get_ai_model():
    try:
        api_key = st.secrets["WATSONX_APIKEY"].strip()
        project_id = st.secrets["WATSONX_PROJECT_ID"].strip()
        url = "https://ca-tor.ml.cloud.ibm.com"

        return ChatWatsonx(
            model_id="ibm/granite-3-8b-instruct",
            url=url,
            project_id=project_id,
            apikey=api_key,
            params={GenParams.DECODING_METHOD: "greedy", GenParams.MAX_NEW_TOKENS: 500, GenParams.TEMPERATURE: 0},
        )
    except Exception as e:
        st.error(f"AI Config Error: {e}")
        return None

# =========================================================
# 2. ENHANCED CATEGORIZATION ENGINE
# =========================================================

def categorize_data(df, model):
    # Base rules for common Canadian vendors
    base_rules = {
        "Dining": ["starbucks", "mcdonald", "tim hortons", "uber eats", "restaurant", "subway", "wendy", "pizza"],
        "Transportation": ["uber", "lyft", "shell", "petro", "esso", "gas", "presto", "ttc", "go transit"],
        "Utilities": ["bell", "rogers", "hydro", "enbridge", "telus", "fido", "internet"],
        "Health and Wellbeing": ["shoppers", "pharmacy", "gym", "dentist", "medical", "hospital", "lifelabs"],
        "Mortgage": ["mortgage", "housing loan", "property tax"],
        "Interest Charge": ["interest charge", "finance charge", "monthly fee", "service fee", "overdraft"],
        "Shopping": ["amazon", "walmart", "costco", "best buy", "canadian tire", "dollarama"]
    }

    def apply_logic(desc):
        desc = str(desc).lower()
        # Check Session State (Learned Rules) first
        for keyword, cat in st.session_state.custom_rules.items():
            if keyword.lower() in desc: return cat
        # Check Base Rules
        for cat, keywords in base_rules.items():
            if any(k in desc for k in keywords): return cat
        return None

    df["Category"] = df["Description"].apply(apply_logic)
    
    # AI Fallback for remaining "Other"
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
# 3. ROBUST PDF PARSER
# =========================================================

def process_pdf(file):
    all_rows = []
    # Enhanced Regex to capture dates like "Apr 04", descriptions, and amounts
    # It accounts for the multi-line nature of Triangle descriptions
    line_regex = re.compile(r"([A-Z][a-z]{2}\s\d{2})\s+([A-Z][a-z]{2}\s\d{2})\s+(.*?)\s+(-?\$?[\d,]+\.\d{2})")

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            # STRATEGY 1: Layout-Aware Table Extraction
            # We use 'text' as the strategy because Triangle statements 
            # often lack visible vertical lines.
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tol": 4, # Helps join nearby text into one cell
                "join_tolerance": 3
            })
            
            for table in tables:
                for row in table:
                    # Filter out empty cells and clean whitespace
                    clean_row = [str(c).replace('\n', ' ').strip() for c in row if c]
                    
                    # Target rows that look like transactions (Date, Date/Desc, Amount)
                    if len(clean_row) >= 3:
                        # Triangle often puts Date in col 0, 
                        # Posting Date + Description in col 1, and Amount in col 2
                        date = clean_row[0]
                        desc = clean_row[1]
                        amount = clean_row[-1]
                        
                        # Validate that the amount looks like a number
                        if re.search(r'\d+\.\d{2}', amount):
                            all_rows.append([date, desc, amount])

            # STRATEGY 2: Visual Line Fallback
            # If tables missed something (like the Returns section), we scrape by visual line
            if not all_rows or page.page_number >= 2:
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    for line in text.split('\n'):
                        # Apply regex to find transactions not caught in table grid
                        match = line_regex.search(line)
                        if match:
                            # groups: 0=Trans Date, 1=Posting Date, 2=Description, 3=Amount
                            all_rows.append([match.group(1), match.group(3), match.group(4)])
                        
    df = pd.DataFrame(all_rows, columns=["Date", "Description", "Amount"])
    
    # Cleaning the data
    # 1. Remove rows that are clearly headers
    noise = ["date", "description", "amount", "total", "posting", "subtotal"]
    df = df[~df['Description'].str.lower().str.contains('|'.join(noise), na=False)]
    
    # 2. Specifically capture "Interest Charges" and "Returns" from Triangle logic
    # Triangle represents returns with a minus sign (e.g., -0.63) 
    df["Amount"] = df["Amount"].apply(lambda x: str(x).replace('$', '').replace(',', '').strip())
    
    return df

# =========================================================
# 4. UI & WORKFLOW
# =========================================================

st.title("🏦 Smart AI Expense Analyzer (Toronto Region)")

# Sidebar: Category Teacher
with st.sidebar:
    st.header("🎓 Teach the AI")
    st.write("If a vendor shows as 'Other', map it here.")
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

# File Upload
files = st.file_uploader("Upload Statements (PDF, CSV, Excel)", type=["pdf", "csv", "xlsx", "xls"], accept_multiple_files=True)

if files:
    ai_model = get_ai_model()
    master_dfs = []

    for f in files:
        df = process_pdf(f) if f.name.endswith('pdf') else pd.read_excel(f) # Add logic for CSV/Excel as before
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
