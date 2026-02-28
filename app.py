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

def clean_currency(val):
    """Helper to clean string currency values into floats."""
    if not val: return 0.0
    # Removes $, commas, and handles trailing minus or parentheses for negatives
    s = str(val).replace('$', '').replace(',', '').strip()
    if s.startswith('(') and s.endswith(')'): s = '-' + s[1:-1]
    if s.endswith('-'): s = '-' + s[:-1]
    try: return float(s)
    except: return 0.0

def process_pdf(file):
    all_rows = []
    # Your original regex for Triangle Mastercard
    line_regex = re.compile(r"([A-Z][a-z]{2}\s\d{2})\s+([A-Z][a-z]{2}\s\d{2})\s+(.*?)\s+(-?[\d,]+\.\d{2})")

    with pdfplumber.open(file) as pdf:
        # Check first page text to determine if it's an RBC statement [cite: 1, 28, 55]
        first_page_text = pdf.pages[0].extract_text() or ""
        is_rbc = "RBC" in first_page_text.upper() or "ROYAL BANK" in first_page_text.upper()

        for page in pdf.pages:
            if is_rbc:
                # RBC Logic: Multi-column Table Extraction 
                tables = page.extract_tables()
                for table in tables:
                    # Look for headers to identify transaction tables 
                    headers = [str(c).lower() for c in table[0] if c]
                    if "description" in headers:
                        for row in table[1:]:
                            clean_row = [str(c).replace('\n', ' ').strip() if c else "" for c in row]
                            if len(clean_row) >= 3:
                                date_val = clean_row[0]
                                desc_val = clean_row[1] or clean_row[2] # Fallback for different RBC layouts
                                
                                # Ignore balance/summary rows 
                                if any(x in desc_val.lower() for x in ["opening balance", "closing balance", "principal balance"]):
                                    continue

                                # Handle RBC Chequing (Debits in col 2, Credits in col 3) 
                                # Handle RBC Credit Line (Amount in col 3) 
                                amt = 0.0
                                if "transaction amount" in headers: # Credit Line format 
                                    amt = clean_currency(clean_row[headers.index("transaction amount")])
                                else: # Chequing/Savings format 
                                    debit = clean_currency(clean_row[2]) if len(clean_row) > 2 else 0
                                    credit = clean_currency(clean_row[3]) if len(clean_row) > 3 else 0
                                    amt = -abs(debit) if debit != 0 else abs(credit)

                                if amt != 0:
                                    all_rows.append([date_val, desc_val, amt])
            else:
                # Triangle Logic: Your original code remains untouched here
                # Strategy 1: Table Settings
                ts = {"vertical_strategy": "lines", "horizontal_strategy": "text", "snap_tolerance": 4, "join_tolerance": 3}
                tables = page.extract_tables(table_settings=ts)
                for table in tables:
                    for row in table:
                        clean_row = [str(c).replace('\n', ' ').strip() for c in row if c]
                        if len(clean_row) >= 3:
                            date_val, amt_val = clean_row[0], clean_row[-1]
                            desc_val = clean_row[1] if len(clean_row) == 3 else " ".join(clean_row[1:-1])
                            if re.search(r'\d+\.\d{2}', amt_val):
                                all_rows.append([date_val, desc_val, clean_currency(amt_val)])

                # Strategy 2: Text Scrape for Interest/Specific Regex
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        if "INTEREST CHARGES" in line.upper():
                            parts = line.split()
                            if len(parts) >= 3:
                                all_rows.append([parts[0], "INTEREST CHARGES", clean_currency(parts[-1])])
                        match = line_regex.search(line)
                        if match:
                            all_rows.append([match.group(1), match.group(3), clean_currency(match.group(4))])

    df = pd.DataFrame(all_rows, columns=["Date", "Description", "Amount"])
    
    # Final cleanup to remove noise rows 
    noise = ["date", "description", "amount", "total", "posting", "page", "balance", "cheques & debits"]
    df = df[~df['Description'].str.lower().str.contains('|'.join(noise), na=False)]
    return df

def process_excel_csv(file):
    filename = file.name.lower()
    try:
        if filename.endswith('.csv'):
            # Load CSV, skipping the first 'Filter' description row
            df = pd.read_csv(file, skiprows=1)
        else:
            df = pd.read_excel(file, engine='openpyxl')

        # 1. Clean up column names (remove extra spaces/newlines)
        df.columns = [str(c).strip() for c in df.columns]

        # 2. Identify columns by position or name
        # Your CSVs have: Filter(0), Date(1), Description(2), Sub-description(3), Type(4), Amount(5)
        # We search for keywords in case the order changes
        cols = df.columns.tolist()
        date_col = next((c for c in cols if "Date" in c), cols[1])
        desc_col = next((c for c in cols if "Description" in c), cols[2])
        sub_desc_col = next((c for c in cols if "Sub-description" in c), None)
        type_col = next((c for c in cols if "Type" in c), None)
        amount_col = next((c for c in cols if "Amount" in c), cols[-1])

        # 3. Create a clean copy with standard names
        new_df = pd.DataFrame()
        new_df["Date"] = df[date_col]
        
        # Combine Description and Sub-description for better AI context
        if sub_desc_col:
            new_df["Description"] = df[desc_col].fillna('') + " " + df[sub_desc_col].fillna('')
        else:
            new_df["Description"] = df[desc_col]

        # 4. Handle Amount Logic (Debit vs Credit)
        # In your CSV, 'Debit' amounts are often negative strings, 
        # but we ensure the logic is solid here.
        temp_amount = pd.to_numeric(df[amount_col].astype(str).str.replace('[$,]', '', regex=True), errors='coerce').fillna(0.0)
        
        if type_col:
            # Ensure Debits are negative and Credits are positive if they aren't already
            new_df["Amount"] = df.apply(
                lambda row: -abs(temp_amount[row.name]) if "Debit" in str(row[type_col]) else abs(temp_amount[row.name]), 
                axis=1
            )
        else:
            new_df["Amount"] = temp_amount

        return new_df[["Date", "Description", "Amount"]]

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
