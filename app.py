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
    # Regex tailored for Triangle: Date | Date | Description | Amount 
    line_regex = re.compile(r"([A-Z][a-z]{2}\s\d{2})\s+([A-Z][a-z]{2}\s\d{2})\s+(.*?)\s+(-?[\d,]+\.\d{2})")

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            # Revised Table Settings to avoid TypeError [cite: 56]
            # Triangle tables often lack clear vertical lines, so we use 'text' for horizontal 
            # but rely on visual layout for column separation. [cite: 51, 108]
            ts = {
                "vertical_strategy": "lines", 
                "horizontal_strategy": "text",
                "snap_tolerance": 4,
                "join_tolerance": 3,
            }
            
            tables = page.extract_tables(table_settings=ts)
            
            for table in tables:
                for row in table:
                    # Clean the row of None values and extra newlines 
                    clean_row = [str(c).replace('\n', ' ').strip() for c in row if c]
                    
                    # Target rows with at least 3 columns (Date, Desc, Amount) [cite: 51, 143]
                    if len(clean_row) >= 3:
                        date_val = clean_row[0]
                        # Often col 1 is Posting Date and col 2 is Description [cite: 56]
                        # We merge them if needed or grab the last as amount [cite: 56, 113]
                        desc_val = clean_row[1] if len(clean_row) == 3 else " ".join(clean_row[1:-1])
                        amount_val = clean_row[-1]
                        
                        # Validate that the amount column actually contains a price [cite: 51, 56]
                        if re.search(r'\d+\.\d{2}', amount_val):
                            all_rows.append([date_val, desc_val, amount_val])

            # Strategy 2: Text Scrape for the "Returns" and "Interest" sections [cite: 53, 113]
            # Sometimes these aren't in a standard grid [cite: 53, 116]
            text = page.extract_text()
            if text:
                for line in text.split('\n'):
                    # Catch Interest Charges specifically [cite: 113]
                    if "INTEREST CHARGES" in line.upper():
                        parts = line.split()
                        if len(parts) >= 3:
                            all_rows.append([parts[0], "INTEREST CHARGES", parts[-1]])
                    
                    # Catch the Purchases that Regex can see 
                    match = line_regex.search(line)
                    if match:
                        all_rows.append([match.group(1), match.group(3), match.group(4)])
                        
    df = pd.DataFrame(all_rows, columns=["Date", "Description", "Amount"])
    
    # Final Data Cleaning for Triangle Formats [cite: 8, 51]
    def clean_triangle_amt(val):
        # Handle the $1,234.56- format for returns 
        s = str(val).replace('$', '').replace(',', '').strip()
        if s.endswith('-'): s = '-' + s[:-1]
        try: return float(s)
        except: return 0.0

    df["Amount"] = df["Amount"].apply(clean_triangle_amt)
    
    # Filter out rows that are actually page footers or headers [cite: 43, 97, 148]
    noise = ["date", "description", "amount", "total", "posting", "page", "statement"]
    df = df[~df['Description'].str.lower().str.contains('|'.join(noise), na=False)]
    
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
