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
        api_key = st.secrets["WATSONX_APIKEY"]
        project_id = st.secrets["WATSONX_PROJECT_ID"]
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
            params=parameters,
        )
    except Exception as e:
        st.error(f"AI Configuration Error: Check Streamlit Secrets. {e}")
        return None

# =========================================================
# 2. ENHANCED EXTRACTION ENGINE (The Fix)
# =========================================================

def clean_amount(val):
    if val is None or pd.isna(val): return 0.0
    # Capture negatives, decimals, and digits only
    clean = re.sub(r'[^\d.-]', '', str(val))
    try:
        # Handle cases like '12.50-' or '-12.50'
        if clean.endswith('-'):
            clean = '-' + clean[:-1]
        return float(clean)
    except:
        return 0.0

def extract_from_pdf(file):
    """Enhanced PDF extraction with multiple fallback strategies."""
    all_rows = []
    
    # Common Regex Pattern for Bank Lines: Date | Description | Amount
    # Matches: 2024-01-01 / Jan 01 / 01-01-24 + Text + $1,234.56
    line_regex = re.compile(r"(\d{1,4}[/-]\d{1,2}[/-]?\d{0,4}|[A-Z][a-z]{2}\s\d{1,2})\s+(.*?)\s+(-?\$?[\d,]+\.\d{2})")

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            # STRATEGY 1: Table Extraction (Best for structured grids)
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "lines", 
                "horizontal_strategy": "text",
                "snap_tol": 3
            })
            
            for table in tables:
                for row in table:
                    # Clean the row of None/Empty strings
                    clean_row = [str(cell).strip() for cell in row if cell and str(cell).strip()]
                    if len(clean_row) >= 3:
                        all_rows.append(clean_row[:3])

            # STRATEGY 2: Text-Line Regex (If Strategy 1 found nothing)
            if not all_rows:
                text = page.extract_text(layout=True) # preserve visual layout
                if text:
                    for line in text.split('\n'):
                        match = line_regex.search(line)
                        if match:
                            all_rows.append(list(match.groups()))

    # Convert to DF and remove common header words
    df = pd.DataFrame(all_rows, columns=["Date", "Description", "Amount"])
    df = df[~df['Description'].str.contains("Description|Balance|Transaction|Date", case=False, na=False)]
    return df

def parse_structured_file(file):
    """Handles CSV and Excel with header mapping."""
    file_ext = file.name.split('.')[-1].lower()
    df = pd.read_csv(file) if file_ext == 'csv' else pd.read_excel(file)
    
    new_df = pd.DataFrame()
    cols = df.columns.tolist()
    
    # Smart column mapping
    date_col = next((c for c in cols if 'date' in c.lower()), None)
    desc_col = next((c for c in cols if any(x in c.lower() for x in ['desc', 'memo', 'trans', 'details'])), None)
    amt_col = next((c for c in cols if any(x in c.lower() for x in ['amt', 'amount', 'value', 'charge', 'debit'])), None)
    
    if date_col and desc_col and amt_col:
        new_df["Date"] = df[date_col]
        new_df["Description"] = df[desc_col]
        new_df["Amount"] = df[amt_col]
        return new_df
    return pd.DataFrame()

# =========================================================
# 3. CLASSIFICATION & UI (Updated)
# =========================================================

def classify_transactions(df, model):
    rules = {
        "Dining": ["starbucks", "mcdonald", "pizza", "uber eats", "tim hortons", "dining", "restaurant", "pub", "bar"],
        "Transportation": ["uber", "lyft", "shell", "petro", "esso", "gas", "parking", "transit", "presto"],
        "Utilities": ["bell", "rogers", "hydro", "water", "electricity", "telus", "internet", "enbridge"],
        "Health and Wellbeing": ["pharmacy", "gym", "dentist", "shoppers", "hospital", "medical", "physio"],
        "Mortgage": ["mortgage", "housing loan", "home loan", "property tax"],
        "Interest Charge": ["interest charge", "finance charge", "interest pd", "service fee"]
    }

    def apply_rules(desc):
        desc = str(desc).lower()
        for cat, keywords in rules.items():
            if any(k in desc for k in keywords): return cat
        return None

    df["Category"] = df["Description"].apply(apply_rules)
    
    mask = df["Category"].isnull()
    unique_to_classify = df[mask]["Description"].unique().tolist()

    if unique_to_classify and model:
        formatted_list = "\n".join([f"- {d}" for d in unique_to_classify[:30]])
        prompt = f"Categorize into {', '.join(REQUIRED_CATEGORIES)}. Return as: 'Description | Category'.\n\n{formatted_list}"
