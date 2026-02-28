import streamlit as st
import pdfplumber
import pandas as pd
import re
from datetime import datetime
from thefuzz import process

# --- AI CONFIG ---
AI_AVAILABLE = True
try:
    from ibm_watsonx_ai.foundation_models import Model
except:
    AI_AVAILABLE = False

st.set_page_config(page_title="Pro Finance AI", layout="wide")

# =========================================================
# 1. BANK ADAPTER DEFINITIONS
# =========================================================
# Each bank has a unique way of identifying itself and a unique column structure.
BANK_MAP = {
    "RBC": {
        "keywords": ["ROYAL BANK", "RBC"],
        "cols": ["Date", "Description", "Withdrawals", "Deposits", "Balance"]
    },
    "TD": {
        "keywords": ["TD CANADA TRUST", "TORONTO-DOMINION"],
        "cols": ["Date", "Description", "Debit", "Credit", "Balance"]
    },
    "CIBC": {
        "keywords": ["CIBC", "CANADIAN IMPERIAL"],
        "cols": ["Date", "Description", "Purchases", "Payments", "Balance"]
    }
}

KNOWN_MERCHANTS = [
    "Amazon", "Uber", "Uber Eats", "Lyft", "Walmart", "Costco", "Netflix", 
    "Spotify", "Starbucks", "McDonalds", "Tim Hortons", "Shell", "Petro-Canada",
    "Bell", "Rogers", "Telus", "Hydro", "Airbnb", "DoorDash", "Instacart"
]

# =========================================================
# 2. CORE UTILITIES & NORMALIZATION
# =========================================================

def detect_bank(text):
    """Identifies the bank based on text content."""
    for bank, config in BANK_MAP.items():
        if any(k in text.upper() for k in config["keywords"]):
            return bank
    return "Generic"

def normalize_merchant(desc):
    """Cleans 'AMZN MKTP CA*1234' into 'Amazon'."""
    if not desc: return "Unknown"
    # Basic cleanup: remove numbers, special chars, and extra spaces
    clean_desc = re.sub(r'[^a-zA-Z\s]', '', desc).strip()
    
    # Fuzzy matching against known high-frequency merchants
    match, score = process.extractOne(clean_desc, KNOWN_MERCHANTS)
    return match if score > 85 else clean_desc

@st.cache_data
def ai_classify_batch(descriptions):
    """Mock/Stub for Batch AI processing to save tokens/time."""
    # In production, you would send 20 descriptions at once to WatsonX
    # and return a list of categories.
    pass

# =========================================================
# 3. ADVANCED EXTRACTION ENGINE
# =========================================================

def extract_intelligent(uploaded_file):
    all_rows = []
    bank_detected = "Generic"
    
    with pdfplumber.open(uploaded_file) as pdf:
        # 1. Peek at first page to detect bank
        first_page_text = pdf.pages[0].extract_text() or ""
        bank_detected = detect_bank(first_page_text)
        
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                df_page = pd.DataFrame(table)
                
                # Filter out rows that are clearly not transactions (empty or header-only)
                for _, row in df_page.iterrows():
                    row_str = " ".join([str(x) for x in row if x])
                    
                    # Look for date-like patterns to identify transaction rows
                    if re.search(r'(\d{2}[/-]\d{2}|\w{3}\s\d{2})', row_str):
                        # Extract amounts (assuming last two columns are usually money)
                        amounts = [re.sub(r'[^\d.-]', '', str(x)) for x in row if x and any(c.isdigit() for c in str(x))]
                        
                        all_rows.append({
                            "Raw_Date": row[0],
                            "Raw_Description": row[1] if len(row) > 1 else "",
                            "Amount": amounts[0] if amounts else 0,
                            "Bank": bank_detected
                        })

    return pd.DataFrame(all_rows)

# =========================================================
# 4. ENRICHMENT & ANALYTICS
# =========================================================

def enrich_data(df):
    if df.empty: return df
    
    # Clean Amounts
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    
    # Merchant Normalization
    df['Clean_Merchant'] = df['Raw_Description'].apply(normalize_merchant)
    
    # Fraud/Anomaly Detection
    # Flag transactions that are 2x the standard deviation of the user's spending
    mean_spend = df[df['Amount'] < 0]['Amount'].mean()
    std_spend = df[df['Amount'] < 0]['Amount'].std()
    df['Potential_Anomaly'] = df['Amount'].apply(lambda x: "⚠️ High" if x < (mean_spend - 2*std_spend) else "Normal")
    
    return df

# =========================================================
# 5. UI INTERFACE
# =========================================================

st.title("🚀 Pro-Fintech AI Analyzer")
st.markdown("### Auto-detects Bank | Normalizes Merchants | Fraud Insights")

files = st.file_uploader("Upload Statements (RBC, TD, CIBC supported)", type="pdf", accept_multiple_files=True)

if files:
    final_dfs = []
    
    with st.status("Processing Financial Data...", expanded=True) as status:
        for f in files:
            st.write(f"Reading {f.name}...")
            raw_df = extract_intelligent(f)
            enriched_df = enrich_data(raw_df)
            final_dfs.append(enriched_df)
        
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    full_data = pd.concat(final_dfs)

    # Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(full_data))
    col2.metric("Banks Detected", ", ".join(full_data['Bank'].unique()))
    col3.metric("Largest Expense", f"${full_data['Amount'].min():,.2f}")

    # Display Data
    st.subheader("Normalized Transaction Feed")
    st.dataframe(full_data[['Raw_Date', 'Clean_Merchant', 'Amount', 'Bank', 'Potential_Anomaly']], use_container_width=True)

    # Insights Section
    st.subheader("💡 Spending Insights")
    top_merchants = full_data.groupby('Clean_Merchant')['Amount'].sum().sort_values().head(5)
    st.bar_chart(top_merchants)

    # Export
    csv = full_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clean Data (CSV)", csv, "cleaned_finance_data.csv", "text/csv")
