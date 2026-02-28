import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
from datetime import datetime
from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =========================================================
# CONFIGURATION & SCHEMA
# =========================================================

st.set_page_config(page_title="Pro AI Expense Analyzer", layout="wide")

# Updated Category List as per requirements
CATEGORIES = [
    "Utilities", "Interest Charge", "Shopping", "Dining", 
    "Transportation", "Health and Wellbeing", "Mortgage", "Income", "Other"
]

# Keyword-based Rules for high-confidence local matching
CATEGORY_RULES = {
    "Dining": ["starbucks", "mcdonald", "uber eats", "restaurant", "pizza", "tim hortons"],
    "Transportation": ["uber", "lyft", "petro", "esso", "shell", "gas", "transit"],
    "Shopping": ["amazon", "walmart", "costco", "best buy", "canadian tire"],
    "Utilities": ["bell", "rogers", "telus", "hydro", "water", "enbridge"],
    "Health and Wellbeing": ["shoppers", "pharmacy", "gym", "dentist", "hospital"],
    "Mortgage": ["mortgage", "chmc", "housing loan"],
}

# =========================================================
# AI INITIALIZATION (IBM Watsonx)
# =========================================================

def get_ai_model():
    # It's best to use environment variables for credentials
    api_key = os.getenv("WATSONX_APIKEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = "https://ca-tor.ml.cloud.ibm.com"

    if api_key == "WATSONX_APIKEY":
        return None

    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 200,
        GenParams.TEMPERATURE: 0
    }

    return ChatWatsonx(
        model_id="ibm/granite-3-8b-instruct", # Latest efficient model
        url=url,
        project_id=project_id,
        params=parameters,
    )

# =========================================================
# PROCESSING LOGIC
# =========================================================

def clean_amount(val):
    if not val: return 0.0
    # Handles negatives and currency symbols
    clean = re.sub(r'[^\d.-]', '', str(val))
    try: return float(clean)
    except: return 0.0

def extract_with_layout(page):
    """Uses spatial positioning to handle multi-column bank statements."""
    rows = []
    # Try table extraction first (best for structured grids)
    tables = page.extract_tables()
    for table in tables:
        for row in table:
            # Filter out empty or header rows
            if len(row) >= 3 and any(row):
                rows.append(row)
    
    # Fallback to regex if table extraction is sparse
    if len(rows) < 3:
        text = page.extract_text()
        pattern = re.compile(r"(\d{2}[/-]\d{2}|\d{4}-\d{2}-\d{2})\s+(.*?)\s+(-?\$?\d+\.\d{2})")
        for line in text.split('\n'):
            match = pattern.search(line)
            if match:
                rows.append(list(match.groups()))
    return rows

def classify_batch(descriptions, model):
    """Classifies multiple descriptions in one AI call to save time/quota."""
    if not model or not descriptions:
        return ["Other"] * len(descriptions)

    formatted_list = "\n".join([f"- {d}" for d in descriptions])
    prompt = f"""Categorize these bank transactions into exactly one of these: {', '.join(CATEGORIES)}.
Return only the category names as a comma-separated list in order.

Transactions:
{formatted_list}
"""
    try:
        response = model.invoke(prompt)
        results = [c.strip() for c in response.content.split(',')]
        # Ensure result length matches input
        return (results + ["Other"] * len(descriptions))[:len(descriptions)]
    except:
        return ["Other"] * len(descriptions)

# =========================================================
# STREAMLIT UI
# =========================================================

st.title("🏦 AI Bank Statement Intelligence")
st.markdown("Upload your PDF statements to automatically detect, categorize, and analyze spending.")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    model = get_ai_model()
    all_rows = []

    with st.spinner("Processing Documents..."):
        for file in uploaded_files:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    all_rows.extend(extract_with_layout(page))

    if all_rows:
        df = pd.DataFrame(all_rows).iloc[:, :3] # Take first 3 columns
        df.columns = ["Date", "Description", "Amount"]
        df["Amount"] = df["Amount"].apply(clean_amount)
        df = df.dropna(subset=["Description"]).drop_duplicates()

        # Step 1: Local Rule Classification
        def rule_check(d):
            d = str(d).lower()
            for cat, keywords in CATEGORY_RULES.items():
                if any(k in d for k in keywords): return cat
            return None

        df["Category"] = df["Description"].apply(rule_check)

        # Step 2: AI Classification for remaining "None"
        to_classify = df[df["Category"].isnull()]
        if not to_classify.empty and model:
            st.info(f"Using AI to classify {len(to_classify)} complex transactions...")
            ai_results = classify_batch(to_classify["Description"].tolist(), model)
            df.loc[df["Category"].isnull(), "Category"] = ai_results
        
        df["Category"] = df["Category"].fillna("Other")

        # UI: Interactive Editor
        st.subheader("📝 Review & Edit Transactions")
        st.caption("Double-click any cell to manually correct the AI's classification.")
        
        edited_df = st.data_editor(
            df,
            column_config={
                "Category": st.column_config.SelectboxColumn("Category", options=CATEGORIES),
                "Amount": st.column_config.NumberColumn("Amount", format="$ %.2f")
            },
            hide_index=True,
            use_container_width=True
        )

        # Dashboard Logic
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spending by Category")
            summary = edited_df.groupby("Category")["Amount"].sum().sort_values()
            st.bar_chart(summary)

        with col2:
            st.subheader("Summary Table")
            st.table(summary.map(lambda x: f"$ {x:,.2f}"))

        # Export
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Refined Report (CSV)", csv, "expense_report.csv", "text/csv")
    else:
        st.error("No transactions found. Try a different PDF layout.")

elif not os.getenv("WATSONX_APIKEY"):
    st.warning("⚠️ AI Model not connected. Set your IBM Watsonx API keys in environment variables to enable smart classification.")
