import streamlit as st
import pdfplumber
import pandas as pd
import re
from datetime import datetime
from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =========================================================
# 1. INITIALIZATION & SECRETS
# =========================================================

st.set_page_config(page_title="AI Expense Analyzer", layout="wide")

# Define the user's specific categories
REQUIRED_CATEGORIES = [
    "Utilities", "Interest Charge", "Shopping", "Dining", 
    "Transportation", "Health and Wellbeing", "Mortgage", "Other"
]

def get_ai_model():
    try:
        # Accessing Streamlit Secret Variables
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
        st.error(f"Configuration Error: {e}")
        return None

# =========================================================
# 2. PARSING ENGINE (Layout Aware)
# =========================================================

def clean_amount(val):
    if val is None: return 0.0
    # Remove currency symbols and commas, keep decimals and negative signs
    clean = re.sub(r'[^\d.-]', '', str(val))
    try:
        return float(clean)
    except:
        return 0.0

def process_pdf(file):
    all_data = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            # Strategy A: Extract Tables
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    # Filter out empty rows or headers (usually rows with < 2 items)
                    clean_row = [r for r in row if r]
                    if len(clean_row) >= 3:
                        all_data.append(clean_row[:3])
            
            # Strategy B: Regex fallback for plain text lines
            if not tables:
                text = page.extract_text()
                # Matches Date, Description, and Amount (e.g., 01/24 Uber $15.00)
                pattern = re.compile(r"(\d{2}[/-]\d{2}|\d{4}-\d{2}-\d{2})\s+(.*?)\s+(-?\$?\d+\.\d{2})")
                for line in text.split('\n'):
                    match = pattern.search(line)
                    if match:
                        all_data.append(list(match.groups()))
                        
    return pd.DataFrame(all_data, columns=["Date", "Description", "Amount"])

# =========================================================
# 3. HYBRID AI CLASSIFICATION
# =========================================================

def classify_transactions_hybrid(df, model):
    # Local rules for instant matching (High confidence)
    rules = {
        "Dining": ["starbucks", "mcdonald", "pizza", "uber eats", "tim hortons", "restaurant"],
        "Transportation": ["uber", "lyft", "shell", "petro", "esso", "gas", "parking"],
        "Utilities": ["bell", "rogers", "hydro", "water", "electricity", "telus"],
        "Health and Wellbeing": ["pharmacy", "gym", "dentist", "shoppers", "hospital", "doctor"],
        "Mortgage": ["mortgage", "housing loan", "principal payment"]
    }

    def apply_rules(desc):
        desc = str(desc).lower()
        for cat, keywords in rules.items():
            if any(k in desc for k in keywords):
                return cat
        return None

    df["Category"] = df["Description"].apply(apply_rules)
    
    # Identify items that still need AI help
    mask = df["Category"].isnull()
    to_classify = df[mask]["Description"].unique().tolist()

    if to_classify and model:
        # Batch classification to save tokens/time
        formatted_list = "\n".join([f"- {d}" for d in to_classify])
        prompt = f"""Categorize these bank transactions into ONLY these categories: {', '.join(REQUIRED_CATEGORIES)}.
Return the results as a Python-style dictionary where key is the description and value is the category.

Transactions:
{formatted_list}
"""
        try:
            response = model.invoke(prompt)
            # Simple parsing of the returned text to find categories
            for desc in to_classify:
                for cat in REQUIRED_CATEGORIES:
                    if cat.lower() in response.content.lower() and desc.lower() in response.content.lower():
                        df.loc[df["Description"] == desc, "Category"] = cat
                        break
        except Exception as e:
            st.warning(f"AI Batch Error: {e}")

    df["Category"] = df["Category"].fillna("Other")
    return df

# =========================================================
# 4. USER INTERFACE
# =========================================================

st.title("🏦 AI Bank Statement Analyzer")
st.markdown("Extracts data using **pdfplumber** and categorizes via **IBM Granite AI**.")

uploaded_files = st.file_uploader("Upload Bank Statements (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    model = get_ai_model()
    master_df = pd.DataFrame()

    with st.spinner("Analyzing statements..."):
        for file in uploaded_files:
            df = process_pdf(file)
            if not df.empty:
                df["Amount"] = df["Amount"].apply(clean_amount)
                df = classify_transactions_hybrid(df, model)
                master_df = pd.concat([master_df, df], ignore_index=True)

    if not master_df.empty:
        # Interactive Editor
        st.subheader("📊 Transaction Breakdown")
        edited_df = st.data_editor(
            master_df,
            column_config={
                "Category": st.column_config.SelectboxColumn("Category", options=REQUIRED_CATEGORIES),
                "Amount": st.column_config.NumberColumn("Amount", format="$ %.2f")
            },
            use_container_width=True,
            hide_index=True
        )

        # Summary Metrics
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("### Spending by Category")
            summary = edited_df.groupby("Category")["Amount"].sum().abs()
            st.bar_chart(summary)

        with c2:
            st.write("### Total Summary")
            total_spent = summary.sum()
            st.metric("Total Expenses Detected", f"${total_spent:,.2f}")
            st.dataframe(summary.reset_index().rename(columns={"Amount": "Total ($)"}))

        # Export
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Excel/CSV", csv, "bank_analysis.csv", "text/csv")
    else:
        st.warning("No transactions could be parsed. Check the PDF format.")
