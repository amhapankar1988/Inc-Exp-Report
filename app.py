import streamlit as st
import pandas as pd
import pdfplumber
import io
import datetime
import re

from ibm_watsonx_ai import APIClient, Credentials
from langchain_ibm import ChatWatsonx

# --------------------------------------------
# LLM INITIALIZATION (Optional intelligence)
# --------------------------------------------
def get_llm():
    api_key = st.secrets["WATSONX_APIKEY"]
    project_id = st.secrets["WATSONX_PROJECT_ID"]

    creds = Credentials(
        url="https://ca-tor.ml.cloud.ibm.com",
        api_key=api_key
    )

    client = APIClient(creds)

    return ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        watsonx_client=client,
        project_id=project_id,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 3000,
            "temperature": 0
        }
    )

# --------------------------------------------
# BANK DETECTION
# --------------------------------------------
def detect_bank_type(filename, text_sample):
    name = filename.lower()

    if "triangle" in name or "mastercard" in name:
        return "triangle"

    if "chequing" in name or "royal bank" in text_sample.lower():
        return "rbc_chequing"

    if "loan" in name or "operating loan" in text_sample.lower():
        return "rbc_loan"

    return "generic"

# --------------------------------------------
# TRIANGLE MASTERCARD PARSER
# --------------------------------------------
def parse_triangle_statement(pdf):
    rows = []
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue

        lines = text.split("\n")

        for line in lines:
            if any(m in line for m in months):

                parts = line.split()

                if len(parts) < 4:
                    continue

                try:
                    amount = float(parts[-1].replace(",", ""))
                except:
                    continue

                date = parts[0] + " " + parts[1]
                description = " ".join(parts[2:-1])

                rows.append({
                    "Date": date,
                    "Description": description,
                    "Amount": amount
                })

    return pd.DataFrame(rows)

# --------------------------------------------
# RBC CHEQUING PARSER
# --------------------------------------------
def parse_rbc_chequing(pdf):
    rows = []

    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue

        lines = text.split("\n")

        for line in lines:
            match = re.match(r"(\d{1,2})\s([A-Za-z].+)\s([\d,]+\.\d{2})", line)

            if match:
                day = match.group(1)
                description = match.group(2)
                amount = float(match.group(3).replace(",", ""))

                rows.append({
                    "Date": day,
                    "Description": description,
                    "Amount": amount
                })

    return pd.DataFrame(rows)

# --------------------------------------------
# RBC LOAN PARSER
# --------------------------------------------
def parse_rbc_loan(pdf):
    rows = []

    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue

        lines = text.split("\n")

        for line in lines:
            match = re.search(r"([A-Za-z]{3}\s\d{1,2}).*?(-?[\d,]+\.\d{2})", line)

            if match:
                rows.append({
                    "Date": match.group(1),
                    "Description": line,
                    "Amount": float(match.group(2).replace(",", ""))
                })

    return pd.DataFrame(rows)

# --------------------------------------------
# GENERIC FALLBACK PARSER
# --------------------------------------------
def parse_generic(pdf):
    rows = []

    for page in pdf.pages:
        table = page.extract_table()

        if not table:
            continue

        for row in table:
            if len(row) < 3:
                continue

            try:
                amount = float(str(row[-1]).replace(",", ""))
                date = row[0]
                desc = " ".join([str(x) for x in row[1:-1]])

                rows.append({
                    "Date": date,
                    "Description": desc,
                    "Amount": amount
                })
            except:
                pass

    return pd.DataFrame(rows)

# --------------------------------------------
# MASTER PARSER
# --------------------------------------------
def extract_transactions(file):

    with pdfplumber.open(file) as pdf:
        sample_text = pdf.pages[0].extract_text()

        bank = detect_bank_type(file.name, sample_text)

        if bank == "triangle":
            df = parse_triangle_statement(pdf)

        elif bank == "rbc_chequing":
            df = parse_rbc_chequing(pdf)

        elif bank == "rbc_loan":
            df = parse_rbc_loan(pdf)

        else:
            df = parse_generic(pdf)

    return df

# --------------------------------------------
# DATA NORMALIZATION
# --------------------------------------------
def normalize_dataframe(df):
    if df.empty:
        return df

    df.columns = [c.strip() for c in df.columns]

    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df

# --------------------------------------------
# SMART CANADIAN CATEGORIZATION
# --------------------------------------------
def classify_transactions(df):

    mapping = {
        "Groceries": [
            "walmart","freshco","fortinos","loblaws","metro",
            "costco","superstore","nofrills"
        ],
        "Dining": [
            "tim hortons","starbucks","restaurant","pizza",
            "jimmy the greek","bubble tea"
        ],
        "Shopping": [
            "amazon","apple","best buy","indigo"
        ],
        "Transportation": [
            "uber","lyft","esso","shell","petro"
        ],
        "Utilities": [
            "rogers","bell","hydro","enbridge"
        ],
        "Fees": [
            "fee","interest","service charge"
        ],
        "Transfers": [
            "transfer","payment","e-transfer"
        ]
    }

    def detect(desc):
        d = str(desc).lower()

        for category, keys in mapping.items():
            for k in keys:
                if k in d:
                    return category

        return "Other"

    df["Category"] = df["Description"].apply(detect)

    return df

# --------------------------------------------
# STREAMLIT UI
# --------------------------------------------
st.set_page_config(page_title="Canada AI Finance", layout="wide")
st.title("🇨🇦 Canadian Bank Statement Intelligence")

files = st.file_uploader(
    "Upload Statements",
    type=["pdf", "csv", "xlsx"],
    accept_multiple_files=True
)

if files:
    if st.button("🚀 Run AI Analysis"):

        master_df = pd.DataFrame()

        for file in files:

            with st.spinner(f"Parsing {file.name}..."):

                ext = file.name.split(".")[-1].lower()

                if ext == "pdf":
                    df = extract_transactions(file)
                elif ext == "csv":
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)

                df = normalize_dataframe(df)
                df = classify_transactions(df)

                df["Source"] = file.name

                master_df = pd.concat([master_df, df], ignore_index=True)

        master_df = master_df.dropna(subset=["Amount"])

        if not master_df.empty:

            master_df["Month-Year"] = master_df["Date"].dt.strftime("%Y-%m")

            pivot = master_df.pivot_table(
                index="Category",
                columns="Month-Year",
                values="Amount",
                aggfunc="sum",
                fill_value=0
            )

            tab1, tab2, tab3 = st.tabs(
                ["📊 Summary", "🧾 Transactions", "📥 Export"]
            )

            with tab1:
                st.subheader("Category Spending")
                st.dataframe(
                    pivot.style.format("${:,.2f}"),
                    use_container_width=True
                )

                trend = master_df.groupby("Month-Year")["Amount"].sum()
                st.bar_chart(trend)

            with tab2:
                st.dataframe(
                    master_df[
                        ["Date","Description","Category","Amount","Source"]
                    ],
                    use_container_width=True
                )

            with tab3:
                output = io.BytesIO()

                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    master_df.to_excel(writer, index=False, sheet_name="Transactions")
                    pivot.to_excel(writer, sheet_name="Summary")

                st.download_button(
                    "Download Excel Report",
                    output.getvalue(),
                    "Financial_Report.xlsx"
                )
