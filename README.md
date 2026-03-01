---
title: AI Finance Multi-Analyzer
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
license: apache-2.0
header: mini
---

Smart AI Expense Analyzer (Universal Canadian Edition)
An intelligent Streamlit application designed to ingest unstructured financial data from all "Big 6" Canadian banks (RBC, TD, Scotiabank, BMO, CIBC, National Bank) plus specialized lenders like Canadian Tire Bank.

Unlike traditional parsers, this tool uses a Text-to-Data pattern, leveraging the Llama-3.1-8b model via IBM Watsonx to identify transactions directly from raw text.

🚀 Key Features
Universal Ingestion: Supports PDF, CSV, and Excel without requiring specific column headers.

Contextual AI Classification: Automatically maps transactions to industry-standard categories:

Utilities, Shopping, Health & Fitness, Interest Charge, Overdraft Fee, NSF, Transportation, Fees and Charges, Food and Dining, Groceries, Entertainment, Mortgage, Withdrawal, Deposits.

AI Training & Fine-Tuning: A sidebar "Memory" feature allows you to embed specific business rules into the AI prompt (e.g., "Mapping transfers to 5826 as Mortgage payments").

RBC Multi-Column Support: Intelligent enough to distinguish between "Cheques & Debits" and "Deposits & Credits" in complex RBC/Scotiabank PDF layouts.

Real-time Analytics: Visualizes spending habits and isolates high-priority items like bank fees (NSF/Overdraft).

🛠️ Installation & Setup
1. Prerequisites
Ensure you have Python 3.9+ and an active IBM Cloud / Watsonx.ai account.

2. Dependencies
Create a requirements.txt file and install:

Plaintext
streamlit
pdfplumber
pandas
openpyxl
langchain-ibm
ibm-watsonx-ai
pip install -r requirements.txt

3. Environment Secrets
For local development, create a .streamlit/secrets.toml file. If deploying to Streamlit Cloud, add these to the "Secrets" dashboard:

Ini, TOML
WATSONX_APIKEY = "your_ibm_cloud_api_key"
WATSONX_PROJECT_ID = "your_watsonx_project_guid"
🧠 Working Logic
Text Extraction: The app uses pdfplumber and pandas to convert any file format into a single stream of raw text.

Prompt Engineering: The text is wrapped in a specialized Instruction Prompt that defines the table schema and the 15 standard banking categories.

LLM Inference: The meta-llama/llama-3-1-8b base model analyzes the text. Because we use WatsonxLLM (Text-to-Text), it bypasses common chat-functionality errors.

Data Structuring: The AI returns a pipe-delimited (|) string which is then converted back into a structured Pandas DataFrame for visualization.

📂 Supported Banks (Tested)
RBC: Business Account Statements (including Operating Loans).

Canadian Tire: Triangle Mastercard PDFs.

Scotiabank: Scene+ Visa and Chequing CSVs/PDFs.

Universal: Any CSV containing Date, Description, and Amount.
