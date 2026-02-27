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

# 🏦 Multi-Statement Finance Analyzer (Powered by watsonx.ai)

An enterprise-grade financial tool that leverages the **IBM watsonx.ai** "Brain" to ingest multiple bank statements, extract transaction details, and automatically categorize spending.

## 🚀 Features
* **Multi-File Support**: Upload multiple PDFs, CSVs, or Excel files simultaneously.
* **Intelligent Extraction**: Uses `pdfplumber` and Llama-3-3-70b to understand unstructured banking layouts.
* **Auto-Categorization**: Automatically labels transactions into categories (e.g., Housing, Food, Salary).
* **Export to Excel**: Download a consolidated, cleaned report for all uploaded statements.

## 🛠️ Tech Stack
* **Brain**: [IBM watsonx.ai](https://www.ibm.com/watsonx) (meta-llama/llama-3-3-70b-instruct)
* **Frontend**: Streamlit
* **Data Processing**: Pandas, pdfplumber
* **Integration**: LangChain IBM

## 🔐 Setup Instructions

### 1. Environment Secrets
To run this Space, you must add the following **Secrets** in your Space Settings:

| Secret Name | Description |
| :--- | :--- |
| `WATSONX_APIKEY` | Your IBM Cloud IAM API Key |
| `PROJECT_ID` | Your watsonx.ai Project ID |
| `WATSONX_URL` | Your IBM Cloud region URL (Default: `https://us-south.ml.cloud.ibm.com`) |

### 2. Local Development
If you wish to run this locally:
1. Clone the repo.
2. Create a `.env` file with your credentials.
3. Install requirements: `pip install -r requirements.txt`.
4. Run: `streamlit run app.py`.

## 📝 Disclaimer
This application is an AI-powered tool. Always verify the categorized output against your original bank statements for accuracy before making financial decisions.
