🛍️ E-Commerce Customer Intelligence

An end-to-end Data Science and Machine Learning platform for customer analytics, churn prediction, and personalized product recommendations.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" />
  <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/Analytics-Power%20BI-F2C811?logo=powerbi" />
  <img src="https://img.shields.io/badge/Status-Development-success" />
</p>---

📌 Overview

E-Commerce Customer Intelligence is an end-to-end data science project that transforms historical e-commerce transaction data into actionable customer insights.

The platform combines customer analytics, machine learning, recommendation algorithms, business intelligence, and API development to help understand customer purchasing behaviour, identify potential churn risks, and generate personalized product recommendations.

Rather than focusing only on sales reporting, this project explores how machine learning and data-driven methods can support customer retention, customer segmentation, cross-selling, and personalized shopping experiences.

🎯 Project Objectives

- Understand customer purchasing behaviour through exploratory data analysis.
- Develop customer-level features using RFM analysis and feature engineering.
- Identify customers who may be at risk of churning.
- Interpret model predictions and investigate customer-level risk factors.
- Develop multiple recommendation approaches.
- Expose trained models through REST APIs.
- Build an interactive customer intelligence dashboard.
- Present business insights through visual analytics and Power BI.

✨ Key Features

1. Data Analytics and Feature Engineering

- Data cleaning and preprocessing.
- Exploratory Data Analysis (EDA).
- Customer-level aggregation.
- Recency, Frequency, and Monetary (RFM) analysis.
- Extended customer behavioural features.
- Purchasing patterns and revenue analysis.
- Customer value and activity segmentation.

2. Customer Churn Prediction

A supervised machine learning pipeline designed to identify customers with a higher likelihood of churn.

Model development includes:

- Model comparison and experimentation.
- Gradient Boosting classification.
- Hyperparameter tuning.
- Classification threshold analysis.
- Model evaluation using classification metrics.
- ROC-AUC analysis.
- SHAP-based model interpretation.

Customer-level outputs:

- Predicted churn probability.
- Churn classification.
- Risk categorization.
- Customer-level retention insights.

3. Three Recommendation Engines

The platform explores three different recommendation approaches.

Engine| Algorithm| Purpose
Engine 1| Item-to-item Cosine Similarity| Recommend products similar to a selected product
Engine 2| FP-Growth and Association Rules| Discover product associations and cross-selling opportunities
Engine 3| Truncated SVD| Generate personalized product recommendations

These approaches demonstrate different ways of extracting recommendation signals from historical purchasing behaviour.

4. Customer 360 Intelligence Dashboard

An interactive Streamlit application that brings customer analytics and machine learning outputs together.

Dashboard modules:

- Executive Summary
- Customer 360 & AI Intelligence
- AI Churn Prediction
- Personalized Product Recommendations
- Customer Behavioural Profile
- Purchase History
- Power BI Analytics Hub
- Static Dashboard Gallery

5. FastAPI Backend

A REST API that serves trained machine learning models and recommendation engines.

Implemented endpoints:

Method| Endpoint| Purpose
GET| "/"| API status
GET| "/health"| Model and service health
POST| "/predict/churn"| Predict customer churn
POST| "/recommend/similar-items"| Find similar products
POST| "/recommend/cart-bundles"| Generate product cross-sell suggestions
POST| "/recommend/user-feed"| Generate personalized recommendations

Interactive API documentation is available through FastAPI Swagger UI.

6. Business Intelligence

Power BI integration and interactive visualizations provide additional perspectives on customer behaviour, sales performance, and business trends.

---

🧰 Technology Stack

Category| Technologies
Programming| Python
Data Manipulation| Pandas, NumPy
Visualization| Matplotlib, Seaborn, Plotly
Machine Learning| Scikit-learn
Recommendation Systems| Cosine Similarity, FP-Growth, Association Rules, Truncated SVD
Model Interpretation| SHAP
Model Persistence| Joblib
Backend| FastAPI, Pydantic
Frontend| Streamlit
Business Intelligence| Microsoft Power BI
Development| Jupyter Notebook, Git, GitHub

---

📂 Project Structure

Commerce-Customer-Intelligence/
│
├── notebooks/
│   ├── 01_data_cleaning_and_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_customer_analytics_and_feature_engineering.ipynb
│   ├── 04_churn_prediction.ipynb
│   └── 05_recommendation_systems.ipynb
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── dashboards/
│
├── reports/
│   └── figures/
│
├── app.py
│
└── README.md

Directory names above represent the intended organization; adjust notebook filenames or folders to match the actual committed files.

---

📊 Dataset

This project uses the Online Retail dataset from the UCI Machine Learning Repository.

The dataset contains historical transactions from a UK-based online retail business covering December 2010 to December 2011.

Dataset characteristics:

- Invoice and transaction identifiers.
- Product descriptions.
- Customer identifiers.
- Purchase quantities.
- Unit prices.
- Invoice timestamps.
- Transaction-level purchasing information.

Currency: British Pound Sterling (GBP).

The transaction data is processed and transformed into customer-level analytical features for modelling and recommendation tasks.

Dataset source: "UCI Machine Learning Repository – Online Retail" (https://archive.ics.uci.edu/dataset/352/online+retail)

---

🚀 Getting Started

1. Clone the repository

git clone https://github.com/nikunjpanthi5/Commerce-Customer-Intelligence.git

cd Commerce-Customer-Intelligence

2. Create a virtual environment

python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

3. Install dependencies

Install the project's required Python packages:

pip install pandas numpy scikit-learn scipy
pip install matplotlib seaborn plotly streamlit
pip install fastapi uvicorn pydantic joblib requests
pip install mlxtend shap

4. Configure the data and model paths

Ensure the processed datasets and trained model artifacts are available.

The application supports configurable paths through environment variables.

Important variables include:

Variable| Purpose
"ECOM_DATA_DIR"| Main data directory
"RFM_PATH"| Processed RFM dataset
"TRANSACTIONS_PATH"| Cleaned transaction dataset
"MODELS_DIR"| Trained model directory
"API_URL"| FastAPI backend URL
"POWERBI_EMBED_URL"| Power BI report embed URL

Configure the paths according to your local environment.

5. Start the FastAPI backend

From the repository root:

uvicorn api.main:app --reload --port 8000

API documentation:

http://127.0.0.1:8000/docs

6. Start the Streamlit application

Open another terminal:

streamlit run app.py

The application should be available at:

http://localhost:8501

Note: The backend requires the corresponding trained model artifacts, and the frontend requires the processed datasets. Running the application does not automatically train the models.

---

🧠 Machine Learning and Recommendation Concepts

This project explores several important data science concepts:

- Customer segmentation and RFM analysis.
- Behavioural feature engineering.
- Supervised classification.
- Ensemble learning.
- Hyperparameter optimization.
- Classification threshold selection.
- Model interpretability.
- Item-based collaborative filtering.
- Association rule mining.
- Matrix factorization.
- REST API integration.
- Business-oriented customer analytics.

---

🔮 Future Improvements

Potential areas for further development include:

- Temporal validation of churn predictions.
- Recommendation evaluation using Precision@K, Recall@K, MAP@K, and NDCG@K.
- Cold-start recommendation strategies.
- Content-based recommendation integration.
- Automated model retraining pipelines.
- Model monitoring and performance tracking.
- Authentication and API security.
- Cloud deployment and scalability improvements.
- A/B testing of customer retention strategies.

---

👨‍💻 Author

Nikunja Panthi

BIT Student | Data Science & Machine Learning Enthusiast

- GitHub: "@nikunjpanthi5" (https://github.com/nikunjpanthi5)
- LinkedIn: "Nikunja Panthi" (https://www.linkedin.com/in/nikunja-panthi-47107a294/)
- Portfolio: "nikunjapanthi.com.np" (https://nikunjapanthi.com.np)

---


---

<p align="center">
  <b>Turning E-Commerce Transactions into Customer Intelligence.</b>
</p>
