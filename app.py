
import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from datetime import datetime


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E-Commerce AI Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ECOM_DATA_DIR", BASE_DIR / "data"))

RFM_PATH = Path(
    os.getenv(
        "RFM_PATH",
        DATA_DIR / "processed" / "rfm_extended_features.csv"
    )
)

TRANSACTIONS_PATH = Path(
    os.getenv(
        "TRANSACTIONS_PATH",
        DATA_DIR / "processed" / "cleaned_ecommerce_data.csv"
    )
)

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

REQUEST_TIMEOUT = 30

CHURN_FEATURES = [
    "Frequency",
    "Monetary",
    "Tenure",
    "AOV",
    "MaxOrderValue",
    "StdDevSpend",
    "TotalUnits",
    "AvgBasketSize",
    "UniqueProducts",
    "AvgDaysBetweenOrders",
    "UnitsPerOrder"
]


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #172554;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 17px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }

    div[data-testid="stMetric"] label {
        color: #64748b;
    }

    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #64748b;
        font-size: 13px;
    }

    .status-good {
        color: #059669;
        font-weight: 700;
    }

    .status-bad {
        color: #dc2626;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. DATA LOADING
# ============================================================

@st.cache_data(ttl=300)
def load_rfm_data():
    if not RFM_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(RFM_PATH)

        required_columns = [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            st.error(
                f"RFM file is missing required columns: {missing}"
            )
            return pd.DataFrame()

        df["CustomerID"] = pd.to_numeric(
            df["CustomerID"], errors="coerce"
        )

        df = df.dropna(subset=["CustomerID"])
        df["CustomerID"] = df["CustomerID"].astype(int)

        numeric_columns = [
            col for col in df.columns
            if col != "CustomerID"
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

        df = df.drop_duplicates(
            subset=["CustomerID"],
            keep="first"
        )

        return df.reset_index(drop=True)

    except Exception as error:
        st.error(f"Could not load RFM data: {error}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_transactions():
    if not TRANSACTIONS_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(TRANSACTIONS_PATH)

        required_columns = [
            "InvoiceNo",
            "InvoiceDate",
            "Description",
            "Quantity",
            "UnitPrice",
            "CustomerID"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            st.warning(
                "Transaction file is missing columns: "
                + ", ".join(missing)
            )
            return pd.DataFrame()

        df["CustomerID"] = pd.to_numeric(
            df["CustomerID"], errors="coerce"
        )

        df["Quantity"] = pd.to_numeric(
            df["Quantity"], errors="coerce"
        )

        df["UnitPrice"] = pd.to_numeric(
            df["UnitPrice"], errors="coerce"
        )

        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"], errors="coerce"
        )

        df["Revenue"] = (
            df["Quantity"] * df["UnitPrice"]
        )

        df = df.dropna(
            subset=[
                "CustomerID",
                "InvoiceDate",
                "Description",
                "Quantity",
                "UnitPrice"
            ]
        )

        # Remove cancelled invoices
        df = df[
            ~df["InvoiceNo"].astype(str).str.startswith("C")
        ]

        # Remove invalid quantities and prices
        df = df[
            (df["Quantity"] > 0) &
            (df["UnitPrice"] > 0)
        ]

        df["CustomerID"] = df["CustomerID"].astype(int)

        return df.reset_index(drop=True)

    except Exception as error:
        st.warning(f"Could not load transaction data: {error}")
        return pd.DataFrame()


rfm_df = load_rfm_data()
transactions_df = load_transactions()


# ============================================================
# 4. API COMMUNICATION
# ============================================================

def api_get(endpoint, params=None):
    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        st.error(f"API GET error: {error}")
        return None

    except ValueError:
        st.error("The API returned an invalid JSON response.")
        return None


def api_post(endpoint, payload):
    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()
        return response.json()

    except requests.HTTPError as error:
        try:
            detail = response.json()
        except Exception:
            detail = response.text

        st.error(
            f"API request failed ({response.status_code}): {detail}"
        )
        return None

    except requests.RequestException as error:
        st.error(f"API connection error: {error}")
        return None

    except ValueError:
        st.error("The API returned an invalid JSON response.")
        return None


def check_api_health():
    result = api_get("/health")
    return result is not None


def predict_churn(customer):
    """
    Sends the 11 features required by the current FastAPI model.
    """

    missing = [
        col for col in CHURN_FEATURES
        if col not in customer.index or pd.isna(customer[col])
    ]

    if missing:
        st.error(
            "Cannot predict churn. Missing customer features: "
            + ", ".join(missing)
        )
        return None

    payload = {
        col: float(customer[col])
        for col in CHURN_FEATURES
    }

    return api_post("/predict/churn", payload)


def get_recommendations(customer_id, limit=5):
    """
    Calls the current SVD personalized recommendation endpoint.
    """

    payload = {
        "customer_id": int(customer_id),
        "top_n": int(limit)
    }

    return api_post("/recommend/user-feed", payload)


# ============================================================
# 5. CHURN RESPONSE HELPERS
# ============================================================

def extract_probability(prediction):
    """
    Attempts to read the churn probability from common API
    response field names.
    """

    if not isinstance(prediction, dict):
        return None

    possible_keys = [
        "churn_probability",
        "probability",
        "churn_prob",
        "risk_probability",
        "probability_of_churn"
    ]

    for key in possible_keys:
        if key in prediction:
            try:
                value = float(prediction[key])

                if value > 1:
                    value = value / 100

                return max(0, min(value, 1))

            except (ValueError, TypeError):
                pass

    # Some APIs return a nested result
    for key in ["result", "prediction", "data"]:
        nested = prediction.get(key)

        if isinstance(nested, dict):
            result = extract_probability(nested)

            if result is not None:
                return result

    return None


def get_risk_level(probability):
    if probability is None:
        return "Unavailable"

    risk = probability * 100

    if risk >= 35:
        return "High"

    elif risk >= 20:
        return "Medium"

    return "Low"


def get_risk_color(risk):
    if risk >= 35:
        return "#ef4444"

    elif risk >= 20:
        return "#f59e0b"

    return "#10b981"


def get_playbook(risk, monetary):
    if risk >= 35:
        return {
            "title": "High-risk customer",
            "description": (
                "This customer has a high predicted churn risk. "
                "Prioritize retention."
            ),
            "actions": [
                "Send a personalized retention campaign.",
                "Offer a targeted discount or loyalty reward.",
                "Recommend products based on previous purchases.",
                "Monitor customer engagement closely."
            ]
        }

    elif risk >= 20:
        return {
            "title": "Medium-risk customer",
            "description": (
                "This customer may need additional engagement "
                "to remain active."
            ),
            "actions": [
                "Send personalized product recommendations.",
                "Offer relevant promotional campaigns.",
                "Encourage repeat purchases.",
                "Monitor changes in purchasing frequency."
            ]
        }

    return {
        "title": "Low-risk customer",
        "description": (
            "This customer currently has a relatively low "
            "predicted churn risk."
        ),
        "actions": [
            "Maintain a positive customer experience.",
            "Recommend complementary products.",
            "Encourage loyalty-program participation.",
            "Continue monitoring purchasing behaviour."
        ]
    }


# ============================================================
# 6. GENERAL HELPER FUNCTIONS
# ============================================================

def format_currency(value):
    try:
        return f"£{float(value):,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def get_customer_transactions(customer_id):
    if transactions_df.empty:
        return pd.DataFrame()

    return transactions_df[
        transactions_df["CustomerID"] == int(customer_id)
    ].copy()


def get_customer_segment(recency):
    if pd.isna(recency):
        return "Unknown"

    if recency <= 30:
        return "Active"

    elif recency <= 90:
        return "At Risk"

    return "Inactive"


def create_churn_gauge(probability):
    percentage = probability * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentage,
            number={
                "suffix": "%",
                "font": {"size": 38}
            },
            title={
                "text": "Predicted Churn Probability"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color": "#334155"
                },
                "steps": [
                    {
                        "range": [0, 20],
                        "color": "#bbf7d0"
                    },
                    {
                        "range": [20, 35],
                        "color": "#fde68a"
                    },
                    {
                        "range": [35, 100],
                        "color": "#fecaca"
                    }
                ],
                "threshold": {
                    "line": {
                        "color": "#111827",
                        "width": 4
                    },
                    "thickness": 0.75,
                    "value": percentage
                }
            }
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=10
        )
    )

    return fig


def make_empty_message(message):
    st.info(message)


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("# 🛍️ E-Commerce AI")

    st.caption("Customer Intelligence Platform")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Executive Summary",
            "Customer 360 & AI Intelligence",
            "Power BI Analytics Hub",
            "Static Dashboard Gallery"
        ]
    )

    st.divider()

    st.markdown("### System Status")

    api_status = check_api_health()

    if api_status:
        st.markdown(
            '<span class="status-good">● API Connected</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-bad">● API Disconnected</span>',
            unsafe_allow_html=True
        )

    st.caption(f"API: {API_URL}")

    st.divider()

    st.markdown("### Dataset")

    st.caption(f"RFM: {RFM_PATH.name}")
    st.caption(f"Transactions: {TRANSACTIONS_PATH.name}")

    st.caption(
        f"Customers loaded: {len(rfm_df):,}"
    )

    st.caption(
        f"Transactions loaded: {len(transactions_df):,}"
    )


# ============================================================
# 8. EXECUTIVE SUMMARY
# ============================================================

if page == "Executive Summary":

    st.markdown(
        '<div class="main-title">Executive Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Business overview, customer behaviour and sales intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    if rfm_df.empty:
        st.error(
            "RFM dataset could not be loaded. "
            "Please check your RFM_PATH."
        )
        st.stop()

    total_customers = rfm_df["CustomerID"].nunique()

    total_revenue = (
        transactions_df["Revenue"].sum()
        if not transactions_df.empty
        else rfm_df["Monetary"].sum()
    )

    average_customer_value = rfm_df["Monetary"].mean()

    average_frequency = rfm_df["Frequency"].mean()

    active_customers = (
        rfm_df["Recency"] <= 30
    ).sum()

    at_risk_customers = (
        (rfm_df["Recency"] > 30) &
        (rfm_df["Recency"] <= 90)
    ).sum()

    inactive_customers = (
        rfm_df["Recency"] > 90
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Total Revenue",
        format_currency(total_revenue)
    )

    col3.metric(
        "Average Customer Value",
        format_currency(average_customer_value)
    )

    col4.metric(
        "Average Purchase Frequency",
        f"{average_frequency:.2f}"
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Customer Activity</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Active Customers",
        f"{active_customers:,}"
    )

    col2.metric(
        "At-Risk Customers",
        f"{at_risk_customers:,}"
    )

    col3.metric(
        "Inactive Customers",
        f"{inactive_customers:,}"
    )

    segment_df = pd.DataFrame({
        "Segment": [
            "Active",
            "At Risk",
            "Inactive"
        ],
        "Customers": [
            active_customers,
            at_risk_customers,
            inactive_customers
        ]
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            segment_df,
            names="Segment",
            values="Customers",
            hole=0.55,
            title="Customer Activity Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:
        fig = px.bar(
            segment_df,
            x="Segment",
            y="Customers",
            color="Segment",
            title="Customer Segments",
            text="Customers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.markdown(
        '<div class="section-title">Customer Value Distribution</div>',
        unsafe_allow_html=True
    )

    fig = px.histogram(
        rfm_df,
        x="Monetary",
        nbins=40,
        title="Distribution of Customer Monetary Value",
        labels={
            "Monetary": "Customer Monetary Value"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    if not transactions_df.empty:

        st.markdown(
            '<div class="section-title">Top-Selling Products</div>',
            unsafe_allow_html=True
        )

        top_products = (
            transactions_df.groupby("Description")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            top_products.sort_values("Revenue"),
            x="Revenue",
            y="Description",
            orientation="h",
            title="Top 10 Products by Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        make_empty_message(
            "Transaction-level data is unavailable. "
            "Revenue and product-level analysis are based on "
            "available RFM data or are omitted."
        )


# ============================================================
# 9. CUSTOMER 360 & AI INTELLIGENCE
# ============================================================

elif page == "Customer 360 & AI Intelligence":

    st.markdown(
        '<div class="main-title">Customer 360 & AI Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Individual customer profiling, machine-learning churn prediction '
        'and personalized product recommendations'
        '</div>',
        unsafe_allow_html=True
    )

    if rfm_df.empty:
        st.error("RFM dataset could not be loaded.")
        st.stop()

    customer_options = sorted(
        rfm_df["CustomerID"].unique().tolist()
    )

    selected_cust = st.selectbox(
        "Select Customer ID",
        customer_options
    )

    customer_rows = rfm_df[
        rfm_df["CustomerID"] == selected_cust
    ]

    if customer_rows.empty:
        st.error("Customer not found.")
        st.stop()

    customer = customer_rows.iloc[0]

    recency = customer.get("Recency", np.nan)
    frequency = customer.get("Frequency", np.nan)
    monetary = customer.get("Monetary", np.nan)

    customer_history = get_customer_transactions(
        selected_cust
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Customer Profile</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customer ID",
        str(int(selected_cust))
    )

    col2.metric(
        "Lifetime Monetary Value",
        format_currency(monetary)
    )

    col3.metric(
        "Purchase Frequency",
        f"{frequency:.0f}"
        if pd.notna(frequency)
        else "N/A"
    )

    col4.metric(
        "Days Since Last Purchase",
        f"{recency:.0f}"
        if pd.notna(recency)
        else "N/A"
    )

    segment = get_customer_segment(recency)

    st.markdown(
        f"**Customer Segment:** `{segment}`"
    )

    if not transactions_df.empty:

        if not customer_history.empty:

            favorite_product = (
                customer_history.groupby("Description")["Quantity"]
                .sum()
                .sort_values(ascending=False)
            )

            if not favorite_product.empty:
                st.markdown(
                    f"**Most Purchased Product:** "
                    f"{favorite_product.index[0]}"
                )

        else:
            st.caption(
                "No transaction history found for this customer."
            )

    else:
        st.caption(
            "Purchase history unavailable because transaction data "
            "could not be loaded."
        )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "AI Churn Prediction",
        "AI Recommendations",
        "Customer Profile",
        "Purchase History"
    ])


    # --------------------------------------------------------
    # TAB 1: CHURN PREDICTION
    # --------------------------------------------------------

    with tab1:

        st.markdown("### Machine Learning Churn Prediction")

        st.write(
            "The trained Gradient Boosting model evaluates "
            "customer behaviour using the 11 model features."
        )

        if not api_status:

            st.error(
                "API is disconnected. Start your FastAPI server "
                "before running predictions."
            )

        else:

            if st.button(
                "Run Churn Prediction",
                type="primary",
                key="run_churn"
            ):

                with st.spinner(
                    "Analyzing customer behaviour..."
                ):

                    prediction = predict_churn(customer)

                if prediction is not None:

                    st.session_state[
                        f"churn_{selected_cust}"
                    ] = prediction

            prediction = st.session_state.get(
                f"churn_{selected_cust}"
            )

            if prediction is not None:

                probability = extract_probability(
                    prediction
                )

                if probability is not None:

                    risk_percentage = probability * 100

                    risk_level = get_risk_level(
                        probability
                    )

                    color = get_risk_color(
                        risk_percentage
                    )

                    col1, col2 = st.columns(
                        [1.2, 1]
                    )

                    with col1:

                        st.plotly_chart(
                            create_churn_gauge(probability),
                            use_container_width=True
                        )

                    with col2:

                        st.markdown(
                            "### Prediction Result"
                        )

                        st.markdown(
                            f"<h2 style='color:{color}'>"
                            f"{risk_level} Risk"
                            f"</h2>",
                            unsafe_allow_html=True
                        )

                        st.metric(
                            "Churn Probability",
                            f"{risk_percentage:.2f}%"
                        )

                        st.write(
                            f"Customer ID: {int(selected_cust)}"
                        )

                    st.divider()

                    playbook = get_playbook(
                        risk_percentage,
                        monetary
                    )

                    st.markdown(
                        f"### Recommended Retention Strategy: "
                        f"{playbook['title']}"
                    )

                    st.write(playbook["description"])

                    for action in playbook["actions"]:
                        st.markdown(f"- {action}")

                else:

                    st.warning(
                        "The API responded, but the churn probability "
                        "could not be identified from its response. "
                        "Check the raw response below and match its "
                        "probability field name in extract_probability()."
                    )

                    st.json(prediction)


    # --------------------------------------------------------
    # TAB 2: AI RECOMMENDATIONS
    # --------------------------------------------------------

    with tab2:

        st.markdown(
            "### Personalized Product Recommendations"
        )

        st.write(
            "The recommendation engine generates a personalized "
            "product feed for the selected customer."
        )

        num_recs = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=5
        )

        if st.button(
            "Generate Recommendations",
            type="primary",
            key="generate_recommendations"
        ):

            if not api_status:

                st.error("API is disconnected.")

            else:

                with st.spinner(
                    "Generating personalized recommendations..."
                ):

                    rec_result = get_recommendations(
                        selected_cust,
                        num_recs
                    )

                if rec_result is not None:

                    st.session_state[
                        f"recommendations_{selected_cust}"
                    ] = rec_result

        rec_result = st.session_state.get(
            f"recommendations_{selected_cust}"
        )

        if rec_result is not None:

            rec_items = rec_result.get(
                "recommendations",
                []
            )

            if rec_items:

                rec_df = pd.DataFrame(rec_items)

                # Actual API response fields:
                # product_name and predicted_score

                rec_df = rec_df.rename(
                    columns={
                        "product_name": "Recommended Product",
                        "predicted_score": "Predicted Preference Score"
                    }
                )

                st.success(
                    f"{len(rec_df)} recommendations generated."
                )

                st.dataframe(
                    rec_df,
                    use_container_width=True,
                    hide_index=True
                )

                if (
                    "Predicted Preference Score" in rec_df.columns
                    and "Recommended Product" in rec_df.columns
                ):

                    fig = px.bar(
                        rec_df.sort_values(
                            "Predicted Preference Score"
                        ),
                        x="Predicted Preference Score",
                        y="Recommended Product",
                        orientation="h",
                        title="Personalized Recommendation Scores",
                        text="Predicted Preference Score"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            else:

                st.info(
                    "No recommendations were returned for this customer."
                )

                st.json(rec_result)


    # --------------------------------------------------------
    # TAB 3: CUSTOMER PROFILE
    # --------------------------------------------------------

    with tab3:

        st.markdown("### Customer Behavioural Features")

        profile_features = [
            "Recency",
            "Frequency",
            "Monetary",
            "Tenure",
            "AOV",
            "MaxOrderValue",
            "StdDevSpend",
            "TotalUnits",
            "AvgBasketSize",
            "UniqueProducts",
            "AvgDaysBetweenOrders",
            "RecencyToTenureRatio",
            "UnitsPerOrder"
        ]

        available_features = [
            col for col in profile_features
            if col in customer.index
        ]

        profile_df = pd.DataFrame({
            "Feature": available_features,
            "Value": [
                customer[col]
                for col in available_features
            ]
        })

        st.dataframe(
            profile_df,
            use_container_width=True,
            hide_index=True
        )

        if (
            "AOV" in customer.index
            and pd.notna(customer["AOV"])
        ):

            st.metric(
                "Average Order Value",
                format_currency(customer["AOV"])
            )

        elif (
            pd.notna(frequency)
            and frequency > 0
        ):

            st.metric(
                "Estimated Average Order Value",
                format_currency(monetary / frequency)
            )


    # --------------------------------------------------------
    # TAB 4: PURCHASE HISTORY
    # --------------------------------------------------------

    with tab4:

        st.markdown("### Customer Purchase History")

        if customer_history.empty:

            make_empty_message(
                "Purchase history unavailable. "
                "Check the transaction CSV path and its column names."
            )

        else:

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Transactions",
                f"{customer_history['InvoiceNo'].nunique():,}"
            )

            col2.metric(
                "Total Items",
                f"{customer_history['Quantity'].sum():,.0f}"
            )

            col3.metric(
                "Total Revenue",
                format_currency(
                    customer_history["Revenue"].sum()
                )
            )

            st.markdown("#### Recent Transactions")

            st.dataframe(
                customer_history.sort_values(
                    "InvoiceDate",
                    ascending=False
                )[
                    [
                        "InvoiceNo",
                        "InvoiceDate",
                        "Description",
                        "Quantity",
                        "UnitPrice",
                        "Revenue"
                    ]
                ].head(100),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("#### Spending Timeline")

            monthly = (
                customer_history
                .set_index("InvoiceDate")
                .resample("ME")["Revenue"]
                .sum()
                .reset_index()
            )

            if not monthly.empty:

                fig = px.line(
                    monthly,
                    x="InvoiceDate",
                    y="Revenue",
                    markers=True,
                    title="Monthly Customer Spending"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# ============================================================
# 10. POWER BI ANALYTICS HUB
# ============================================================

elif page == "Power BI Analytics Hub":

    st.markdown(
        '<div class="main-title">Power BI Analytics Hub</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Embedded business intelligence reports'
        '</div>',
        unsafe_allow_html=True
    )

    powerbi_url = os.getenv(
        "POWERBI_EMBED_URL",
        "https://app.powerbi.com/reportEmbed?reportId=4cd60f1d-7b4a-4e17-9251-a0ea5490d570&autoAuth=true&ctid=458b1bac-da7a-436a-a7d5-ca4121f19371"
    )

    if powerbi_url:

        st.components.v1.iframe(
            powerbi_url,
            height=750,
            scrolling=True
        )

    else:

        st.info(
            "Power BI report URL has not been configured."
        )

        st.markdown(
            """
            To embed your report:

            1. Publish your report to Power BI.
            2. Obtain an appropriate embed URL.
            3. Set the `POWERBI_EMBED_URL` environment variable.
            4. Restart Streamlit.

            Do not use a private report link as a public embed URL.
            Choose an embedding method appropriate for your report's
            access and security requirements.
            """
        )


# ============================================================
# 11. STATIC DASHBOARD GALLERY
# ============================================================

elif page == "Static Dashboard Gallery":

    st.markdown(
        '<div class="main-title">Static Dashboard Gallery</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Exploratory analytics and customer intelligence visualizations'
        '</div>',
        unsafe_allow_html=True
    )

    if rfm_df.empty:
        st.error("RFM dataset could not be loaded.")
        st.stop()

    st.markdown("### RFM Feature Distributions")

    available_rfm_features = [
        col for col in [
            "Recency",
            "Frequency",
            "Monetary",
            "Tenure",
            "AOV"
        ]
        if col in rfm_df.columns
    ]

    selected_feature = st.selectbox(
        "Select feature",
        available_rfm_features
    )

    fig = px.histogram(
        rfm_df,
        x=selected_feature,
        nbins=40,
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.markdown("### Customer Frequency vs Monetary Value")

    fig = px.scatter(
        rfm_df,
        x="Frequency",
        y="Monetary",
        color="Recency",
        hover_data=["CustomerID"],
        title="Customer Purchase Behaviour",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.markdown("### Recency vs Monetary Value")

    fig = px.scatter(
        rfm_df,
        x="Recency",
        y="Monetary",
        color="Frequency",
        hover_data=["CustomerID"],
        title="Customer Recency and Monetary Analysis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if not transactions_df.empty:

        st.divider()

        st.markdown("### Transaction Revenue Over Time")

        daily_revenue = (
            transactions_df
            .set_index("InvoiceDate")
            .resample("ME")["Revenue"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            daily_revenue,
            x="InvoiceDate",
            y="Revenue",
            markers=True,
            title="Monthly Revenue Trend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.markdown("### Dataset Preview")

    st.dataframe(
        rfm_df.head(100),
        use_container_width=True
    )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:15px;">
        <b>E-Commerce AI Intelligence Platform</b><br>
        Customer Analytics · Machine Learning ·
        Personalized Recommendations
    </div>
    """,
    unsafe_allow_html=True
)
