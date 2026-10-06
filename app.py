import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# PROJECT PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "customers.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.pkl")

SRC_PATH = os.path.join(BASE_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


# STREAMLIT CONFIG

st.set_page_config(
    page_title="AI Powered Business",
    page_icon="🤖",
    layout="wide"
)


# CUSTOM CSS

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# CACHED PROJECT FILE LOADERS

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_feature_columns():
    return joblib.load(FEATURE_COLUMNS_PATH)


@st.cache_resource
def load_metadata():

    if os.path.exists(METADATA_PATH):
        return joblib.load(METADATA_PATH)

    return {}


# LOAD PROJECT FILES

try:

    df = load_data()
    model = load_model()
    scaler = load_scaler()
    feature_columns = load_feature_columns()
    metadata = load_metadata()

except Exception as e:

    st.error(f"Unable to load project files: {e}")

    st.stop()


# CACHED RAG

@st.cache_resource
def load_rag():

    from genai.rag import CustomerRAG

    rag = CustomerRAG(
        data_path=DATA_PATH
    )

    return rag


# CACHED GENAI ASSISTANT

@st.cache_resource
def load_genai_assistant():

    from genai.llm_assistant import GenAICustomerAssistant

    assistant = GenAICustomerAssistant()

    return assistant


# SIDEBAR

st.sidebar.title("🤖 AI Powered Business")

st.sidebar.markdown(
    """
    **Customer Analytics & AI Platform**

    Built with:

    • Python  
    • SQL  
    • Machine Learning  
    • Deep Learning  
    • NLP  
    • Computer Vision  
    • RAG  
    • GenAI / LLM
    """
)

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Business Dashboard",
        "🔮 Churn Prediction",
        "🔎 Customer Search",
        "🤖 GenAI Assistant"
    ]
)


# PAGE 1 — BUSINESS DASHBOARD

if page == "📊 Business Dashboard":

    st.title("📊 AI Powered Business Dashboard")

    st.markdown(
        "Customer analytics, churn insights and business intelligence."
    )

    # KPI CARDS

    total_customers = len(df)

    churned_customers = (
        df["churn"]
        .astype(str)
        .str.lower()
        .eq("yes")
        .sum()
    )

    churn_rate = (
        churned_customers / total_customers * 100
        if total_customers > 0
        else 0
    )

    avg_satisfaction = df["satisfaction_score"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )

    with col3:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    with col4:
        st.metric(
            "Avg Satisfaction",
            f"{avg_satisfaction:.2f}"
        )

    st.divider()

    # CHURN DISTRIBUTION

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Customer Churn Distribution")

        churn_counts = (
            df["churn"]
            .value_counts()
            .reset_index()
        )

        churn_counts.columns = [
            "Churn",
            "Customers"
        ]

        fig = px.pie(
            churn_counts,
            names="Churn",
            values="Customers",
            title="Churn Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # CHURN VS SUPPORT TICKETS

    with col2:

        st.subheader("Churn Rate vs Support Tickets")

        support_analysis = (
            df.groupby("support_tickets")["churn"]
            .apply(
                lambda x:
                (x.astype(str).str.lower() == "yes").mean() * 100
            )
            .reset_index()
        )

        support_analysis.columns = [
            "Support Tickets",
            "Churn Rate"
        ]

        fig = px.line(
            support_analysis,
            x="Support Tickets",
            y="Churn Rate",
            markers=True,
            title="Churn Rate vs Support Tickets"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # SATISFACTION VS CHURN

    st.subheader("Churn Rate by Satisfaction Score")

    satisfaction_analysis = (
        df.groupby("satisfaction_score")["churn"]
        .apply(
            lambda x:
            (x.astype(str).str.lower() == "yes").mean() * 100
        )
        .reset_index()
    )

    satisfaction_analysis.columns = [
        "Satisfaction Score",
        "Churn Rate"
    ]

    fig = px.bar(
        satisfaction_analysis,
        x="Satisfaction Score",
        y="Churn Rate",
        title="Churn Rate by Satisfaction Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # CUSTOMER DATA

    st.subheader("Customer Data")

    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )


# PAGE 2 — CHURN PREDICTION

elif page == "🔮 Churn Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.markdown(
        "Enter customer information to predict churn risk."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        income = st.number_input(
            "Income",
            min_value=0,
            value=50000
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            value=24
        )

    with col2:

        support_tickets = st.number_input(
            "Support Tickets",
            min_value=0,
            value=2
        )

        website_visits = st.number_input(
            "Website Visits",
            min_value=0,
            value=20
        )

        app_usage = st.number_input(
            "App Usage Hours",
            min_value=0.0,
            value=5.0
        )

        satisfaction = st.number_input(
            "Satisfaction Score",
            min_value=0.0,
            value=5.0
        )

    if st.button(
        "🚀 Predict Churn",
        use_container_width=True
    ):

        try:

            gender_value = (
                1 if gender.lower() == "male"
                else 0
            )

            input_data = pd.DataFrame(
                {
                    "age": [age],
                    "gender": [gender_value],
                    "income": [income],
                    "tenure_months": [tenure],
                    "support_tickets": [support_tickets],
                    "website_visits": [website_visits],
                    "app_usage_hours": [app_usage],
                    "satisfaction_score": [satisfaction]
                }
            )

            # Align columns with training data

            input_data = input_data.reindex(
                columns=feature_columns,
                fill_value=0
            )

            # Scale

            input_scaled = scaler.transform(
                input_data
            )

            # Prediction

            prediction = model.predict(
                input_scaled
            )[0]

            # Probability

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(
                    input_scaled
                )[0][1]

            else:

                probability = 0.0

            probability_percent = probability * 100

            # Display

            if str(prediction).lower() == "yes":

                st.error(
                    "⚠️ Customer is predicted to CHURN"
                )

            else:

                st.success(
                    "✅ Customer is predicted NOT to churn"
                )

            st.metric(
                "Churn Probability",
                f"{probability_percent:.2f}%"
            )

            # Risk

            if probability_percent >= 70:

                risk = "HIGH"

            elif probability_percent >= 40:

                risk = "MEDIUM"

            else:

                risk = "LOW"

            st.info(
                f"Customer Churn Risk: **{risk}**"
            )

            # Recommendation

            st.subheader(
                "💡 Business Recommendation"
            )

            if risk == "HIGH":

                st.warning(
                    "Contact the customer immediately, "
                    "investigate support issues and provide "
                    "a retention offer."
                )

            elif risk == "MEDIUM":

                st.info(
                    "Monitor the customer and consider "
                    "a proactive engagement campaign."
                )

            else:

                st.success(
                    "Customer currently appears stable. "
                    "Continue normal engagement."
                )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# PAGE 3 — CUSTOMER SEARCH / RAG

elif page == "🔎 Customer Search":

    st.title("🔎 AI Customer Search")

    st.markdown(
        """
        Search customers using natural language.

        Example:
        `customers who are unhappy and have many support complaints`
        """
    )

    query = st.text_input(
        "Enter your search query",
        value="customers who are unhappy and have many support complaints"
    )

    top_k = st.slider(
        "Number of customers",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button(
        "🔍 Search Customers",
        use_container_width=True
    ):

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        else:

            try:

                with st.spinner(
                    "Searching customer knowledge base..."
                ):

                    # Cached RAG object
                    rag = load_rag()

                    results = rag.search(
                        query,
                        top_k=top_k
                    )

                st.success(
                    f"Found {len(results)} relevant customers."
                )

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        f"### Customer #{i}"
                    )

                    st.write(
                        f"**Customer ID:** "
                        f"{result['customer_id']}"
                    )

                    st.write(
                        f"**Similarity:** "
                        f"{result['similarity']:.4f}"
                    )

                    st.write(
                        result["document"]
                    )

                    st.divider()

            except Exception as e:

                st.error(
                    f"Customer search error: {e}"
                )


# PAGE 4 — GENAI ASSISTANT

elif page == "🤖 GenAI Assistant":

    st.title("🤖 GenAI Business Assistant")

    st.markdown(
        """
        Ask questions about customers using the
        **RAG + LLM architecture**.
        """
    )

    question = st.text_area(
        "Ask your question",
        value="Which customers appear most at risk and why?",
        height=100
    )

    if st.button(
        "🤖 Ask AI Assistant",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "AI assistant is analyzing customer data..."
                ):

                    # Cached GenAI assistant
                    assistant = load_genai_assistant()

                    answer, retrieved_results = (
                        assistant.generate_answer(
                            question
                        )
                    )

                # AI RESPONSE

                st.subheader(
                    "🤖 AI Response"
                )

                st.write(
                    answer
                )

                # EVIDENCE

                st.subheader(
                    "📚 Retrieved Customer Evidence"
                )

                for i, result in enumerate(
                    retrieved_results,
                    start=1
                ):

                    with st.expander(
                        f"Customer {i} — "
                        f"Similarity: "
                        f"{result['similarity']:.4f}"
                    ):

                        st.write(
                            result["document"]
                        )

            except Exception as e:

                st.error(
                    f"GenAI Assistant error: {e}"
                )


# FOOTER

st.divider()

st.caption(
    "AI Powered Business | "
    "Python • SQL • ML • DL • NLP • Computer Vision • "
    "RAG • GenAI"
)
