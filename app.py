import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Diabetes Risk Prediction & Analytics Dashboard",
    page_icon="🩺",
    layout="wide"
)


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv("diabetes.csv")

    # Replace impossible zero values
    columns_with_zero = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in columns_with_zero:
        df[column] = df[column].replace(0, np.nan)

    return df


df = load_data()


# ==========================================
# TRAIN MODEL
# ==========================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probability)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("""
🩺 Diabetes Risk Prediction & Analytics

An interactive machine learning application
for diabetes risk assessment and data analysis
""")



page = st.sidebar.radio(
    "Navigation",
    [
        "About This Project",
        "Risk Calculator",
        "Dashboard",
        "Data Analysis",
        "Model Performance"
    ]
)


# ==========================================
# ABOUT THIS PROJECT
# ==========================================

if page == "About This Project":

    st.title("🩺 Diabetes Risk Prediction & Analytics Web Application")

    st.subheader("About This Project")

    st.write(
        """
        This project is an interactive web application that uses machine learning and data analysis to assess diabetes risk. It includes a risk prediction tool, data visualizations, and analytics to provide meaningful insights from diabetes related data.
        """
    )

    st.divider()

    # --------------------------------------
    # Project overview
    # --------------------------------------

    st.subheader("What You Can Explore")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🧮 1. Risk Calculator")

        st.write(
            """
            Enter selected health and demographic characteristics
            to obtain a model-based estimated probability of diabetes.
            """
        )

        st.markdown("### 📊 2. Dashboard")

        st.write(
            """
            Explore an overview of the dataset through summary
            statistics and visualizations.
            """
        )

        st.markdown("### 🔍 3. Data Analysis")

        st.write(
            """
            Examine the dataset, missing values, distributions,
            group comparisons, and correlations between variables.
            """
        )

    with col2:

        st.markdown("### 📈 4. Model Performance")

        st.write(
            """
            Review the machine-learning model using accuracy,
            ROC-AUC, a confusion matrix, and an ROC curve.
            """
        )

        st.markdown("### 🤖 Machine Learning Model")

        st.write(
            """
            A Logistic Regression model is used with median
            imputation and feature standardization.
            """
        )

    st.divider()
    

    st.subheader("Technologies Used")

    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

    tech_col1.metric("Language", "Python")
    tech_col2.metric("Dashboard", "Streamlit")
    tech_col3.metric("Model", "Logistic\nRegression")
    tech_col4.metric("Evaluation", "ROC-AUC")

    st.divider()

    st.caption("Developed by Meleesha Wijekoon")


# ==========================================
# RISK CALCULATOR
# ==========================================

elif page == "Risk Calculator":

    st.title("🧮 Diabetes Risk Calculator")

    st.write(
        """
        Enter the required characteristics below to obtain
        a model-based estimated probability.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=1.0,
            max_value=250.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=1.0,
            max_value=150.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=1.0,
            max_value=100.0,
            value=20.0
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=1.0,
            max_value=900.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=70.0,
            value=30.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    st.divider()

    if st.button(
        "🔍 Calculate Risk",
        type="primary"
    ):

        input_data = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [
                diabetes_pedigree
            ],
            "Age": [age]
        })

        probability = model.predict_proba(
            input_data
        )[0][1]

        prediction = model.predict(
            input_data
        )[0]

        st.subheader("Prediction Result")

        st.metric(
            "Estimated Probability",
            f"{probability * 100:.1f}%"
        )

        if prediction == 1:

            st.error(
                "Higher predicted risk"
            )

        else:

            st.success(
                "Lower predicted risk"
            )

        st.progress(
            float(probability)
        )


# ==========================================
# DASHBOARD
# ==========================================

elif page == "Dashboard":

    st.title("📊 Dashboard")

    st.write(
        "Overview of the diabetes dataset and key variables."
    )

    # --------------------------------------
    # Summary statistics
    # --------------------------------------

    total = len(df)

    diabetes_percentage = (
        df["Outcome"].mean() * 100
    )

    average_glucose = (
        df["Glucose"].mean()
    )

    average_bmi = (
        df["BMI"].mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        total
    )

    col2.metric(
        "Diabetes Outcome",
        f"{diabetes_percentage:.1f}%"
    )

    col3.metric(
        "Average Glucose",
        f"{average_glucose:.1f}"
    )

    col4.metric(
        "Average BMI",
        f"{average_bmi:.1f}"
    )

    st.divider()

    # --------------------------------------
    # Outcome chart
    # --------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Diabetes Outcome Distribution"
        )

        counts = df["Outcome"].value_counts()

        fig, ax = plt.subplots()

        ax.bar(
            ["No Diabetes", "Diabetes"],
            [
                counts.get(0, 0),
                counts.get(1, 0)
            ]
        )

        ax.set_ylabel("Number of Records")

        st.pyplot(fig)

    # --------------------------------------
    # Glucose distribution
    # --------------------------------------

    with col2:

        st.subheader(
            "Glucose Distribution"
        )

        fig, ax = plt.subplots()

        sns.histplot(
            data=df,
            x="Glucose",
            hue="Outcome",
            kde=True,
            ax=ax
        )

        st.pyplot(fig)


# ==========================================
# DATA ANALYSIS
# ==========================================

elif page == "Data Analysis":

    st.title("🔍 Data Analysis")

    # --------------------------------------
    # Dataset preview
    # --------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------
    # Missing values
    # --------------------------------------

    st.subheader("Missing Values")

    missing = df.isnull().sum()

    missing_table = pd.DataFrame({
        "Variable": missing.index,
        "Missing Values": missing.values
    })

    st.dataframe(
        missing_table,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------
    # Glucose
    # --------------------------------------

    st.subheader(
        "Glucose by Diabetes Outcome"
    )

    fig, ax = plt.subplots()

    sns.boxplot(
        data=df,
        x="Outcome",
        y="Glucose",
        ax=ax
    )

    ax.set_xticklabels(
        ["No Diabetes", "Diabetes"]
    )

    st.pyplot(fig)

    # --------------------------------------
    # BMI
    # --------------------------------------

    st.subheader(
        "BMI by Diabetes Outcome"
    )

    fig, ax = plt.subplots()

    sns.boxplot(
        data=df,
        x="Outcome",
        y="BMI",
        ax=ax
    )

    ax.set_xticklabels(
        ["No Diabetes", "Diabetes"]
    )

    st.pyplot(fig)

    # --------------------------------------
    # Correlation
    # --------------------------------------

    st.subheader("Correlation Matrix")

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.heatmap(
        df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)


# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "Performance of the Logistic Regression model on the test dataset."
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

    col2.metric(
        "ROC-AUC",
        f"{auc:.3f}"
    )

    st.divider()

    # --------------------------------------
    # Confusion matrix
    # --------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    st.pyplot(fig)

    # --------------------------------------
    # ROC curve
    # --------------------------------------

    st.subheader("ROC Curve")

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )

    fig, ax = plt.subplots()

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Receiver Operating Characteristic"
    )

    ax.legend()

    st.pyplot(fig)


# ==========================================
# FOOTER
# ==========================================

st.sidebar.divider()



st.sidebar.caption(
    "Developed by Meleesha Wijekoon"
)
