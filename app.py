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
    page_title="Diabetes Risk Prediction",
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

st.sidebar.title("🩺 Diabetes Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Data Analysis",
        "Risk Prediction",
        "Model Performance"
    ]
)

# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("🩺 Diabetes Risk Prediction Dashboard")

    st.write(
        "Interactive analysis and machine-learning prediction "
        "using clinical and demographic data."
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

    st.title("📊 Data Analysis")

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
# RISK PREDICTION
# ==========================================

elif page == "Risk Prediction":

    st.title("🤖 Diabetes Risk Prediction")

    st.write(
        "Enter the required patient characteristics "
        "to obtain a model-based prediction."
    )

    st.info(
        "This is a machine-learning demonstration and "
        "not a medical diagnosis."
    )

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
        "🔍 Predict Risk",
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

        st.subheader(
            "Prediction Result"
        )

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
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

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
