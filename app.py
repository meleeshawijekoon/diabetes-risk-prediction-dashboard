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
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    classification_report
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

        

    st.divider()
    

    st.subheader("Technologies Used")

    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

    tech_col1.metric("Language", "Python")
    tech_col2.metric("Dashboard", "Streamlit")
    tech_col3.markdown("""
<div style="text-align: center;">
    <div style="font-size: 14px;">Model</div>
    <div style="font-size: 22px; font-weight: 600;">
        Logistic<br>Regression
    </div>
</div>
""", unsafe_allow_html=True)

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

    st.title("📊 Diabetes Analytics Dashboard")

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
    # Outcome and Glucose
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

        ax.set_xlabel("Glucose")
        ax.set_ylabel("Number of Records")

        st.pyplot(fig)

    st.divider()

    # --------------------------------------
    # Glucose vs BMI
    # --------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Glucose vs BMI"
        )

        fig, ax = plt.subplots()

        sns.scatterplot(
            data=df,
            x="Glucose",
            y="BMI",
            hue="Outcome",
            ax=ax
        )

        ax.set_xlabel("Glucose")
        ax.set_ylabel("BMI")

        st.pyplot(fig)

    # --------------------------------------
    # Diabetes Outcome by Age Group
    # --------------------------------------

    with col2:

        st.subheader(
            "Diabetes Outcome by Age Group"
        )

        df_age = df.copy()

        df_age["Age Group"] = pd.cut(
            df_age["Age"],
            bins=[17, 29, 39, 49, 59, 100],
            labels=[
                "18–29",
                "30–39",
                "40–49",
                "50–59",
                "60+"
            ]
        )

        age_outcome = (
            df_age.groupby(
                "Age Group",
                observed=False
            )["Outcome"]
            .mean() * 100
        )

        fig, ax = plt.subplots()

        ax.bar(
            age_outcome.index,
            age_outcome.values
        )

        ax.set_xlabel("Age Group")
        ax.set_ylabel("Diabetes Outcome (%)")

        ax.set_ylim(0, 100)

        st.pyplot(fig)


# ==========================================
# DATA ANALYSIS
# ==========================================

elif page == "Data Analysis":

    st.title("🔍 Data Analysis")

    st.write(
        "Explore the dataset through descriptive statistics, "
        "missing values, distributions, group comparisons, "
        "and correlations."
    )

    # --------------------------------------
    # Dataset Preview
    # --------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------
    # Summary Statistics
    # --------------------------------------

    st.subheader("Summary Statistics")

    summary = df.describe().T

    summary = summary.round(2)

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------
    # Missing Values
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
    # Glucose and BMI Comparison
    # --------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Glucose by Diabetes Outcome")

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

        ax.set_xlabel("Diabetes Outcome")
        ax.set_ylabel("Glucose")

        st.pyplot(fig)

    with col2:

        st.subheader("BMI by Diabetes Outcome")

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

        ax.set_xlabel("Diabetes Outcome")
        ax.set_ylabel("BMI")

        st.pyplot(fig)

    st.divider()

    # --------------------------------------
    # Age and Blood Pressure Distributions
    # --------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Age Distribution")

        fig, ax = plt.subplots()

        sns.histplot(
            data=df,
            x="Age",
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Records")

        st.pyplot(fig)

    with col2:

        st.subheader("Blood Pressure Distribution")

        fig, ax = plt.subplots()

        sns.histplot(
            data=df,
            x="BloodPressure",
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Blood Pressure")
        ax.set_ylabel("Number of Records")

        st.pyplot(fig)

    st.divider()

    # --------------------------------------
    # Correlation Matrix
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
        "Evaluation of the Logistic Regression model "
        "using the test dataset."
    )

    # --------------------------------------
    # Performance Metrics
    # --------------------------------------

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

    col2.metric(
        "ROC-AUC",
        f"{auc:.3f}"
    )

    col3.metric(
        "Precision",
        f"{precision:.3f}"
    )

    col4.metric(
        "Recall",
        f"{recall:.3f}"
    )

    col5.metric(
        "F1-Score",
        f"{f1:.3f}"
    )

    st.divider()

    # --------------------------------------
    # Confusion Matrix
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
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.divider()

    # --------------------------------------
    # Classification Report
    # --------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Diabetes", "Diabetes"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    report_df = report_df.round(3)

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------
    # ROC Curve
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

    st.divider()

    # --------------------------------------
    # Prediction Probability Distribution
    # --------------------------------------

    st.subheader("Prediction Probability Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        y_probability,
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_xlabel(
        "Predicted Probability of Diabetes"
    )

    ax.set_ylabel(
        "Number of Test Records"
    )

    st.pyplot(fig)

    st.divider()

    # --------------------------------------
    # Model Information
    # --------------------------------------

    st.subheader("Model Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.write("**Algorithm:** Logistic Regression")
        st.write("**Training/Test Split:** 80% / 20%")
        st.write("**Preprocessing:** Median Imputation + Standard Scaling")

    with info_col2:

        st.write("**Maximum Iterations:** 1000")
        st.write("**Evaluation:** Accuracy, ROC-AUC, Precision, Recall and F1-score")
        st.write("**Test Samples:**", len(y_test))
