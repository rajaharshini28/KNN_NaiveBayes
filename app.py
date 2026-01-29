import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# ML Models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Preprocessing & Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_text

# ---------------------------------------------------
# Logger
# ---------------------------------------------------
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# ---------------------------------------------------
# Directory Setup
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

log("Application started")

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config("End-to-End ML Platform", layout="wide")
st.title("End-to-End Machine Learning Platform")

# ---------------------------------------------------
# Sidebar – Algorithm Settings
# ---------------------------------------------------
st.sidebar.header("Model Configuration")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["Decision Tree", "KNN", "Naive Bayes"]
)

# Decision Tree Params
if algorithm == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

# KNN Params
if algorithm == "KNN":
    k = st.sidebar.slider("Number of Neighbors (K)", 1, 15, 5)

# ---------------------------------------------------
# Step 1: Data Ingestion
# ---------------------------------------------------
st.header("Step 1: Data Ingestion")

option = st.radio("Choose Data Source", ["Download Iris Dataset", "Upload CSV"])
df = None

if option == "Download Iris Dataset":
    if st.button("Download Dataset"):
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)
        raw_path = os.path.join(RAW_DIR, "iris.csv")
        with open(raw_path, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(raw_path)
        st.success("Iris Dataset Downloaded Successfully")
        log("Iris dataset downloaded")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        raw_path = os.path.join(RAW_DIR, uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(raw_path)
        st.success("File Uploaded Successfully")
        log("Custom dataset uploaded")

# ---------------------------------------------------
# Step 2: Exploratory Data Analysis
# ---------------------------------------------------
if df is not None:
    st.header("Step 2: Exploratory Data Analysis")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing Values:", df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------
# Step 3: Data Cleaning
# ---------------------------------------------------
if df is not None:
    st.header("Step 3: Data Cleaning")

    strategy = st.selectbox(
        "Missing Value Strategy",
        ["Mean", "Median", "Drop Rows"]
    )

    df_clean = df.copy()

    if strategy == "Drop Rows":
        df_clean.dropna(inplace=True)
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy == "Mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

    st.session_state.df_clean = df_clean
    st.success("Data Cleaning Completed")

# ---------------------------------------------------
# Step 4: Save Cleaned Data
# ---------------------------------------------------
if st.button("Save Cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned dataset found")
    else:
        filename = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(CLEAN_DIR, filename)
        st.session_state.df_clean.to_csv(path, index=False)
        st.success("Cleaned Dataset Saved")
        log(f"Saved cleaned dataset at {path}")

# ---------------------------------------------------
# Step 5: Load Cleaned Dataset
# ---------------------------------------------------
st.header("Step 5: Load Cleaned Dataset")

clean_files = os.listdir(CLEAN_DIR)

if clean_files:
    selected_file = st.selectbox("Select Dataset", clean_files)
    df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected_file))
    st.dataframe(df_model.head())
else:
    st.warning("No cleaned datasets available")
    st.stop()

# ---------------------------------------------------
# Step 6: Model Training & Evaluation
# ---------------------------------------------------
st.header("Step 6: Model Training & Evaluation")

# Target Selection
categorical_cols = df_model.select_dtypes(include=["object", "category"]).columns
if len(categorical_cols) == 0:
    st.error("No categorical target column found")
    st.stop()

target = st.selectbox("Select Target Column", categorical_cols)

y = df_model[target]
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

X = df_model.drop(columns=[target])
X = X.select_dtypes(include=np.number)

if X.empty:
    st.error("No numeric features found")
    st.stop()

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Model Selection
if algorithm == "Decision Tree":
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=42
    )

elif algorithm == "KNN":
    model = KNeighborsClassifier(n_neighbors=k)

elif algorithm == "Naive Bayes":
    model = GaussianNB()

# Train
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.success(f"{algorithm} Accuracy: {accuracy:.2f}")
log(f"{algorithm} trained | Accuracy = {accuracy:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# ---------------------------------------------------
# Decision Tree Visualization
# ---------------------------------------------------
if algorithm == "Decision Tree":
    st.subheader("Decision Tree Visualization")

    feature_names = df_model.drop(columns=[target]).select_dtypes(include=np.number).columns
    class_names = [str(c) for c in np.unique(y)]

    fig, ax = plt.subplots(figsize=(22, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Decision Tree Rules (Text Format)")
    rules = export_text(model, feature_names=list(feature_names))
    st.text(rules)
