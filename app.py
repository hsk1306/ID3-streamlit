import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

st.title("ID3 Decision Tree using Streamlit")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data)

    # Select target column
    target_column = st.selectbox("Select Target Column", data.columns)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Convert categorical data to numbers
    le = LabelEncoder()
    for col in X.columns:
        X[col] = le.fit_transform(X[col].astype(str))

    y = le.fit_transform(y.astype(str))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ID3 = entropy
    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X_train, y_train)

    # Accuracy
    accuracy = model.score(X_test, y_test)
    st.success(f"Model Accuracy: {accuracy:.2f}")

    # Plot tree
    fig, ax = plt.subplots(figsize=(12, 8))
    tree.plot_tree(model, feature_names=X.columns, filled=True)
    st.pyplot(fig)
