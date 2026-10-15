
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def prepare_data(df):
    # Copy the original dataset
    clean_df = df.copy()

    # Check that the target column exists
    if "glyhb" not in clean_df.columns:
        raise ValueError("Dataset must contain a 'glyhb' column.")

    # Remove rows without target values
    clean_df = clean_df.dropna(subset=["glyhb"])

    # Create the target variable
    clean_df["Outcome"] = (clean_df["glyhb"] >= 6.5).astype(int)

    # Remove identifiers and target-source columns
    cols_to_drop = ["id", "glyhb", "bp.2s", "bp.2d", "location"]
    clean_df = clean_df.drop(columns=cols_to_drop, errors="ignore")

    # Encode categorical features
    categorical_cols = [
        col for col in ["gender", "frame"]
        if col in clean_df.columns
    ]

    clean_df = pd.get_dummies(
        clean_df,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    # Remove remaining missing values
    clean_df = clean_df.dropna()

    # Separate features and target
    X = clean_df.drop(columns=["Outcome"])
    y = clean_df["Outcome"]

    # Check that both classes exist
    if y.nunique() < 2:
        raise ValueError(
            "The dataset must contain both outcome classes."
        )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale the features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data prepared successfully!")
    print("Training shape:", X_train_scaled.shape)
    print("Testing shape:", X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train_scaled, y_train, X_test_scaled, y_test):
    # Initialize the model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model


if __name__ == "__main__":
    # Load your dataset.
    # Change the filename to match your actual CSV file.
    df = pd.read_csv("dataset.csv")

    # Prepare the data FIRST
    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    ) = prepare_data(df)

    # Train and evaluate the model AFTER preparing the data
    model = train_model(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test
    )

    print("\nModel training completed!")
