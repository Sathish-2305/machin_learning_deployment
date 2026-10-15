
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
def prepare_data(df):
    # Create a copy of the original dataset
    clean_df = df.copy()

    # Remove rows where glyhb is missing
    clean_df = clean_df.dropna(subset=["glyhb"])

    # Create the binary target variable
    # 1 = diabetic threshold reached; 0 = below threshold
    clean_df["Outcome"] = (
        clean_df["glyhb"] >= 6.5
    ).astype(int)

    # Drop identifiers, target source, and unwanted columns
    cols_to_drop = [
        "id",
        "glyhb",
        "bp.2s",
        "bp.2d",
        "location"
    ]

    clean_df = clean_df.drop(
        columns=cols_to_drop,
        errors="ignore"
    )

    # Encode categorical columns if they exist
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

    # Remove remaining rows containing missing values
    clean_df = clean_df.dropna()

    # Separate features and target
    X = clean_df.drop(columns=["Outcome"])
    y = clean_df["Outcome"]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Standardize numerical features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data successfully prepared, split, and scaled!")
    print(f"Training shape: {X_train_scaled.shape}")
    print(f"Testing shape: {X_test_scaled.shape}")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


if __name__ == "__main__":
    # Make sure df has been loaded before calling this function.
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = (
        prepare_data(df)
    )
