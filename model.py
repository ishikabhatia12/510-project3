# AI helped generate this code.

# 1. Import packages
import os
import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Read MLflow and Databricks settings from the .env file when one exists
load_dotenv()


# 2. Load data
with open("config.yaml") as file:
    config = yaml.safe_load(file)

df = pd.read_csv(config["clean_data_path"])


# 3. Check data quality
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isna().sum())

# This CSV contains only three columns. A repeated combination of
# grades and outcome can occur in different original student records.
print("Repeated combinations in the selected columns:", df.duplicated().sum())


# 4. Select inputs and target
# X contains the two grades. y contains the recorded student outcome.
X = df[["Admission grade", "Previous qualification (grade)"]]
y = df["Target"]


# 5. Split data
# Set aside rows for validation and testing
reserved_size = config["validation_size"] + config["test_size"]

X_train, X_reserved, y_train, y_reserved = train_test_split(
    X,
    y,
    test_size=reserved_size,
    stratify=y,
    random_state=config["random_state"],
)

# Divide the reserved rows into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_reserved,
    y_reserved,
    test_size=config["test_size"] / reserved_size,
    stratify=y_reserved,
    random_state=config["random_state"],
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# 6. Explore training data
print("Training grade summary:\n", X_train.describe())
print("Training outcome counts:\n", y_train.value_counts())


# 7. Prepare and train the model
# The scaler learns from the training set and is saved with the classifier
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=config["max_iter"]),
)
model.fit(X_train, y_train)

print("Model trained.")


# 8. Make predictions and calculate accuracy
validation_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

validation_accuracy = accuracy_score(y_val, validation_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"Validation accuracy: {validation_accuracy:.2%}")
print(f"Test accuracy: {test_accuracy:.2%}")
print("Correct test predictions:", (test_predictions == y_test).sum())
print("Total test students:", len(y_test))


# 9. Log and register the model with MLflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", config["mlflow_experiment"])
model_name = os.getenv("MLFLOW_MODEL_NAME", config["registered_model_name"])

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    mlflow.log_params(config)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("validation_accuracy", validation_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # The registry requires a signature describing the model's inputs and outputs
    signature = infer_signature(X_train, model.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        model, name="model", signature=signature, input_example=X_train.head(3)
    )

# Register the logged model and mark this version as the champion
version = mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
mlflow.MlflowClient().set_model_version_tag(model_name, version.version, "confidence_threshold", str(config["confidence_threshold"]))
mlflow.MlflowClient().set_registered_model_alias(model_name, "champion", version.version)


print(f"Logged to {tracking_uri}; registered {model_name} v{version.version} as champion.")