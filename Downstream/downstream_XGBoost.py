import pandas as pd
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders.leave_one_out import LeaveOneOutEncoder  # Explicit import
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--file','-f',type=str, required=True, help="The absolute path of the dataset to run downstream task.")
args = parser.parse_args()

# original dataset
data = pd.read_csv(f"{args.file}")
# don't need EMPLOYMENT_STATUS, because it only has one same value
data.drop(["cat_EMPLOYMENT_STATUS"],axis = 1, inplace = True)


# handling categorical data -> three ways:
# 1. if it's binary string -> map to 0,1 (the dominant category as 0)
convert_to_int_dict = {'URBAN': 1, 'RURAL': 0, # cat_REGION_TYPE
                       'Immigrated': 1, 'Member of the household': 0, # cat_MEM_STATUS
                       'M': 0, 'F': 1} # cat_GENDER
data['cat_REGION_TYPE'] = data.cat_REGION_TYPE.map(convert_to_int_dict)
data['cat_MEM_STATUS'] = data.cat_MEM_STATUS.map(convert_to_int_dict)
data['cat_GENDER'] = data.cat_GENDER.map(convert_to_int_dict)
# 2. if it's binary number 0,1 -> keep it
# 3. if it's multiclass (>4) -> Leave one out encoder
categorical_features = ["cat_STATE", "cat_HR", "cat_RELIGION", "cat_PLACE_OF_WORK"]

# Separate features and target
X = data.drop('cat_IS_HEALTHY', axis=1)
y = data['cat_IS_HEALTHY']

# Split the data - stratify to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Print class distribution
print("Class distribution in training set:")
print(pd.Series(y_train).value_counts(normalize=True))

# Create preprocessing steps
categorical_transformer = LeaveOneOutEncoder(
    cols=categorical_features,
    random_state=42,
    sigma=0.05
)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', StandardScaler(), [col for col in X.columns if col[:3] == 'con'])
    ])

# Create imbalanced pipeline with SMOTE
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),  # Adjust sampling_strategy as needed
    ('classifier', RandomForestClassifier(
        random_state=42,
        class_weight='balanced',  # Use balanced class weights
        n_estimators=200,         # Increase number of trees
        max_depth=None,           # Allow deep trees to capture rare patterns
        min_samples_leaf=1        # Allow leaf nodes to be smaller
    ))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Print classification report with detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and print additional metrics
print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Average Precision Score: {average_precision_score(y_test, y_pred_proba):.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Plot Precision-Recall curve
plt.subplot(2, 1, 2)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, label=f'AP={average_precision_score(y_test, y_pred_proba):.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('XGBoost.pdf')

# Feature importance
feature_importance = pd.DataFrame(
    pipeline.named_steps['classifier'].feature_importances_,
    index=categorical_features + [col for col in X.columns if col not in categorical_features],
    columns=['importance']
).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))