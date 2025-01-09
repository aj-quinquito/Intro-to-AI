import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

# Load the data
data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')

# Define feature engineering function
def feature_engineering(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Cabin_assigned'] = df['Cabin'].notnull().astype(int)
    df.drop('Cabin', axis=1, inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: x.split()[0] if len(x.split()) > 1 else 'NoPrefix')
    df['NameLength'] = df['Name'].apply(len)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[1, 2, 3, 4])
    df['AgeBin'] = pd.cut(df['Age'].astype(int), bins=[0, 16, 32, 48, 64, 80], labels=[1, 2, 3, 4, 5])
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

data = feature_engineering(data)
test_data = feature_engineering(test_data)

# One-hot encoding
data = pd.get_dummies(data, columns=['Sex', 'Embarked', 'Title', 'FareBin', 'AgeBin', 'TicketPrefix'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked', 'Title', 'FareBin', 'AgeBin', 'TicketPrefix'], drop_first=True)

# Ensure columns alignment
test_data = test_data.reindex(columns=data.columns, fill_value=0)
data.drop(['Name', 'Ticket'], axis=1, inplace=True, errors='ignore')
test_data.drop(['Name', 'Ticket'], axis=1, inplace=True, errors='ignore')

# Define features and target
features = [col for col in data.columns if col != 'Survived']
X = data[features]
y = data['Survived']

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data[features])

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
ada_model = AdaBoostClassifier(random_state=42)

# Hyperparameter tuning
param_grid_rf = {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5], 'bootstrap': [True, False]}
random_search_rf = RandomizedSearchCV(rf_model, param_grid_rf, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search_rf.fit(X_train, y_train)
best_rf_model = random_search_rf.best_estimator_

param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
random_search_xgb = RandomizedSearchCV(xgb_model, param_grid_xgb, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search_xgb.fit(X_train, y_train)
best_xgb_model = random_search_xgb.best_estimator_

# Stacking Model
stacking_model = StackingClassifier(
    estimators=[('rf', best_rf_model), ('gb', gb_model), ('xgb', best_xgb_model), ('ada', ada_model)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)
stacking_model.fit(X_train, y_train)

# Evaluate on validation set
y_pred_stack = stacking_model.predict(X_val)
print("Stacking Model Accuracy on Validation Set:", accuracy_score(y_val, y_pred_stack))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred_stack))
print("\nClassification Report:\n", classification_report(y_val, y_pred_stack))

# Cross-validation
cv_scores = cross_val_score(stacking_model, X_scaled, y, cv=10, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))

# Make predictions for submission
test_predictions = stacking_model.predict(test_data_scaled)

# Create submission file
submission = pd.DataFrame({'PassengerId': pd.read_csv('titanic/test.csv')['PassengerId'], 'Survived': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created: 0398108.csv")
