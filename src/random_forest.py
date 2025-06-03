import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath):
    """Load and preprocess the data"""
    data = pd.read_csv(filepath)
    
    # Drop rows with missing values
    data = data.dropna()
    
    # Separate features and target
    X = data.iloc[:, 1:78]  # Protein expression features
    y = data['class']       # Target classes
    
    # Encode the target classes
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, le.classes_

def train_random_forest(X, y, class_names):
    """Train and evaluate Random Forest classifier"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train the model
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # Evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('out/random_forest_confusion_matrix.png')
    plt.close()
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Protein': pd.read_csv('Data_Cortex_Nuclear.csv').columns[1:78],
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Important Proteins:")
    print(feature_importance.head(10))
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Protein', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Proteins - Random Forest')
    plt.tight_layout()
    plt.savefig('out/random_forest_feature_importance.png')
    plt.close()
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5)
    print(f"\nCross-Validation Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")
    
    return rf

if __name__ == "__main__":
    print("Running Random Forest Classifier...")
    X, y, class_names = load_and_preprocess_data('Data_Cortex_Nuclear.csv')
    rf_model = train_random_forest(X, y, class_names)