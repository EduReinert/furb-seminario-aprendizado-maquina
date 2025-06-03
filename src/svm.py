import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
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

def train_svm(X, y, class_names):
    """Train and evaluate SVM classifier"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train the model
    svm = SVC(
        kernel='rbf', 
        C=1.0, 
        gamma='scale', 
        probability=True, 
        random_state=42, 
        class_weight='balanced'
    )
    svm.fit(X_train, y_train)
    
    # Predictions
    y_pred = svm.predict(X_test)
    
    # Evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - SVM')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('out/svm_confusion_matrix.png')
    plt.close()
    
    # Cross-validation
    cv_scores = cross_val_score(svm, X, y, cv=5)
    print(f"\nCross-Validation Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")
    
    # PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis')
    plt.title('PCA Visualization of Protein Expression Data - SVM')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Class', labels=class_names)
    plt.tight_layout()
    plt.savefig('out/svm_pca_visualization.png')
    plt.close()
    
    return svm

if __name__ == "__main__":
    print("Running SVM Classifier...")
    X, y, class_names = load_and_preprocess_data('Data_Cortex_Nuclear.csv')
    svm_model = train_svm(X, y, class_names)