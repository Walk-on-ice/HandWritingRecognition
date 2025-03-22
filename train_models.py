import numpy as np
import os
import idx2numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# Define MNIST dataset path - adjust this to match your actual path
MNIST_PATH = r"D:\AI Project\data\mnist"
MODEL_PATH = r"D:\AI Project\models"  # New path for storing models

# Create models directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

def load_mnist(image_filename, label_filename):
    """Load MNIST images and labels from .idx files."""
    try:
        image_path = os.path.join(MNIST_PATH, image_filename)
        label_path = os.path.join(MNIST_PATH, label_filename)

        if not os.path.exists(image_path) or not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing file: {image_path} or {label_path}")

        images = idx2numpy.convert_from_file(image_path)  # Shape: (60000, 28, 28)
        labels = idx2numpy.convert_from_file(label_path)  # Shape: (60000,)

        images = images.reshape(images.shape[0], -1)  # Flatten to (60000, 784)
        return images, labels
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        exit(1)  # Stop execution if MNIST files fail to load

try:
    print("🔄 Loading MNIST dataset...")
    X_train_full, y_train_full = load_mnist('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
    X_test, y_test = load_mnist('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    print("✅ MNIST dataset loaded successfully!")

    # Convert to float and normalize
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Use 20,000 samples with balanced classes
    X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, 
                                            train_size=20000, 
                                            stratify=y_train_full, 
                                            random_state=42)

    print("🔄 Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)
    knn_accuracy = knn.score(X_test, y_test)
    print(f"✅ KNN trained successfully! Accuracy: {knn_accuracy:.4f}")

    print("🔄 Training SVM model...")
    svm = SVC(kernel='rbf', C=5, gamma='scale', probability=True)
    svm.fit(X_train, y_train)
    svm_accuracy = svm.score(X_test, y_test)
    print(f"✅ SVM trained successfully! Accuracy: {svm_accuracy:.4f}")
    
    # Test hybrid model on test data to calculate its expected accuracy
    print("🔄 Evaluating Hybrid model...")
    hybrid_correct = 0
    
    for i in range(len(X_test)):
        x = X_test[i:i+1]
        
        # Get confidence scores from both models
        knn_probs = knn.predict_proba(x)[0]
        svm_probs = svm.predict_proba(x)[0]
        
        # Hybrid approach: weight both models and combine their probabilities
        knn_max_prob = np.max(knn_probs)
        svm_max_prob = np.max(svm_probs)
        
        knn_weight = 0.5
        svm_weight = 0.5
        
        if knn_max_prob > 0.8:
            knn_weight = 0.7
            svm_weight = 0.3
        if svm_max_prob > 0.8:
            svm_weight = 0.7
            knn_weight = 0.3
            
        # In case both are very confident but disagree, trust the more confident one
        if knn_max_prob > 0.8 and svm_max_prob > 0.8:
            if knn_max_prob > svm_max_prob:
                knn_weight = 0.9
                svm_weight = 0.1
            else:
                svm_weight = 0.9
                knn_weight = 0.1
                
        # Combine probabilities with weighting
        hybrid_probs = knn_probs * knn_weight + svm_probs * svm_weight
        hybrid_pred = np.argmax(hybrid_probs)
        
        if hybrid_pred == y_test[i]:
            hybrid_correct += 1
    
    hybrid_accuracy = hybrid_correct / len(X_test)
    print(f"✅ Hybrid model evaluation complete! Accuracy: {hybrid_accuracy:.4f}")

    # Save models
    print("🔄 Saving models to disk...")
    joblib.dump(knn, os.path.join(MODEL_PATH, 'knn_mnist.joblib'))
    joblib.dump(svm, os.path.join(MODEL_PATH, 'svm_mnist.joblib'))
    
    # Save model accuracies
    with open(os.path.join(MODEL_PATH, 'model_info.txt'), 'w') as f:
        f.write(f"KNN Accuracy: {knn_accuracy:.4f}\n")
        f.write(f"SVM Accuracy: {svm_accuracy:.4f}\n")
        f.write(f"Hybrid Accuracy: {hybrid_accuracy:.4f}\n")
        f.write(f"Training data size: {len(X_train)} samples\n")

    print(f"✅ Models saved successfully to {MODEL_PATH}!")
    print("✅ Training complete! You can now run your application with faster startup.")

except Exception as e:
    print(f"❌ ERROR: {e}")