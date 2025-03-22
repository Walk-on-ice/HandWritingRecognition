"""
EMNIST Model Training Script (Windows Version)

This script trains KNN and SVM models on the EMNIST dataset with 
improved preprocessing to better recognize characters like 'A' vs 'B'.

Usage:
    python train_emnist_models.py [--include-corrections]

Options:
    --include-corrections  Include user corrections in training data
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import cv2
import pickle
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import argparse
import csv
from datetime import datetime
from PIL import Image
import base64
import io
import time

# Set paths (using Windows path style)
MODEL_PATH = "D:\\AI Project\\models"
CORRECTION_PATH = "D:\\AI Project\\corrections"
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(CORRECTION_PATH, exist_ok=True)

# Define character mapping (EMNIST classes to characters)
def create_emnist_mapping():
    """Create mapping from EMNIST class indices to characters"""
    mapping = {}
    # EMNIST Balanced dataset has 47 classes:
    # 0-9 (digits)
    # 10-35 (uppercase letters)
    # 36-46 (lowercase letters that differ in shape from their uppercase counterparts)
    
    # Add digits (0-9)
    for i in range(10):
        mapping[i] = str(i)
    
    # Add uppercase letters (A-Z)
    for i in range(26):
        mapping[i + 10] = chr(ord('A') + i)
    
    # Add lowercase letters that have different shapes (a, b, d, etc.)
    lowercase_diff = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
    for i, char in enumerate(lowercase_diff):
        mapping[i + 36] = char
    
    return mapping

def preprocess_image(image):
    """Apply preprocessing steps to an image"""
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Create a square image with padding
    side_length = max(binary.shape[0], binary.shape[1]) + 10
    square_img = np.zeros((side_length, side_length), dtype=np.uint8)
    
    # Center the character
    y_offset = (side_length - binary.shape[0]) // 2
    x_offset = (side_length - binary.shape[1]) // 2
    
    square_img[y_offset:y_offset+binary.shape[0], 
              x_offset:x_offset+binary.shape[1]] = binary
    
    # Resize to 28x28 (EMNIST format)
    resized = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # EMNIST format: rotate 90° clockwise and flip left-right
    rotated = np.fliplr(np.rot90(resized, k=3))
    
    # Return the preprocessed image
    return rotated

def load_user_corrections():
    """Load correction data from logs and image files"""
    correction_data = []
    
    # Check if correction log exists
    log_file = os.path.join(CORRECTION_PATH, 'correction_log.csv')
    if os.path.exists(log_file):
        # Load corrections from CSV
        corrections_df = pd.read_csv(log_file)
        print(f"Loaded {len(corrections_df)} corrections from log")
        
        # Search for correction images
        image_files = glob.glob(os.path.join(CORRECTION_PATH, "*.png"))
        print(f"Found {len(image_files)} correction images")
        
        # Process each image
        for img_file in image_files:
            try:
                # Parse filename to get correction info
                filename = os.path.basename(img_file)
                if '_to_' in filename:
                    incorrect, rest = filename.split('_to_', 1)
                    correct = rest[0] # Just take the first character
                    
                    # Load and preprocess the image
                    img = cv2.imread(img_file)
                    if img is not None:
                        processed_img = preprocess_image(img)
                        
                        # Flatten image to feature vector
                        features = processed_img.flatten().astype(np.float32) / 255.0
                        
                        # Get label from mapping
                        emnist_mapping = create_emnist_mapping()
                        label = None
                        
                        # Find the class index for the correct character
                        for class_idx, char in emnist_mapping.items():
                            if char == correct:
                                label = class_idx
                                break
                        
                        if label is not None:
                            correction_data.append((features, label))
                            print(f"Added correction: {incorrect} -> {correct} (class {label})")
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    return correction_data

def train_and_save_models(include_corrections=False):
    """Train KNN and SVM models on EMNIST dataset and save"""
    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("ERROR: torchvision is required. Install it with:")
        print("pip install torch torchvision")
        return
    
    print("Loading EMNIST dataset... This might take a while.")
    
    try:
        # Load EMNIST dataset (using torchvision)
        # Note: This requires torchvision package
        emnist_train = datasets.EMNIST(
            root='./data',
            split='balanced',  # balanced split has 47 classes
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        
        emnist_test = datasets.EMNIST(
            root='./data',
            split='balanced',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
    except Exception as e:
        print(f"ERROR loading EMNIST dataset: {e}")
        print("Make sure you have an internet connection and try again.")
        return
    
    # Extract training data
    train_images = []
    train_labels = []
    
    # Use a subset for faster training (adjust as needed)
    train_subset_size = 40000  # Reduced from full dataset for speed
    indices = np.random.choice(len(emnist_train), train_subset_size, replace=False)
    
    print("Extracting training samples...")
    for idx in indices:
        img, label = emnist_train[idx]
        # Convert PyTorch tensor to numpy array and flatten
        img = img.numpy().reshape(-1)
        train_images.append(img)
        train_labels.append(label)
    
    X_train = np.array(train_images)
    y_train = np.array(train_labels)
    
    # Extract test data
    test_subset_size = 5000  # Reduced for speed
    indices = np.random.choice(len(emnist_test), test_subset_size, replace=False)
    
    print("Extracting test samples...")
    test_images = []
    test_labels = []
    for idx in indices:
        img, label = emnist_test[idx]
        img = img.numpy().reshape(-1)
        test_images.append(img)
        test_labels.append(label)
    
    X_test = np.array(test_images)
    y_test = np.array(test_labels)
    
    print(f"Loaded {len(X_train)} training and {len(X_test)} test samples")
    
    # Add user corrections if requested
    if include_corrections:
        correction_data = load_user_corrections()
        if correction_data:
            print(f"Adding {len(correction_data)} user corrections to training data")
            
            # Duplicate corrections to give them more weight
            DUPLICATION_FACTOR = 5  # Multiply corrections to increase their impact
            for _ in range(DUPLICATION_FACTOR):
                for features, label in correction_data:
                    X_train = np.vstack([X_train, features.reshape(1, -1)])
                    y_train = np.append(y_train, label)
    
    # Train KNN model (with improved parameters)
    print("Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    knn.fit(X_train, y_train)
    knn_accuracy = knn.score(X_test, y_test)
    print(f"KNN Accuracy: {knn_accuracy:.4f}")
    
    # Train SVM model
    print("Training SVM model...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(X_train, y_train)
    svm_accuracy = svm.score(X_test, y_test)
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    # Evaluate specific character confusions
    print("\nEvaluating specific character confusions...")
    
    # Get predictions
    knn_preds = knn.predict(X_test)
    svm_preds = svm.predict(X_test)
    
    # Create confusion matrices
    knn_cm = confusion_matrix(y_test, knn_preds)
    svm_cm = confusion_matrix(y_test, svm_preds)
    
    # Map class indices to characters for analysis
    emnist_mapping = create_emnist_mapping()
    
    # Define character pairs to analyze for confusion
    char_pairs = [('A', 'B'), ('B', 'A'), ('0', 'O'), ('I', '1')]
    
    print("Confusion Rates:")
    for char1, char2 in char_pairs:
        # Find class indices for these characters
        idx1 = None
        idx2 = None
        for class_idx, char in emnist_mapping.items():
            if char == char1:
                idx1 = class_idx
            elif char == char2:
                idx2 = class_idx
        
        if idx1 is not None and idx2 is not None:
            # Calculate confusion rates
            knn_confusion = knn_cm[idx1, idx2] / np.sum(knn_cm[idx1]) if np.sum(knn_cm[idx1]) > 0 else 0
            svm_confusion = svm_cm[idx1, idx2] / np.sum(svm_cm[idx1]) if np.sum(svm_cm[idx1]) > 0 else 0
            
            print(f"{char1} mistaken as {char2}: KNN={knn_confusion:.4f}, SVM={svm_confusion:.4f}")
    
    # Save models
    print("\nSaving models...")
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Save models and mapping
    joblib.dump(knn, os.path.join(MODEL_PATH, 'knn_emnist.joblib'))
    joblib.dump(svm, os.path.join(MODEL_PATH, 'svm_emnist.joblib'))
    joblib.dump(emnist_mapping, os.path.join(MODEL_PATH, 'emnist_mapping.joblib'))
    
    # Save model info including training date and accuracy
    with open(os.path.join(MODEL_PATH, 'emnist_model_info.txt'), 'w') as f:
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"KNN Accuracy: {knn_accuracy:.4f}\n")
        f.write(f"SVM Accuracy: {svm_accuracy:.4f}\n")
        f.write(f"Included User Corrections: {include_corrections}\n")
        if include_corrections:
            f.write(f"Number of Corrections Added: {len(correction_data)}\n")
    
    print("Models saved successfully!")
    
    # Plot and save confusion matrix for 'B' and 'A'
    b_idx = None
    a_idx = None
    for class_idx, char in emnist_mapping.items():
        if char == 'B':
            b_idx = class_idx
        elif char == 'A':
            a_idx = class_idx
    
    if b_idx is not None and a_idx is not None:
        plt.figure(figsize=(10, 5))
        
        # KNN confusion subplot
        plt.subplot(1, 2, 1)
        plt.imshow([[knn_cm[b_idx, a_idx], knn_cm[b_idx, b_idx]], 
                   [knn_cm[a_idx, a_idx], knn_cm[a_idx, b_idx]]], 
                  cmap='Blues')
        plt.xticks([0, 1], ['A', 'B'])
        plt.yticks([0, 1], ['B', 'A'])
        plt.colorbar()
        plt.title('KNN Confusion: A vs B')
        
        # SVM confusion subplot
        plt.subplot(1, 2, 2)
        plt.imshow([[svm_cm[b_idx, a_idx], svm_cm[b_idx, b_idx]], 
                   [svm_cm[a_idx, a_idx], svm_cm[a_idx, b_idx]]], 
                  cmap='Blues')
        plt.xticks([0, 1], ['A', 'B'])
        plt.yticks([0, 1], ['B', 'A'])
        plt.colorbar()
        plt.title('SVM Confusion: A vs B')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_PATH, 'ab_confusion.png'))
        print("Saved confusion matrix visualization for A vs B")

# Entry point of the script
def main():
    parser = argparse.ArgumentParser(description='Train EMNIST models with improved parameters')
    parser.add_argument('--include-corrections', action='store_true', 
                       help='Include user corrections in training data')
    
    # Check if we have any arguments - Windows command prompt might not pass them correctly
    if len(sys.argv) == 1:
        print("No arguments provided. Running without corrections.")
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    print("==== EMNIST Model Training ====")
    print(f"Using corrections from users: {args.include_corrections}")
    
    # Check required packages
    try:
        import torch
        import torchvision
    except ImportError:
        print("ERROR: torch and torchvision are required for this script.")
        print("Install them with: pip install torch torchvision")
        return
    
    train_and_save_models(include_corrections=args.include_corrections)
    print("\nTraining complete! Press Enter to exit...")
    input()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nAn error occurred. Press Enter to exit...")
        input()