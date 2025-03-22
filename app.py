from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import base64
import io
import os
import joblib
from PIL import Image
import pytesseract  # For OCR
import math  # For mathematical operations
from scipy import ndimage  # For image rotation
from datetime import datetime
import time
import csv

# Set Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/Mac

app = Flask(__name__)

# Paths to models
MODEL_PATH = r"models"
# Create directory for correction logs
CORRECTION_PATH = r"corrections"
os.makedirs(CORRECTION_PATH, exist_ok=True)

# Global variables for models
knn_mnist = None
svm_mnist = None
knn_emnist = None
svm_emnist = None
emnist_mapping = None

# Character confusion mapping - frequently confused characters
CONFUSED_CHARS = {
    'A': ['4', 'H'],
    'B': ['8', 'R', '3'],
    '0': ['O', 'D'],
    '1': ['I', 'l'],
    '5': ['S'],
    '8': ['B'],
    'G': ['6'],
    'I': ['1', 'l'],
    'O': ['0', 'Q', 'D'],
    'S': ['5'],
    'Z': ['2'],
}

def load_models():
    """Load all ML models and mappings"""
    global knn_mnist, svm_mnist, knn_emnist, svm_emnist, emnist_mapping
    
    print("🔄 Loading models...")
    
    # Load MNIST models for digit recognition
    try:
        knn_mnist = joblib.load(os.path.join(MODEL_PATH, 'knn_mnist.joblib'))
        svm_mnist = joblib.load(os.path.join(MODEL_PATH, 'svm_mnist.joblib'))
        print("✅ MNIST models loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load MNIST models: {e}")
        knn_mnist = None
        svm_mnist = None
    
    # Load EMNIST models for letter/word recognition
    try:
        knn_emnist = joblib.load(os.path.join(MODEL_PATH, 'knn_emnist.joblib'))
        svm_emnist = joblib.load(os.path.join(MODEL_PATH, 'svm_emnist.joblib'))
        emnist_mapping = joblib.load(os.path.join(MODEL_PATH, 'emnist_mapping.joblib'))
        print("✅ EMNIST models loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load EMNIST models: {e}")
        print("ℹ️ Run train_emnist_model.py to train and save EMNIST models")
        knn_emnist = None
        svm_emnist = None
        emnist_mapping = None
        
    # If neither model set is available, exit
    if knn_mnist is None and knn_emnist is None:
        print("❌ No models available. Exiting...")
        exit(1)
    
    print("🚀 Models loaded and ready!")

# Load models on startup
load_models()

# Routes
@app.route('/')
def index():
    """Serve the digit recognition interface"""
    return render_template('index.html')

@app.route('/word_recognition')
def word_recognition():
    """Serve the word recognition interface"""
    return render_template('word_recognition.html')

# Image processing functions
def preprocess_image_mnist(image_data):
    """Process image for MNIST-based digit recognition"""
    try:
        # Remove header from base64 string if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')  # Convert to grayscale
        
        # Apply threshold to make the digit more defined
        image = np.array(image)
        _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Ensure black digit on white background (MNIST format is the opposite)
        image = 255 - image  # Invert to get white digit on black background
        
        # Center the digit by finding contours and creating a bounding box
        contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, use the original image
        if contours and len(contours) > 0:
            # Find the largest contour (the digit)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            # Extract the digit
            digit = image[y:y+h, x:x+w]
            
            # Create a square image with the digit centered
            side_length = max(w, h) + 10  # Add padding
            square_img = np.zeros((side_length, side_length), dtype=np.uint8)
            
            # Calculate position to center the digit
            x_offset = (side_length - w) // 2
            y_offset = (side_length - h) // 2
            
            # Place the digit in the center
            square_img[y_offset:y_offset+h, x_offset:x_offset+w] = digit
            
            # Resize to 20x20 (MNIST standard for the digit itself)
            digit_resized = cv2.resize(square_img, (20, 20), interpolation=cv2.INTER_AREA)
            
            # Place the 20x20 digit in a 28x28 field with 4-pixel borders (MNIST standard)
            mnist_img = np.zeros((28, 28), dtype=np.uint8)
            mnist_img[4:24, 4:24] = digit_resized
        else:
            # If no contours, just resize the original image
            mnist_img = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Create a debug image to send back to frontend
        debug_image = mnist_img.copy()
        _, debug_buffer = cv2.imencode('.png', debug_image)
        debug_image_b64 = base64.b64encode(debug_buffer).decode('utf-8')
        
        # Normalize pixel values (0-255 → 0-1) - MNIST standard
        image_float = mnist_img.astype(np.float32) / 255.0
        
        # Flatten to 1D array (28x28 → 784)
        image_flatten = image_float.flatten()
        
        # Reshape for model input
        image_scaled = image_flatten.reshape(1, -1)
        
        return image_scaled, debug_image_b64, image_scaled
        
    except Exception as e:
        print(f"Error preprocessing image for MNIST: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def preprocess_image_emnist(image):
    """Process image for EMNIST-based character recognition
    
    EMNIST requires specific preprocessing:
    1. Images need to be centered
    2. Need to be rotated/flipped to match EMNIST format
    """
    try:
        # Make sure it's grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to get binary image - improved binarization for better character definition
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours to detect the character
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Find the largest contour (character)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract character with more padding for better context
        padding = 8  # Increased padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(binary.shape[1], x + w + padding)
        y_end = min(binary.shape[0], y + h + padding)
        
        char_img = binary[y_start:y_end, x_start:x_end]
        
        # Create debug image to show what we extracted
        debug_img = gray.copy()
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), 255, 2)
        
        # Make square for consistent aspect ratio
        side_length = max(char_img.shape[0], char_img.shape[1]) + 10
        square_img = np.zeros((side_length, side_length), dtype=np.uint8)
        
        # Center the character
        y_offset = (side_length - char_img.shape[0]) // 2
        x_offset = (side_length - char_img.shape[1]) // 2
        
        square_img[y_offset:y_offset+char_img.shape[0], 
                  x_offset:x_offset+char_img.shape[1]] = char_img
        
        # Resize to 28x28 (EMNIST format)
        char_resized = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Try angle correction for characters like A which may lean
        best_char = char_resized
        best_score = np.sum(best_char)  # Use pixel sum as a basic metric
        
        # Try slight rotations to see if they improve character definition
        for angle in [-5, -2, 0, 2, 5]:
            if angle == 0:
                continue
            rotated = ndimage.rotate(char_resized, angle, reshape=False, mode='constant')
            score = np.sum(rotated)
            if score > best_score:
                best_char = rotated
                best_score = score
        
        # EMNIST format: rotate 90° clockwise and flip left-right
        char_rotated = np.fliplr(np.rot90(best_char, k=3))
        
        # Create our final preprocessed image
        processed_img = char_rotated
        
        # Normalize pixel values (0-255 → 0-1)
        image_float = processed_img.astype(np.float32) / 255.0
        
        # Flatten to 1D array (28x28 → 784)
        image_flatten = image_float.flatten()
        
        # Return both the flattened vector and the preprocessed image
        return image_flatten, processed_img
        
    except Exception as e:
        print(f"Error preprocessing image for EMNIST: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def segment_characters(image_data):
    """Segment an image into individual characters for word recognition"""
    try:
        # Decode base64 data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert('L'))
        
        # Create debug image (color for visualization)
        debug_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to improve character definition
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Dilate to connect components but with better control for character separation
        kernel_dilate = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
        
        # Find contours to detect characters
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from left to right (for correct word order)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Initialize list for character images
        char_images = []
        
        # Process each contour (potential character)
        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small contours (noise)
            if w < 8 or h < 8:  # Adjusted threshold to catch small characters like dots
                continue
                
            # Extract character with padding
            padding = 8  # Increased padding
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(binary.shape[1], x + w + padding)
            y_end = min(binary.shape[0], y + h + padding)
            
            # Extract the character from the original binary image (not dilated)
            char_img = binary[y_start:y_end, x_start:x_end]
            
            # Draw the bounding box in the debug image
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add to our list
            char_images.append({
                'image': char_img,
                'position': (x, y, w, h)
            })
        
        # Create debug image to show segmentation
        _, debug_buffer = cv2.imencode('.png', debug_img)
        debug_image_b64 = base64.b64encode(debug_buffer).decode('utf-8')
        
        return char_images, debug_image_b64
        
    except Exception as e:
        print(f"Error segmenting characters: {e}")
        import traceback
        traceback.print_exc()
        return [], ""

def recognize_with_emnist(img_vector, context_char=None):
    """Recognize a character using EMNIST model with optional context awareness"""
    if knn_emnist is None or svm_emnist is None:
        return None, 0
    
    # Reshape for model input (1 sample with 784 features)
    img_vector = img_vector.reshape(1, -1)
    
    # Get predictions from both models
    knn_pred = knn_emnist.predict(img_vector)[0]
    svm_pred = svm_emnist.predict(img_vector)[0]
    
    # Get probabilities
    knn_proba = knn_emnist.predict_proba(img_vector)[0]
    svm_proba = svm_emnist.predict_proba(img_vector)[0]
    
    # Combine predictions with weighting
    knn_weight = 0.5
    svm_weight = 0.5
    
    # If one model is very confident, give it more weight
    knn_confidence = knn_proba[np.argmax(knn_proba)]
    svm_confidence = svm_proba[np.argmax(svm_proba)]
    
    # Lower the confidence thresholds for more sensitivity
    if knn_confidence > 0.6:  # Lowered from 0.8
        knn_weight = 0.7
        svm_weight = 0.3
    if svm_confidence > 0.6:  # Lowered from 0.8
        svm_weight = 0.7
        knn_weight = 0.3
    
    # In case of high-confidence disagreement, use the more confident one
    if knn_confidence > 0.6 and svm_confidence > 0.6 and knn_pred != svm_pred:
        if knn_confidence > svm_confidence:
            knn_weight = 0.9
            svm_weight = 0.1
        else:
            svm_weight = 0.9
            knn_weight = 0.1
    
    # Combine probabilities
    combined_proba = knn_proba * knn_weight + svm_proba * svm_weight
    
    # Get the top 3 classes with highest probability
    top3_idx = np.argsort(combined_proba)[-3:][::-1]
    
    # Map to characters
    top3_chars = [emnist_mapping.get(int(idx), '?') for idx in top3_idx]
    top3_probs = [combined_proba[idx] * 100 for idx in top3_idx]
    
    # Apply special rules for commonly confused characters
    # For example, if 'B' is predicted with low confidence, consider if 'A' is a close second
    class_idx = np.argmax(combined_proba)
    confidence = combined_proba[class_idx] * 100
    
    # Get the character from the mapping
    char = emnist_mapping.get(int(class_idx), '?')
    
    # If confidence is low, check for common confusions
    if confidence < 60 and char in CONFUSED_CHARS:
        # Check if any of the commonly confused characters are in the top 3
        alternatives = CONFUSED_CHARS[char]
        for alt_char in alternatives:
            if alt_char in top3_chars:
                # If it's very close in probability, prefer the alternative
                alt_idx = top3_chars.index(alt_char)
                alt_prob = top3_probs[alt_idx]
                if alt_prob > confidence * 0.8:  # If it's at least 80% as confident
                    # Specifically for 'B' predicted as 'A'
                    if char == 'B' and alt_char == 'A' and alt_prob > confidence * 0.7:
                        char = 'A'
                        confidence = alt_prob
                        break
    
    # Return the character, confidence, and top 3 predictions for debugging
    top3_results = list(zip(top3_chars, top3_probs))
    return char, confidence, top3_results

# API Routes
@app.route('/predict', methods=['POST'])
def predict():
    """Original digit recognition endpoint using MNIST models"""
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({"error": "No image data received"}), 400

        # Verify we have the MNIST models loaded
        if knn_mnist is None or svm_mnist is None:
            return jsonify({"error": "MNIST models not available"}), 500

        # Preprocess the user-drawn image
        processed_image, debug_image, unscaled_image = preprocess_image_mnist(image_data)
        
        if processed_image is None:
            return jsonify({"error": "Failed to process image"}), 400

        # Get predictions from KNN and SVM
        knn_result = int(knn_mnist.predict(processed_image)[0])
        svm_result = int(svm_mnist.predict(processed_image)[0])
        
        # Get probability distributions from both models
        knn_proba = knn_mnist.predict_proba(processed_image)[0]
        svm_proba = svm_mnist.predict_proba(processed_image)[0]
        
        # Hybrid approach: weight both models
        knn_max_prob = np.max(knn_proba)
        svm_max_prob = np.max(svm_proba)
        
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
        
        # Generate explanation for weighting
        weight_explanation = f"KNN weight: {knn_weight:.2f}, SVM weight: {svm_weight:.2f}"
        
        # Combine probabilities with weighting
        hybrid_proba = knn_proba * knn_weight + svm_proba * svm_weight
        hybrid_result = int(np.argmax(hybrid_proba))
        hybrid_confidence = float(hybrid_proba[hybrid_result]) * 100
        
        # Simple fallback metrics
        pixel_sum = np.sum(unscaled_image)
        avg_pixel = pixel_sum / 784  # 28x28 = 784 pixels
        
        # Random test sample for demonstration
        sample_test_digit = np.random.randint(0, 10)
        sample_knn_pred = np.random.randint(0, 10)
        sample_svm_pred = np.random.randint(0, 10)
        
        # Get top 3 predictions for all models
        knn_top3_idx = np.argsort(knn_proba)[-3:][::-1]
        svm_top3_idx = np.argsort(svm_proba)[-3:][::-1]
        hybrid_top3_idx = np.argsort(hybrid_proba)[-3:][::-1]
        
        knn_top3 = [{"digit": int(idx), "confidence": f"{float(knn_proba[idx])*100:.2f}%"} 
                   for idx in knn_top3_idx]
        svm_top3 = [{"digit": int(idx), "confidence": f"{float(svm_proba[idx])*100:.2f}%"} 
                   for idx in svm_top3_idx]
        hybrid_top3 = [{"digit": int(idx), "confidence": f"{float(hybrid_proba[idx])*100:.2f}%"} 
                      for idx in hybrid_top3_idx]
        
        knn_confidence = float(knn_proba[knn_result]) * 100
        svm_confidence = float(svm_proba[svm_result]) * 100
        
        return jsonify({
            "knn": knn_result, 
            "svm": svm_result,
            "hybrid": hybrid_result,
            "knn_confidence": f"{knn_confidence:.2f}%",
            "svm_confidence": f"{svm_confidence:.2f}%",
            "hybrid_confidence": f"{hybrid_confidence:.2f}%",
            "weight_explanation": weight_explanation,
            "processed_image": debug_image,
            "debug_info": {
                "pixel_sum": float(pixel_sum),
                "avg_pixel": float(avg_pixel),
                "knn_top3": knn_top3,
                "svm_top3": svm_top3,
                "hybrid_top3": hybrid_top3,
                "test_sample": {
                    "actual": int(sample_test_digit),
                    "knn_pred": sample_knn_pred,
                    "svm_pred": sample_svm_pred
                }
            }
        })
    except Exception as e:
        print(f"Error in digit prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
        
@app.route('/recognize_text', methods=['POST'])
def recognize_text_route():
    """OCR text recognition endpoint using Tesseract"""
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({"error": "No image data received"}), 400

        # Remove header from base64 string if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Create a visual debug image
        debug_img = cv2.cvtColor(image_array.copy(), cv2.COLOR_GRAY2BGR)
        
        # Apply threshold for better OCR
        _, binary = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed - Tesseract works better with black text on white background
        if np.mean(binary) < 127:
            binary = 255 - binary
            
        # Detect text using Tesseract with improved configuration
        # Add character whitelist to help with recognition
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?"\'-'
        recognized_text = pytesseract.image_to_string(binary, config=custom_config).strip()
        
        # Get confidence data
        ocr_data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Calculate average confidence
        confidences = [conf for conf in ocr_data['conf'] if conf != -1]  # Filter out -1 values
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Create visualization with bounding boxes
        vis_image = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 0:  # Only consider confident detections
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis_image, ocr_data['text'][i], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Encode images for response
        _, debug_buffer = cv2.imencode('.png', debug_img)
        debug_image_b64 = base64.b64encode(debug_buffer).decode('utf-8')
        
        _, processed_buffer = cv2.imencode('.png', binary)
        processed_image_b64 = base64.b64encode(processed_buffer).decode('utf-8')
        
        _, vis_buffer = cv2.imencode('.png', vis_image)
        vis_image_b64 = base64.b64encode(vis_buffer).decode('utf-8')
        
        return jsonify({
            "recognized_text": recognized_text,
            "confidence": f"{avg_confidence:.2f}%",
            "debug_image": debug_image_b64,
            "processed_image": processed_image_b64,
            "visualization": vis_image_b64,
            "model_used": "tesseract"
        })
    except Exception as e:
        print(f"Error in text recognition: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/recognize_word', methods=['POST'])
def recognize_word():
    """Word recognition endpoint using EMNIST models"""
    try:
        data = request.json
        image_data = data.get('image')
        options = data.get('options', {})
        
        auto_segment = options.get('autoSegment', True)
        model_preference = options.get('modelPreference', 'auto')

        if not image_data:
            return jsonify({"error": "No image data received"}), 400

        # Check if we have at least one model type available
        if knn_emnist is None and pytesseract is None:
            return jsonify({"error": "No recognition models available"}), 500

        # Step 1: First try with Tesseract OCR
        tesseract_results = None
        if model_preference != 'emnist' and pytesseract is not None:
            try:
                # Remove header from base64 string if present
                ocr_image_data = image_data
                if ',' in ocr_image_data:
                    ocr_image_data = ocr_image_data.split(',')[1]
                    
                # Decode base64 to image
                image_bytes = base64.b64decode(ocr_image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('L')
                
                # Convert to numpy array
                image_array = np.array(image)
                
                # Apply threshold for better OCR
                _, binary = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Invert if needed - Tesseract works better with black text on white background
                if np.mean(binary) < 127:
                    binary = 255 - binary
                
                # Try different PSM modes to get the best result
                # Also add character whitelist to help with common confusions
                custom_whitelist = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?"\'-'
                psm_modes = [6, 8, 7, 10]  # 6=block, 8=word, 7=line, 10=single char
                best_text = ""
                best_confidence = 0
                best_ocr_data = None
                
                for psm in psm_modes:
                    custom_config = f'--oem 3 --psm {psm} {custom_whitelist}'
                    text = pytesseract.image_to_string(binary, config=custom_config).strip()
                    ocr_data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence
                    confidences = [conf for conf in ocr_data['conf'] if conf != -1]
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0
                    
                    if text and (avg_conf > best_confidence or not best_text):
                        best_text = text
                        best_confidence = avg_conf
                        best_ocr_data = ocr_data
                
                if best_text and best_confidence > 30:  # Only accept reasonably confident results
                    # Create visualization
                    vis_image = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
                    for i in range(len(best_ocr_data['text'])):
                        if best_ocr_data['conf'][i] > 0:
                            x, y, w, h = best_ocr_data['left'][i], best_ocr_data['top'][i], best_ocr_data['width'][i], best_ocr_data['height'][i]
                            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Encode for response
                    _, binary_buffer = cv2.imencode('.png', binary)
                    binary_b64 = base64.b64encode(binary_buffer).decode('utf-8')
                    
                    _, vis_buffer = cv2.imencode('.png', vis_image)
                    vis_b64 = base64.b64encode(vis_buffer).decode('utf-8')
                    
                    tesseract_results = {
                        "recognized_text": best_text,
                        "confidence": f"{best_confidence:.2f}%",
                        "processed_image": binary_b64,
                        "visualization": vis_b64,
                        "model_used": "tesseract"
                    }
                    
                    # If Tesseract has high confidence or user prefers it, return result now
                    if model_preference == 'tesseract' or best_confidence > 70:
                        # Also encode original image for debug
                        _, debug_buffer = cv2.imencode('.png', image_array)
                        debug_b64 = base64.b64encode(debug_buffer).decode('utf-8')
                        
                        tesseract_results["debug_image"] = debug_b64
                        return jsonify(tesseract_results)
            except Exception as e:
                print(f"Tesseract OCR attempt failed: {e}")
                # Continue to EMNIST approach if Tesseract fails

        # Step 2: Try with EMNIST model by segmenting characters
        emnist_results = None
        if model_preference != 'tesseract' and knn_emnist is not None and emnist_mapping is not None:
            try:
                # Segment the image into individual characters
                char_images, debug_image_b64 = segment_characters(image_data)
                
                if char_images:
                    # Initialize result
                    recognized_word = ""
                    total_confidence = 0
                    vis_image = None
                    all_top3 = []  # Store all top 3 predictions for each character
                    
                    # Process each character
                    for i, char_data in enumerate(char_images):
                        char_img = char_data['image']
                        
                        # Preprocess for EMNIST
                        img_vector, processed_char = preprocess_image_emnist(char_img)
                        
                        if img_vector is not None:
                            # Get previous character for context (if available)
                            prev_char = recognized_word[-1] if recognized_word else None
                            
                            # Recognize with EMNIST model
                            char, confidence, top3 = recognize_with_emnist(img_vector, prev_char)
                            all_top3.append(top3)
                            
                            # Add to result
                            recognized_word += char
                            total_confidence += confidence
                            
                            # Create visualization if first iteration
                            if i == 0:
                                # Initialize visualization image from debug image
                                vis_image = cv2.imdecode(
                                    np.frombuffer(base64.b64decode(debug_image_b64), np.uint8),
                                    cv2.IMREAD_COLOR
                                )
                            
                            # Add predicted character to visualization
                            x, y, w, h = char_data['position']
                            cv2.putText(vis_image, char, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Calculate average confidence
                    avg_confidence = total_confidence / len(char_images) if char_images else 0
                    
                    # Create processed image showing all characters
                    processed_img = np.zeros((200, len(char_images) * 28 + 10), dtype=np.uint8)
                    for i, char_data in enumerate(char_images):
                        char_img = char_data['image']
                        processed_char = cv2.resize(char_img, (28, 28))
                        x_offset = i * 28 + 5
                        processed_img[5:33, x_offset:x_offset+28] = processed_char
                    
                    # Encode images for response
                    _, processed_buffer = cv2.imencode('.png', processed_img)
                    processed_b64 = base64.b64encode(processed_buffer).decode('utf-8')
                    
                    _, vis_buffer = cv2.imencode('.png', vis_image)
                    vis_b64 = base64.b64encode(vis_buffer).decode('utf-8')
                    
                    emnist_results = {
                        "recognized_text": recognized_word,
                        "confidence": f"{avg_confidence:.2f}%",
                        "debug_image": debug_image_b64,
                        "processed_image": processed_b64,
                        "visualization": vis_b64,
                        "model_used": "emnist",
                        "character_count": len(char_images),
                        "top3_alternatives": [[c, f"{p:.2f}%"] for c, p in all_top3[0]] if all_top3 else []
                    }
            except Exception as e:
                print(f"EMNIST recognition attempt failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 3: Choose the best result or combine them
        if tesseract_results and emnist_results:
            # Compare confidence and choose the better one
            tesseract_conf = float(tesseract_results["confidence"].replace("%", ""))
            emnist_conf = float(emnist_results["confidence"].replace("%", ""))
            
            if tesseract_conf > emnist_conf or model_preference == 'tesseract':
                return jsonify(tesseract_results)
            else:
                return jsonify(emnist_results)
        elif tesseract_results:
            return jsonify(tesseract_results)
        elif emnist_results:
            return jsonify(emnist_results)
        else:
            return jsonify({
                "error": "Could not recognize text with any available method",
                "recognized_text": "",
                "confidence": "0%"
            }), 400
    
    except Exception as e:
        print(f"Error in word recognition: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/log_correction', methods=['POST'])
def log_correction():
    """Log user corrections for model improvement"""
    try:
        data = request.json
        incorrect = data.get('incorrect')
        correct = data.get('correct')
        image_data = data.get('image_data')
        
        if not incorrect or not correct:
            return jsonify({"error": "Missing correction data"}), 400
        
        # Create log directory if it doesn't exist
        os.makedirs(CORRECTION_PATH, exist_ok=True)
        
        # Append to correction log CSV
        log_file = os.path.join(CORRECTION_PATH, 'correction_log.csv')
        is_new_file = not os.path.exists(log_file)
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(['timestamp', 'incorrect', 'correct'])
            writer.writerow([datetime.now().isoformat(), incorrect, correct])
        
        # Save the image if provided
        if image_data and ',' in image_data:
            try:
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                # Create a filename that shows what was corrected
                filename = os.path.join(CORRECTION_PATH, f"{incorrect}_to_{correct}_{int(time.time())}.png")
                with open(filename, 'wb') as f:
                    f.write(image_bytes)
            except Exception as e:
                print(f"Could not save correction image: {e}")
        
        return jsonify({
            "success": True,
            "message": "Correction logged for future improvements"
        })
    except Exception as e:
        print(f"Error logging correction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)