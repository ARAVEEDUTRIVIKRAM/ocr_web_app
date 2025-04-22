from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import pytesseract
import os

app = Flask(__name__)

# Configure Tesseract OCR
os.environ["TESSDATA_PREFIX"] = "./tessdata"  # Make sure `tessdata` is in the same directory

# Function to preprocess the image
def preprocess_image(image_path, mode="auto"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    if mode == "simple":
        _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
    elif mode == "adaptive":
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    else:
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# Function to extract text
def extract_text_from_image(image_path, lang="eng", mode="auto"):
    preprocessed_image = preprocess_image(image_path, mode)
    text = pytesseract.image_to_string(preprocessed_image, lang=lang)
    return text

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle OCR processing
@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return "No file uploaded", 400

    image_file = request.files["image"]
    language = request.form.get("language", "eng")
    mode = request.form.get("preprocessing_mode", "auto")

    # Save the uploaded image
    image_path = os.path.join("uploads", image_file.filename)
    image_file.save(image_path)

    # Perform OCR
    try:
        text = extract_text_from_image(image_path, lang=language, mode=mode)
        return render_template("result.html", text=text)
    except Exception as e:
        return f"Error: {e}", 500

# Run the Flask app
if __name__ == "__main__":
    # Ensure the `uploads` directory exists
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
