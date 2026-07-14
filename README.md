# 🌍 Multilingual OCR Web Application

A web-based Optical Character Recognition (OCR) application that extracts text from uploaded images using Tesseract OCR. Users can select different image preprocessing techniques to improve OCR accuracy across multiple Indian languages.

Built using Python, Flask, OpenCV, Tesseract OCR, HTML, CSS, JavaScript, and Docker.

---


# Features

✔ Image Upload

✔ Multilingual OCR

✔ User-selectable Image Preprocessing

✔ Simple Thresholding

✔ Adaptive Thresholding

✔ Otsu Thresholding

✔ OCR using Tesseract

✔ Responsive Flask UI

✔ Multiple UI Themes

✔ Real-time OCR Result Display

✔ Docker Support

---

# Supported Languages

| Language | Supported |
|-----------|-----------|
| English | ✅ |
| Hindi | ✅ |
| Telugu | ✅ |
| Tamil | ✅ |
| Kannada | ✅ |
| Malayalam | ✅ |

---

# Tech Stack

## Backend

- Python
- Flask
- OpenCV
- Tesseract OCR

## Frontend

- HTML5
- CSS3
- JavaScript

## Deployment

- Docker

## Tools

- Git
- GitHub
- Google Colab

---

# Project Architecture

```text
User
      │
      ▼
Upload Image
      │
      ▼
Select Language
      │
      ▼
Choose Preprocessing
(Simple / Adaptive / Otsu)
      │
      ▼
Flask Backend
      │
      ▼
OpenCV Image Processing
      │
      ▼
Tesseract OCR
      │
      ▼
Extract Text
      │
      ▼
Display Result

```

# Folder Structure 

```text
ocr_web_app
│
├── static
│   ├── CSS
│   ├── Background Images
│   └── Themes
│
├── templates
│   ├── index.html
│   └── result.html
│
├── tessdata
│   ├── eng.traineddata
│   ├── hin.traineddata
│   ├── tel.traineddata
│   ├── tam.traineddata
│   ├── kan.traineddata
│   └── mal.traineddata
│
├── uploads
│
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```


# OCR Processing Workflow

Upload Image

↓

Select OCR Language

↓

Choose Preprocessing Method

↓

OpenCV Image Processing

↓

Tesseract OCR Engine

↓

Extract Text

↓

Display OCR Result


# Image Preprocessing Techniques

The application provides multiple preprocessing techniques that users can choose before OCR execution.

| Method |	Purpose | 
|---------|------------------|
| Simple Threshold |	Basic binary conversion for clean images | 
| Adaptive Threshold |	Handles uneven lighting conditions |
| Otsu Threshold |	Automatically determines optimal threshold values |

These preprocessing methods improve OCR accuracy depending on image quality.


# Installation 

git clone https://github.com/ARAVEEDUTRIVIKRAM/ocr_web_app.git

cd ocr_web_app

pip install -r requirements.txt

python app.py


# Testing

Validated using
- Multiple Languages
- Different Image Qualities
- OCR Accuracy Comparison
- Multiple Preprocessing Methods
- Browser Testing


# Known Limitations

- OCR accuracy depends on image quality.
- Handwritten text recognition is limited.
- Large images increase processing time.
- Complex backgrounds may reduce recognition accuracy.


# Future Enhancements

- PDF OCR
- Batch OCR
- EasyOCR Integration
- Google Vision API
- AWS Textract
- Drag & Drop Upload
- Copy-to-Clipboard
- Export as PDF
- OCR History


# Author

Araveedu Trivikram
