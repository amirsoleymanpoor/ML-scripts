# -*- coding: utf-8 -*-
"""Untitled35.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EmYAHNgmVLB8jpVapG0ucyGuzmbOLB4M
"""

!apt-get update
!apt-get install -y tesseract-ocr
!pip install pytesseract

import cv2
import pytesseract


# Load the image
image = cv2.imread('/content/Untitled.jpg')


# Extract text from the image using Tesseract OCR
text = pytesseract.image_to_string(image)


# Print the extracted text
print(text)