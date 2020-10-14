import cv2 as cv
import pytesseract
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

#example.jpg
#image-asset.jpeg
#example2.png

img = cv.imread(r'example2.png')
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
gray, img_bin = cv.threshold(gray,128,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
gray = cv.bitwise_not(img_bin)
kernel = np.ones((2, 1), np.uint8)
img = cv.erode(gray, kernel, iterations=1)
img = cv.dilate(img, kernel, iterations=1)
#cv.imshow('window', img)
out_below = pytesseract.image_to_string(img)
print("OUTPUT:", out_below)

