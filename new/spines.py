# Python standard library
import datetime
import io
import os
import pickle
import sys
import time
import requests
import json

# Scientific computing
import cv2

# Google cloud vision
from google.cloud import vision
from google.cloud.vision import types

# Shelfy
sys.path.append('..')
import book_functions
import lines as image_processing
import book_getter

# Set environment variable 
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:/Users/paulb/git/Tessa/data/Tessa-d6d474e9af90.json'

def get(url):
    response = requests.get(url)
    return json.loads(response.content)


def run(img_path):

    # Instantiates a google vision API client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open(img_path, 'rb') as img_file:
        content = img_file.read()
    img_bin = types.Image(content=content)

    # Query the image on google cloud vision API
    response = client.document_text_detection(image=img_bin)
    texts = response.text_annotations[1:]

    # Create word objects from the google word objects
    words = [book_functions.Word.from_google_text(text) for text in texts[1:]]

    # Get lines
    raw_img = cv2.imread(img_path)

    lines = image_processing.get_book_lines(raw_img)

    # Group the words into spines (using lines)
    spines = book_functions.get_spines_from_words_lines(words, lines)

    # Run the scraping pipeline for each spine
    books = []

    for spine in spines:
        # Get query
        query = spine.sentence
        books.append(query)
        
    return books

