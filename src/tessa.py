'''
DEPRECIATED
'''
def func2() :
    import cv2
    import pytesseract
    
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    
    # Grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread('image-asset.jpeg', 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    
    # Perform text extraction
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
    print(data)
    
    cv2.imshow('thresh', thresh)
    cv2.imshow('opening', opening)
    cv2.imshow('invert', invert)
    cv2.waitKey()



'''
DEPRECIATED
'''
def func(): 
    import cv2
    import pytesseract
    # reading image using opencv
    image = cv2.imread('example2.png')
    #converting image into gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # converting it to binary image by Thresholding
    # this step is require if you have colored image because if you skip this part 
    # then tesseract won't able to detect text correctly and this will give incorrect result
    threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # display image
    '''cv2.imshow(‘threshold image’, threshold_img)
    # Maintain output window until user presses a key
    cv2.waitKey(0)
    # Destroying present windows on screen
    cv2.destroyAllWindows() '''
    
    #configuring parameters for tesseract
    custom_config = r'--oem 3 --psm 6'
    # now feeding image to tesseract
    details = pytesseract.image_to_data(threshold_img, output_type=pytesseract.Output.DICT, config=custom_config, lang='eng')
    #print(details.keys())
    
    total_boxes = len(details['text'])
    for sequence_number in range(total_boxes):
    	if int(details['conf'][sequence_number]) >30:
    		(x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
    		threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    '''# display image
    cv2.imshow('captured text', threshold_img)
    # Maintain output window until user presses a key
    cv2.waitKey(0)
    # Destroying present windows on screen
    cv2.destroyAllWindows() '''
    
    parse_text = []
    word_list = []
    last_word = ''
    for word in details['text']:
        if word!='':
            word_list.append(word)
            last_word = word
        if (last_word!='' and word == '') or (word==details['text'][-1]):
            parse_text.append(word_list)
            word_list = []
    import csv
    with open('result_text.txt',  'w', newline="") as file:
        csv.writer(file, delimiter=" ").writerows(parse_text)
        







# Import libraries
import csv
import cv2
import pytesseract
import string

# Recognise spines from video
def get_spines(url):
    print('a')

# Takes image and performs pre-processing
def pre_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    #cv2.imshow('threshold image', thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return thresh

# Feeds to tesseract to predict text, returns metadata dict
def parse_text(threshold_img):
    # Tesseract params
    tesseract_config = r'--oem 3 --psm 6'
    # Now feeding image to tesseract
    details = pytesseract.image_to_data(threshold_img, output_type=pytesseract.Output.DICT,
                                        config=tesseract_config, lang='eng')
    return details

# Draws boxes around text areas
def draw_boxes(image, details, threshold_point):
    total_boxes = len(details['text'])
    for sequence_number in range(total_boxes):
        if int(details['conf'][sequence_number]) > threshold_point:
            (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                            details['width'][sequence_number], details['height'][sequence_number])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    # Display image
    #cv2.imshow('captured text', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# Arrange text into appropriate format
def format_text(details):
    parse_text = []
    word_list = []
    last_word = ''
    for word in details['text']:
        word = word.translate(str.maketrans('','', string.punctuation)) # Remove punctuation
        if word != '':
            word_list.append(word)
            last_word = word
        if (last_word != '' and word == '') or (word == details['text'][-1]):
            parse_text.append(word_list)
            word_list = []

    return parse_text

# Remove noise in text
def clarify(text):
    new = []
    for t in range(len(text)):
        if text[t]: # Remove empty ones
            maxlen = 0
            mainword = ''
            for x in text[t]: # Only leave longest word
                if len(x) > maxlen:
                    maxlen = len(x)
                    mainword = x
            new.append(mainword)
    return new

# Write text to file
def write_text(formatted_text):
    with open('../data/output/result_text.txt', 'w', newline="") as file:
        csv.writer(file, delimiter=" ").writerows(formatted_text)


if __name__ == "__main__":
    #image = cv2.imread('../data/image-asset.jpeg') # Read image
    image = cv2.imread('../data/example3.jpg')
    thresholds_image = pre_processing(image) # Preprocess
    parsed_data = parse_text(thresholds_image) # Get text
    accuracy_threshold = 30 # Threshold to draw box
    draw_boxes(thresholds_image, parsed_data, accuracy_threshold) # Draw boxes
    arranged_text = format_text(parsed_data) # Format text
    clarified_text = clarify(arranged_text)
    print(clarified_text)
    #write_text(arranged_text) # Write to file
    
    # TODO: More accurate recognition - check out google papers
    # TODO: Does the API need a 100% accurate input or can it infer?
    
