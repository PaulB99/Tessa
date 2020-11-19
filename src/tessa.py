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
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.stats
import image_processing

# A line object 
class Line(object):
    '''
    Simple class that holds the information related to a line;
    i.e., the slope, y-intercept, and center point along the line
    '''

    vertical_threshold = 30

    def __init__(self, m, b, center, min_x, max_x, min_y, max_y):
        '''
        m: slope
        b: y-intercept
        center: center point along the line (tuple)
        '''

        self.m = m
        self.b = b

        self.center = center

        self.min_x = min_x
        self.max_x = max_x

        self.min_y = min_y
        self.max_y = max_y


    def y(self, x):
        '''
        Returns the y-value of the line at position x.
        If the line is vertical (i.e., slope is close to infinity), the y-value
        will be returned as None
        '''

        # Line is vertical
        if self.m > self.vertical_threshold:
            return None

        else:
            return self.m*x + self.b


    def x(self, y):
        '''
        Returns the x-value of the line at posiion y.
        If the line is vertical (i.e., slope is close to infinity), will always
        return the center point of the line
        '''

        # Line is vertical
        if self.m > self.vertical_threshold:
            return self.center[0]

        # Line is not vertical
        else:
            return (y - self.b)/self.m


# Draw lines on image
def get_lines(img, angles = [0], spaces = ['h']):
    gs_img = np.mean(img, axis = 2)
    final_img = np.zeros((gs_img.shape[0], gs_img.shape[1])) # I'll put the lines onto this
    lines = [] # The lines in the image
    for angle in angles:
        
        # Rotate to given angle
        proc_img = scipy.ndimage.rotate(gs_img, angle = angle, reshape = False)
        
        
        # Guassian blur
        sigma = 3
        proc_img = scipy.ndimage.filters.gaussian_filter(proc_img, sigma = (sigma, sigma))


        # Sobel x
        proc_img = cv2.Sobel(proc_img, cv2.CV_64F, 1, 0, ksize = -1)**2.


        # Down sample
        num_downsamples = 2
        for i in range(num_downsamples):
            proc_img = scipy.ndimage.interpolation.zoom(proc_img,.5)
         
            
        # Standardise intensity
        proc_img = (proc_img - np.min(proc_img))/(np.max(proc_img)-np.min(proc_img))
        
        
        # Binarise (Any pixel with intensity > 0 is 1)
        cutoff = np.max(proc_img)/12.
        for i in range(20):
            cutoff = i*.01
            bright_pixel_ratio = len(np.where(proc_img > cutoff)[0])/(proc_img.shape[0]*proc_img.shape[1])
            if bright_pixel_ratio <= 0.4:
                break

        proc_img[proc_img > cutoff] = 1
        proc_img[proc_img <= cutoff] = 0
        
        
        # Vertical erode (Get rid of horizontal lines)
        structure_length = 200
        iterations = 8
        structure = np.array([[0,1,0],[0,1,0],[0,1,0]])*structure_length
        proc_img = scipy.ndimage.morphology.binary_erosion(proc_img, structure, iterations)
        
        
        # Vertical dilate (Combine lines that are close to each other)
        structure_length = 500
        iterations = 10
        structure = np.array([[0,1,0],[0,1,0],[0,1,0]])*structure_length
        proc_img = scipy.ndimage.morphology.binary_dilation(proc_img, structure, iterations)
        
        
        # Find connected components and give a unique value per component
        proc_img, levels = scipy.ndimage.label(proc_img, structure = np.ones((3,3)))
        levels = list(range(1, levels + 1))
        
        
        # Removes components (lines) that are shorter than the longest line * threshold
        threshold_fraction = 0.10
        proc_img = remove_short(proc_img, levels, threshold_fraction)
        
        # Upsample (Restore to original size)
        upsample_factor = 2**num_downsamples
        proc_img = proc_img.repeat(upsample_factor, axis = 0).repeat(upsample_factor, axis = 1)
        
        # Un-rotate image
        proc_img = scipy.ndimage.rotate(proc_img, angle = -1*angle, reshape = False)
        proc_img.resize((proc_img.shape[0], proc_img.shape[1]))
        final_img = final_img + proc_img
        
        # Conver the final image to binary
        final_img[final_img > 0] = 1
    
        # Connect components label
        final_img, levels = scipy.ndimage.label(final_img, structure = np.ones((3,3)))
        levels = list(range(1, levels + 1))
    
        # Get the lines from the label
        lines = get_lines_from_img(final_img, levels)
        
    return lines


# Removes short lines
def remove_short(img, levels, threshold_fraction):
    drop_values = []
    ptps = []

    # Calculate peak-to-peak height of line
    for level in levels:
        bright_pixels = np.where(img == level)
        ptp = np.ptp(bright_pixels[0])
        ptps.append(ptp)

    # Determine which lines to drop
    threshold = np.max(ptps)/2.
    for i in range(len(ptps)):
        if ptps[i] < threshold:
            drop_values.append(levels[i])

    # Drop the lines
    for drop_value in drop_values:
        img[img == drop_value] = 0

    return img
      
# Get the equations for all lines and return a list of line objects
def get_lines_from_img(img, levels, test=True):
    lines = []
    for level in levels:
        line = np.where(img == level)
        xs = line[1]
        ys = line[0]
        center = [np.mean(xs), np.mean(ys)]

        min_x = np.min(xs)
        max_x = np.max(xs)
        min_y = np.min(ys)
        max_y = np.max(ys)
        
        spread = (np.max(ys) - np.min(ys))/(np.max(xs) - np.min(xs))

        # Line is vertical
        if spread > 10:
            line = Line(1000, 0, center, min_x, max_x, min_y, max_y)

        # Line is not vertical
        else:
            m, b, r, p, std = scipy.stats.linregress(xs,ys)
            line = Line(m, b, center, min_x, max_x, min_y, max_y)


        lines.append(line)

    # Sort the lines by their centre x positions
    lines.sort(key = lambda line: line.center[0])
    
    # Show the lines drawn if I'm testing
    if test:
        new_img = np.copy(img)

        fig = plt.figure(figsize = (16,12))
        plt.imshow(img, cmap = 'gray', interpolation = 'none')
        plt.xticks([])
        plt.yticks([])
        
        for line in lines:
            y0 = line.min_y
            y1 = line.max_y

            x0 = line.x(y0)
            x1 = line.x(y1)

            plt.plot([x0, x1], [y0, y1], color = np.array([0,169,55])/255., lw = 6)

        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('proc_img.png', bbox_inches = 'tight', dpi = 300)

        plt.show()


    return lines
     
'''
DEPRECIATED!!!

# Rotate an image a given angle
def rotate_image(image, angle):
    image_centre = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_centre, angle, 1.0)
    output = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return output

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
  
        '''
if __name__ == "__main__":
    
    imgpath = '../data/example5.jpg'
    img = cv2.imread(imgpath)
    #lines = get_lines(img)
    lines = image_processing.get_book_lines(img)
    
    '''
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey,100,300,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,120)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        
    cv2.imwrite('houghlines3.jpg',img)
    '''
    '''
    image = cv2.imread('../data/example5.jpg')
    edges = cv2.Canny(image, 100, 200)
    plt.subplot(121),plt.imshow(image,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    '''
    '''
    #rot_image = rotate_image(img, 270)
    thresholds_image = pre_processing(img) # Preprocess
    parsed_data = parse_text(thresholds_image) # Get text
    accuracy_threshold = 30 # Threshold to draw box
    draw_boxes(thresholds_image, parsed_data, accuracy_threshold) # Draw boxes
    arranged_text = format_text(parsed_data) # Format text
    clarified_text = clarify(arranged_text)
    print(clarified_text)
    #write_text(arranged_text) # Write to file
    '''
    
    # TODO: More accurate recognition - check out google papers
    # TODO: Does the API need a 100% accurate input or can it infer?
    
