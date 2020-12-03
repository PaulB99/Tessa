# Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.stats


# Line class
class Line(object):
    
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

# Show the image (for debugging)
def plot_img(img, show = True):

    fig = plt.figure(figsize = (16,12))
    plt.imshow(img, cmap = 'gray', interpolation = 'none')
    plt.xticks([])
    plt.yticks([])
    if show:
        plt.show()

# Apply gaussian blur with kernel length sigma
def gaussian_blur(img, sigma):
    proc_img = scipy.ndimage.filters.gaussian_filter(img, sigma = (sigma, sigma))
    return proc_img

# Reduce size given times
def downsample(img, num_downsamples):
    proc_img = np.copy(img)
    for i in range(num_downsamples):
        proc_img = scipy.ndimage.interpolation.zoom(proc_img,.5)

    return proc_img

# Calculate sobel x^2
def sobel_x_squared(img):
    
    proc_img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = -1)**2.
    return proc_img

# Sobel y gradient
def sobel_y_squared(img):
    proc_img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = -1)**2.
    return proc_img

# Laplace transformation
def laplace_squared(img):
    sobel_x_img = sobel_x_squared(img)
    sobel_y_img = sobel_y_squared(img)
    proc_img = (sobel_x_img**2. + sobel_y_img**2.)**.5

    return proc_img

# Standardise the image intensity
def standardise(img):
    proc_img = (img - np.min(img))/(np.max(img)-np.min(img))
    return proc_img

# Binarise by setting any value > 0 to 1
def dynamic_binarise(img, cutoff):
   

    for i in range(20):
        cutoff = i*.01
        bright_pixel_ratio = len(np.where(img > cutoff)[0])/(img.shape[0]*img.shape[1])


        if bright_pixel_ratio <= 0.4:
            break

    img[img > cutoff] = 1
    img[img <= cutoff] = 0

    return img


# Get rid of horizontal lines
def vertical_erode(img, structure_length, iterations):

    structure = np.array([[0,1,0],[0,1,0],[0,1,0]])*structure_length
    proc_img = scipy.ndimage.morphology.binary_erosion(img, structure, iterations)

    return proc_img

# Connect close lines
def vertical_dilate(img, structure_length, iterations):
    structure = np.array([[0,1,0],[0,1,0],[0,1,0]])*structure_length
    proc_img = scipy.ndimage.morphology.binary_dilation(img, structure, iterations)

    return proc_img

# Find connected components and assign values to component
def connected_components(img):
    proc_img, levels = scipy.ndimage.label(img, structure = np.ones((3,3)))
    levels = list(range(1, levels + 1))

    return proc_img, levels


# Remove lines that are too short
def remove_short_clusters_vertical(img, levels, threshold_fraction):
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

# Restore image to overall size
def upsample(img, upsample_factor):
    proc_img = img.repeat(upsample_factor, axis = 0).repeat(upsample_factor, axis = 1)

    return proc_img


# Get lines in a binary image and return Line objects
def get_lines_from_img(img, levels):

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

    return lines

# Get edges of spines in image
def get_book_lines(img, angles = [0], spaces = ['h']):

    # Convert to HSV
    gs_img = np.mean(img, axis = 2)
    final_img = np.zeros((gs_img.shape[0], gs_img.shape[1]))
    lines = []
    for angle in angles:

        # Rotate
        proc_img = scipy.ndimage.rotate(gs_img, angle = angle, reshape = False)

        # Blur
        sigma = 3
        proc_img = gaussian_blur(proc_img, sigma = sigma)

        # Sobel x
        proc_img = sobel_x_squared(proc_img)

        # Down sample
        num_downsamples = 2
        proc_img = downsample(proc_img, num_downsamples)

        # Standardise
        proc_img = standardise(proc_img)

        # Binarize
        cutoff = np.max(proc_img)/12.
        proc_img = dynamic_binarise(proc_img, cutoff)

        # Vertical erode
        structure_length = 200
        iterations = 8
        proc_img = vertical_erode(proc_img, structure_length, iterations)

        # Vertical dilate
        structure_length = 500
        iterations = 10
        proc_img = vertical_dilate(proc_img, structure_length, iterations)

        # Connected components
        proc_img, levels = connected_components(proc_img)

        # Remove short clusters
        threshold_fraction = 0.10
        proc_img = remove_short_clusters_vertical(proc_img, levels, threshold_fraction)

        # Up sample
        upsample_factor = 2**num_downsamples
        proc_img = upsample(proc_img, upsample_factor)

        # Un-rotate image
        proc_img = scipy.ndimage.rotate(proc_img, angle = -1*angle, reshape = False)
        proc_img.resize((img.shape[0], img.shape[1]))
        final_img = final_img + proc_img

    # Convert the final image to binary
    final_img[final_img > 0] = 1

    # Connect components label
    final_img, levels = connected_components(final_img)

    # Get the lines from the label
    lines = get_lines_from_img(final_img, levels)

    # Plot the result
    new_img = np.copy(img)

    plot_img(new_img, show = False)
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

    return lines

