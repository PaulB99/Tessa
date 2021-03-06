import cv2
import numpy as np
import glob as gb

img_width = 1000
img_height = 450


# --------------Hough Transfrom---------
# 图像预处理（缩放、高斯滤波）
def pre_process(name):
    img = cv2.imread(name, 0)
    img_resized = cv2.resize(img, (img_width, img_height))
    img_blur = cv2.GaussianBlur(img_resized, (3, 3), 0)
    return img_resized, img_blur


def draw(img, lines):
    new_img = img.copy()
    for rho, theta in lines[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(new_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return new_img


def draw_vertical(img, lines):
    new_img = img.copy()
    for rho, theta in lines[:]:
        x1 = rho
        x2 = rho
        y1 = 0
        y2 = img_width
        cv2.line(new_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return new_img


def line_reduce(lines):
    i = 0
    j = 0
    lines_final = []
    while i < len(lines) - 1:
        if j >= len(lines) - 1:
            break
        j = i + 1
        lines_final.append(lines[i])
        while j < len(lines) - 1:
            if lines[j][0] - lines[i][0] > 10:
                i = j
                break
            else:
                j = j + 1
    return lines_final


def line_sifting(lines_list):
    lines = []
    for rho, theta in lines_list[:]:
        if (theta < (np.pi / 6.0)) or (theta > (11 * np.pi / 6.0)) or ((theta > (5 *np.pi / 6.0)) and (theta < (7 * np.pi / 6.0))):  # 限制与y轴夹角小于30度的线
            lines.append([rho, theta])
    lines.sort()
    lines_final = line_reduce(lines)
    return lines_final


# ------------Region Grow---------------
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y


def get_seeds(lines):
    seeds = []
    i = 0
    j = 1
    while i < len(lines)-2:
        y = int(lines[i][0] + (lines[j][0] - lines[i][0])/2) # 图片索引的x、y与我们理解的x、y相反
        x = int(img_height/2)
        seeds.append(Point(x, y))
        i = i + 1
        j = j + 1
    return seeds


def get_gray_diff(img, current_point, adjacent_point):
    return abs(int(img[current_point.x][current_point.y]) - int(img[adjacent_point.x][adjacent_point.y]))


def get_connects():
    connects = [Point(-1, -1), Point(-1, 0), Point(-1, 1), Point(0, -1), Point(0, 1), Point(1, -1), Point(1, 0),
                Point(1, 1)]
    return connects


def region_grow(img, seeds, thresh):
    seed_mark = np.zeros(img.shape)
    seed_stack = []
    for seed in seeds:
        seed_stack.append(seed)
    mark = 1
    connects = get_connects()
    while len(seed_stack) > 0:
        current_point = seed_stack.pop(0)
        seed_mark[current_point.x][current_point.y] = mark
        for connect in connects:
            adjacent_x = int(current_point.x + connect.x)
            adjacent_y = int(current_point.y + connect.y)
            if adjacent_x < 0 or adjacent_y < 0 or adjacent_x >= img_height or adjacent_y >= img_width:
                continue
            gray_diff = get_gray_diff(img, current_point, Point(adjacent_x, adjacent_y))
            if gray_diff < thresh and seed_mark[adjacent_x][adjacent_y] == 0:
                seed_mark[adjacent_x][adjacent_y] = mark
                seed_stack.append(Point(adjacent_x, adjacent_y))
    return seed_mark


# --------------image segmentation---------------
def segmentation(img, lines):
    imgs = []
    i = 0
    j = 1
    while i < len(lines) - 2:
        x1 = int(lines[i][0])
        x2 = int(lines[j][0])
        book_img = img[0:img_height, x1:x2]
        imgs.append(book_img)
        i = i + 1
        j = j + 1

    return imgs


def seg_horizontal(img):
    thresh = img.shape[1] - 10
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines_pre = cv2.HoughLines(edges, 1, np.pi / 180, thresh)  # 最后一个参数可调节，会影响直线检测的效果
    lines = lines_pre[:, 0, :]
    lines_horizontal = []
    for rho, theta in lines[:]:
        if ((theta < (12 * np.pi / 18.0)) and (theta > (4 * np.pi / 18.0))) or ((theta > (22 * np.pi / 18.0)) and (theta < (32 * np.pi / 18.0))):
            lines_horizontal.append([rho, theta])
    lines_horizontal.sort()
    lines_horizontal = line_reduce(lines_horizontal)
    if len(lines_horizontal) == 0:
        return img
    y1 = int(lines_horizontal[0][0])
    y2 = int(lines_horizontal[len(lines_horizontal)-1][0])
    book_img = img[y1:y2, 0:img_width]
    return book_img


def main():
    img_path = gb.glob("../data/example5.jpg")
    for path in img_path:
        img_gray, img = pre_process(path)
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 160)  # 最后一个参数可调节，会影响直线检测的效果
        lines1 = lines[:, 0, :]
        houghlines = line_sifting(lines1)  # 存储并筛选检测出的垂直线
        img_show = draw_vertical(img_gray, houghlines)
        img_segmentation = segmentation(img_gray, houghlines)
        i = 0
        for img_s in img_segmentation:
            if img_s.shape[0] == 0:
                print(i)
            # img_s = seg_horizontal(img_s)
            str1 = path[6:]
            str1 = str1[:-4]
            string = '../data/results/' + str(i) + '.jpg'
            print("Write " + string)
            cv2.imwrite(string, img_s)  # 保持切割后的图像
            i +=1
        #cv2.imshow("result", img_show)
        #cv2.waitKey()
main()
