import cv2
import numpy
from scipy.spatial import distance

CONST_BLURRING = (89, 89)
CONST_AMOUNT = 12
CONST_PATH = "./images/"

def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

def show(img, type=cv2.WINDOW_KEEPRATIO):
    cv2.namedWindow("Image", type)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def check(box):
    mbox = [mid(box[0], box[1]), mid(box[2], box[3]), mid(box[0], box[3]), mid(box[1], box[2])]
    w = distance.euclidean(mbox[0], mbox[1])
    h = distance.euclidean(mbox[2], mbox[3])
    if (w * 10 < h):
        return True
    if (h * 10 < w):
        return True
    return False


dataset = []
modified = []

for i in range(CONST_AMOUNT):
    dataset.append(cv2.imread(CONST_PATH + "img (" + str(i + 1) +  ").jpg"))
    modified.append(cv2.imread(CONST_PATH + "img (" + str(i + 1) +  ").jpg", cv2.IMREAD_GRAYSCALE))
    modified[i] = cv2.GaussianBlur(modified[i], CONST_BLURRING, 0)
    (_, modified[i]) = cv2.threshold(modified[i], 127, 255, cv2.THRESH_BINARY)

for i in range(CONST_AMOUNT):
    good = []
    cnts, _ = cv2.findContours(modified[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnts) > 0):
        for cnt in cnts:               
            box = numpy.array(cv2.boxPoints(cv2.minAreaRect(cnt)), dtype="int")
            if (check(box)):
                good.append([cnt])
    for cnt in good:
        cv2.drawContours(dataset[i], cnt, -1, (0, 0, 255), 9, cv2.LINE_AA)
    print("Image:", (i + 1), "\nPencils found:", len(good))
    show(dataset[i])
    cv2.imwrite(CONST_PATH + "img (" + str(i + 1) + ")_found.jpg", dataset[i])
