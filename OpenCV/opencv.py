
import cv2 as cv
import numpy as np
from PIL.ImImagePlugin import split

img = cv.imread("C:/Users/brayi/Downloads/logoprueba.png",1)
print(img.shape)
imgn = np.zeros(img.shape[:2], np.uint8)
b,g,r = cv.split(img)
#img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#img3 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
imgb = cv.merge([b, imgn, imgn])

cv.imshow('b',b)
cv.imshow('g',g)
cv.imshow('r',r)

cv.imshow('img',img)
#cv.imshow('img2',img2)
#cv.imshow('hsv',img3)
cv.imshow('imgb',imgb)

cv.waitKey()

cv.destroyAllWindows()