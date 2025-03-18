
import cv2 as cv
import numpy as np

img = cv.imread("C:/Users/brayi/Downloads/manzanaA.jpg",1)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
ub = np.array([0, 40, 40])
ua = np.array([10, 255, 255])

ub1 = np.array([170, 40, 40])
ua1 = np.array([180, 255, 255])

mascara1 = cv.inRange(hsv, ub, ua)
mascara2 = cv.inRange(hsv, ub1, ua1)

mascara = mascara1 + mascara2

resultado = cv.bitwise_and(img, img , mask = mascara)
cv.imshow('manzana', img)
cv.imshow('mascara', mascara)
cv.imshow('hsv', hsv)
cv.imshow('resultado', resultado)

cv.waitKey(0)

cv.destroyAllWindows()