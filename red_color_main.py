# import cv2
# import numpy as np

# ## Read
# img = cv2.imread("frame0.jpg")
# img = cv2.resize(img, (540, 960))

# ## convert to hsv
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ## mask of green (36,0,0) ~ (70, 255,255)
# mask1 = cv2.inRange
# #mask1 = cv2.inRange(hsv, (170, 70, 50), (180, 255,255))
# mask1 = cv2.inRange(hsv, (10, 100, 100), (10, 255,255))
# ## mask o yellow (15,0,0) ~ (36, 255, 255)
# #mask2 = cv2.inRange(hsv, (110, 50, 70), (180, 255, 255))
# mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
# ## final mask and masked
# mask = cv2.bitwise_or(mask1, mask2)
# target = cv2.bitwise_and(img,img, mask=mask)
# #bgr_image = cv2.cvtColor(mask1, cv2.COLOR_HSV2BGR)
# cv2.imwrite("target.png", target)
# cv2.imshow('target',target)
# cv2.imshow('mask',mask)
# cv2.imshow('mask1',mask1)
# cv2.imshow('mask2',mask2)
# #cv2.imshow('bgr_image',bgr_image)
# cv2.waitKey(0)

import cv2
import numpy as np

## Read
img = cv2.imread("frame302.jpg")
#img = cv2.resize(img, (540, 960))

## convert to hsv    
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## mask of green (36,0,0) ~ (70, 255,255)
#mask1 = cv2.inRange
#mask1 = cv2.inRange(hsv, (170, 70, 50), (180, 255,255))
#mask1 = cv2.inRange(hsv, (-1, 70, 50), (5, 255,255))
mask1 = cv2.inRange(hsv, (-1, 100, 100), (1, 255,255))
## mask o yellow (15,0,0) ~ (36, 255, 255)
#mask2 = cv2.inRange(hsv, (110, 50, 70), (180, 255, 255))
mask2 = cv2.inRange(hsv, (130, 100, 100), (190, 255, 255))
## final mask and masked
mask = cv2.bitwise_or(mask1, mask2)
target = cv2.bitwise_and(img,img, mask=mask)
#bgr_image = cv2.cvtColor(mask1, cv2.COLOR_HSV2BGR)
cv2.imwrite("image.jpg",img)
cv2.imwrite("target.png", target)
cv2.imshow('target',target)
cv2.imshow('mask',mask)
cv2.imshow('mask1',mask1)
cv2.imshow('mask2',mask2)
#cv2.imshow('bgr_image',bgr_image)
cv2.waitKey(0)
