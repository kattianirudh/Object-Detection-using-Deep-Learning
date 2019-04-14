import cv2
from matplotlib import pyplot as plt
img = cv2.imread('target.png')

search = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
im2, contours, hierarchy = cv2.findContours(search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bboxes = [cv2.boundingRect(c) for c in contours]

fig, axes = plt.subplots(1, sum(rect[2]*rect[3] > 250 for rect in bboxes))
fig.set_size_inches([12,3])
fig.tight_layout()
figi = 0
for i in range(len(contours)):
    rect = cv2.boundingRect(contours[i])
    area = rect[2] * rect[3]
    #if area < 250:
    if area < 4000:
        continue

    obj = img[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1, :]
    obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
    imgg = cv2.imread("image.jpg")
    crop_img = imgg[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1]
    cv2.imwrite("%.3d.jpg" % figi, crop_img)
    #axes[figi].imshow(obj)
    cv2.imshow("cropped", crop_img)
    #cv2.waitKey(0)
    figi += 1

fig.show()
cv2.waitKey(0)