# import cv2;
# katti = cv2.VideoCapture('video1.mp4');
# success,image = katti.read()
# count = 0
# while success:
# 	cv2.imwrite("evidence/video1/photo%d.jpg"%count,image)
# 	print('photo %d', count)
# 	count +=1


# # import cv2
# # import numpy as np
# # import os

# # # Playing video from file:
# # cap = cv2.VideoCapture('video1.mp4')

# # try:
# #     if not os.path.exists('data'):
# #         os.makedirs('data')
# # except OSError:
# #     print ('Error: Creating directory of data')

# # currentFrame = 0
# # while(True):
# #     # Capture frame-by-frame
# #     ret, frame = cap.read()

# #     # Saves image of the current frame in jpg file
# #     name = './data/frame' + str(currentFrame) + '.jpg'
# #     print ('Creating...' + name)
# #     cv2.imwrite(name, frame)

# #     # To stop duplicate images
# #     currentFrame += 1

# # # When everything done, release the capture
# # cap.release()
# # cv2.destroyAllWindows()


import cv2
vidcap = cv2.VideoCapture('video28.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("Temp/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1