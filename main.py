#  Author : Anirudh Katti
#  Project : Undergraduate Capstone Project
#  Topic : Traffic sign Board Detection using Deep
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import train_test_split
from keras import backend as K
#K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
# --------------------------------------------------------------------------------------------------------

PATH = os.getcwd()
# Define data path
data_path = PATH + '/Augmentation'
data_path_new = PATH + '/video27'
data_dir_list = os.listdir(data_path)
num_classes = 3
img_data_list=[]
img_rows=128
img_cols=128
num_channel=1
num_epoch=20

# for file in listing:
#     im = Image.open(path1 + '\\' +file)
#     img = im.resize((img_rows, img_cols))
#     gray = image.convert('L')
for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(128,128))
		img_data_list.append(input_img_resize)
		print("done")

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)
if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)
USE_SKLEARN_PREPROCESSING=False

if USE_SKLEARN_PREPROCESSING:
	# using sklearn for preprocessing
	from sklearn import preprocessing
	
	def image_to_feature_vector(image, size=(128, 128)):
		# resize the image to a fixed size, then flatten the image into
		# a list of raw pixel intensities
		return cv2.resize(image, size).flatten()
	
	img_data_list=[]
	for dataset in data_dir_list:
		img_list=os.listdir(data_path+'/'+ dataset)
		print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
		for img in img_list:
			input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
			input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
			input_img_flatten=image_to_feature_vector(input_img,(128,128))
			img_data_list.append(input_img_flatten)
	
	img_data = np.array(img_data_list)
	img_data = img_data.astype('float32')
	print (img_data.shape)
	img_data_scaled = preprocessing.scale(img_data)
	print (img_data_scaled.shape)
	
	print (np.mean(img_data_scaled))
	print (np.std(img_data_scaled))
	
	print (img_data_scaled.mean(axis=0))
	print (img_data_scaled.std(axis=0))
	
	if K.image_dim_ordering()=='th':
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
		print (img_data_scaled.shape)
		
	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
		print (img_data_scaled.shape)
	
	
	if K.image_dim_ordering()=='th':
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
		print (img_data_scaled.shape)
		
	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
		print (img_data_scaled.shape)
if USE_SKLEARN_PREPROCESSING:
	img_data=img_data_scaled
#%%
# Assigning Labels

# Define the number of classes
num_classes = 4

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:100]=0
labels[100:200]=1
labels[200:300]=2
labels[300:]=3
	  
names = ['stop','hump','HTV','HTV']
	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#%%
# Defining the model
input_shape=img_data[0].shape
					
model = Sequential()

model.add(Convolution2D(60, 5,5,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(60, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(30, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(30, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
	


model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))
# Evaluating the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
model.save('my_model.hdf5')
test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])
# Testing a new image


# test_image = cv2.imread('C:/Users/katti/Desktop/Project/Final-Year-Final/Augmentation/2/006.jpg')
# test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# test_image=cv2.resize(test_image,(128,128))
# test_image = np.array(test_image)
# test_image = test_image.astype('float32')
# test_image /= 255
# print (test_image.shape)
   
# if num_channel==1:
# 	if K.image_dim_ordering()=='th':
# 		test_image= np.expand_dims(test_image, axis=0)
# 		test_image= np.expand_dims(test_image, axis=0)
# 		print (test_image.shape)
# 	else:
# 		test_image= np.expand_dims(test_image, axis=3) 
# 		test_image= np.expand_dims(test_image, axis=0)
# 		print (test_image.shape)
		
# else:
# 	if K.image_dim_ordering()=='th':
# 		test_image=np.rollaxis(test_image,2,0)
# 		test_image= np.expand_dims(test_image, axis=0)
# 		print (test_image.shape)
# 	else:
# 		test_image= np.expand_dims(test_image, axis=0)
# 		print (test_image.shape)
		
# # Predicting the test image
# print((model.predict(test_image)))
# val = model.predict_classes(test_image)
# print(model.predict_classes(test_image))
# print(names[val[0]])

# --------------------------------------------------------------------------------------------------------

# path1 = "C:\Users\katti\Desktop\Current Projects\Final Year Project\codes\evidence\video3"
# path2 = "C:\Users\katti\Desktop\Current Projects\Final Year Project\codes\evidence\grey"
def function_first(img) :
	#img = cv2.imread("Dataset/20190403_070757.jpg")
	#img = cv2.resize(img, (540, 960))

	## convert to hsv    
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	## mask of green (36,0,0) ~ (70, 255,255)
	#mask1 = cv2.inRange
	#mask1 = cv2.inRange(hsv, (170, 70, 50), (180, 255,255))
	mask1 = cv2.inRange(hsv, (-1, 100, 100), (5, 255,255))
	#mask1 = cv2.inRange(hsv, (-1, 70, 50), (10, 255,255))
	## mask o yellow (15,0,0) ~ (36, 255, 255)
	#mask2 = cv2.inRange(hsv, (110, 50, 70), (180, 255, 255))
	mask2 = cv2.inRange(hsv, (130, 100, 100), (190, 255, 255))
	## final mask and masked
	mask = cv2.bitwise_or(mask1, mask2)
	target = cv2.bitwise_and(img,img, mask=mask)
	#bgr_image = cv2.cvtColor(mask1, cv2.COLOR_HSV2BGR)
	cv2.imwrite("image.jpg",img)
	cv2.imwrite("target.png", target)

# img = cv2.imread("Dataset/20190403_070757.jpg")
# #img = cv2.resize(img, (540, 960))

# ## convert to hsv    
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ## mask of green (36,0,0) ~ (70, 255,255)
# #mask1 = cv2.inRange
# #mask1 = cv2.inRange(hsv, (170, 70, 50), (180, 255,255))
# mask1 = cv2.inRange(hsv, (-1, 100, 100), (5, 255,255))
# #mask1 = cv2.inRange(hsv, (-1, 70, 50), (10, 255,255))
# ## mask o yellow (15,0,0) ~ (36, 255, 255)
# #mask2 = cv2.inRange(hsv, (110, 50, 70), (180, 255, 255))
# mask2 = cv2.inRange(hsv, (130, 100, 100), (190, 255, 255))
# ## final mask and masked
# mask = cv2.bitwise_or(mask1, mask2)
# target = cv2.bitwise_and(img,img, mask=mask)
# #bgr_image = cv2.cvtColor(mask1, cv2.COLOR_HSV2BGR)
# cv2.imwrite("image.jpg",img)
# cv2.imwrite("target.png", target)
#-----------------------------------------------------------------------------------------------------

def func1(img) :
	test_image = img
	test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
	test_image=cv2.resize(test_image,(128,128))
	test_image = np.array(test_image)
	test_image = test_image.astype('float32')
	test_image /= 255	
	print (test_image.shape)
	   
	if num_channel==1:
		if K.image_dim_ordering()=='th':
			test_image= np.expand_dims(test_image, axis=0)
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
		else:
			test_image= np.expand_dims(test_image, axis=3) 
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
			
	else:
		if K.image_dim_ordering()=='th':
			test_image=np.rollaxis(test_image,2,0)
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
		else:
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
			
	# Predicting the test image
	print((model.predict(test_image)))
	val = model.predict_classes(test_image)
	print(model.predict_classes(test_image))
	print(names[val[0]])
	return names[val[0]]



img_list=os.listdir(data_path_new+'/')
for i,img in enumerate(img_list):
	img = cv2.imread(data_path_new + '/'+ img )
	#function_first(images) 
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	## mask of green (36,0,0) ~ (70, 255,255)
	#mask1 = cv2.inRange
	#mask1 = cv2.inRange(hsv, (170, 70, 50), (180, 255,255))
	mask1 = cv2.inRange(hsv, (-1, 100, 100), (5, 255,255))
	#mask1 = cv2.inRange(hsv, (-1, 70, 50), (10, 255,255))
	## mask o yellow (15,0,0) ~ (36, 255, 255)
	#mask2 = cv2.inRange(hsv, (110, 50, 70), (180, 255, 255))
	mask2 = cv2.inRange(hsv, (130, 100, 100), (190, 255, 255))
	## final mask and masked
	mask = cv2.bitwise_or(mask1, mask2)
	target = cv2.bitwise_and(img,img, mask=mask)
	#bgr_image = cv2.cvtColor(mask1, cv2.COLOR_HSV2BGR)
	cv2.imwrite("image.jpg",img)
	cv2.imwrite("target.png", target)
	#img = cv2.imread('target.png')
	img_red = target
	font  = cv2.FONT_HERSHEY_COMPLEX
	search = cv2.dilate(img_red, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
	search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
	im2, contours, hierarchy = cv2.findContours(search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	bboxes = [cv2.boundingRect(c) for c in contours]

	fig, axes = plt.subplots(1, sum(rect[2]*rect[3] > 250 for rect in bboxes))
	fig.set_size_inches([12,3])
	fig.tight_layout()
	figi = 0
	for j in range(len(contours)):
	    rect = cv2.boundingRect(contours[j])
	    area = rect[2] * rect[3]
	    #if area < 250:
	    if area < 4000:
	        continue

	    obj = img[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1, :]
	    obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
	    #imgg = cv2.imread("image.jpg")
	    crop_img = img[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1]
	    label = func1(crop_img)
	    cv2.imwrite("cuts/%.3d-%.2d.jpg" % (i , j),crop_img)
	    #axes[figi].imshow(obj)
	    #cv2.rectangle(imgg, (rect[1],rect[1]+rect[3]+1), (rect[0], rect[0]+rect[2]+1), (255,255,0), 2)
	    #test_image = cv2.imread('C:/Users/katti/Desktop/Project/Final-Year-Final/Augmentation/2/006.jpg')
	    
	    cv2.rectangle(img, (rect[0],rect[1]),(rect[0]+rect[2]+3,rect[1]+rect[3]+3),(0,0,255),5)
	    cv2.putText(img, label,(rect[0],rect[1]),font,1,(200,255,255),2,cv2.LINE_AA)
	    cv2.imwrite("final/image%.3d.jpg" % i,img)
	    #cv2.imshow("cropped", crop_img)
	    #cv2.waitKey(0)
	    figi += 1

	#fig.show()
	#cv2.waitKey(0)

#-------------- IF THE CODE WORKED DON'T FORGET TO THANK KATTI---------------------------	