import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob

filename = input("Enter Folder Name in which images are stored in .JPG/.png format : ")
print("Psress any key to move to next image in the folder.")

for image in glob.glob(filename+'/*.*'):
	try : 
		#reading image from the folder 
		img1 = cv.imread(image)

		#resizing image 
		img = cv.resize(img1,(0,0),fx=0.25,fy=0.25, interpolation = cv.INTER_CUBIC)
		
		#converting image to grayscale
		gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
		
		#Separation based on the variation of intensity between the object pixels and the background pixels
		ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

		cv.imshow('gray' , gray)

		#noise removal
		kernel = np.ones((3,3),np.uint8)
		opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations = 2)

		#sure background area
		sure_bg = cv.dilate(opening,kernel,iterations = 3)
		cv.imshow('sure_background',sure_bg)

		#finding sure foreground area
		dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
		ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
		cv.imshow('sure_foreground',sure_fg)

		#finding unknown region
		sure_fg = np.uint8(sure_fg)
		unknown = cv.subtract(sure_bg,sure_fg)
		cv.imshow('not_sure',unknown)

		#Marker labelling
		ret,markers =cv.connectedComponents(sure_fg)
		
		#Add one to all labels so that sure background is not 0, but 1
		markers = markers+1

		#Now, marking the region of unknown with zero
		markers[unknown==255]=0

		#applying watershed algorithm to mark the boundary region
		markers = cv.watershed(img,markers)
		img[markers == -1] = [255,0,0]

		cv.imshow('thresh',thresh)

		width, height = thresh.shape

		#finding the region of interest
		count=0
		for i in range(width):
			if 0 in thresh[i]:
				count+=1
				if count>=10:
					start = i
					break
		count=0
		for i in range(width-1,0,-1):
			if 0 in thresh[i]:
				count+=1
				if count>=10:
					end = i
					break
		ROI = img[start:end, 0:height]

		cv.imshow('ROI',ROI)
		if 'png' in str(image):
			cv.imwrite(str(image)+'_ROI.png',ROI) #saving output image
		if 'JPG' in str(image):
			cv.imwrite(str(image)+'_ROI.JPG',ROI) #saving output image
		
		cv.waitKey(0) #Waits till any key is pressed
		cv.destroyAllWindows() #close all opened window 

	except Exception as e :
		print(e)

