import cv2
import dlib
import time
import threading
import math

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('cars.mp4')

WIDTH = 1280
HEIGHT = 720

'''
Helper function calculating euclidien distance between first and second location, and then calculating speed
'''

def calculate_speed(location1, location2):
	
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	
	ppm = 8.8
	d_meters = d_pixels / ppm
	
	fps = 12
	speed = d_meters * fps * 3.6
	return speed
	
'''
1. Identifying the car in the frame 
2. Keeping their carID in dictionary
3. Displaying speed above the bounding boxes
4. Deleting the carids from the dictionaries who are outside the observation frame
'''

def ObjectsTracking():
	rectangleColor = (0, 255, 0)
	frameCounter = 0
	currentCarID = 0
	fps = 0
	
	'''
	Creating dictionaries
	'''
	
	carTracker = {}
	carNumbers = {}
	carLocation1 = {}
	carLocation2 = {}

	'''
	Creating a list of 1000 elements
	'''
	
	speed = [None] * 1000  
	
	'''
	Write output to video file
	'''
	
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH,HEIGHT))


	while True:
		start_time = time.time()
		rc, image = video.read()
		if type(image) == type(None):
			break
		
		image = cv2.resize(image, (WIDTH, HEIGHT))
		resultImage = image.copy()
		
		frameCounter = frameCounter + 1
		
		carIDtoDelete = []

		'''
		Appending Unique Car IDs to cars
		'''
	
		for carID in carTracker.keys():
			trackingQuality = carTracker[carID].update(image)
			
			if trackingQuality < 7: # 7 is our confidence threshold
				carIDtoDelete.append(carID)
				
		for carID in carIDtoDelete:
			print ('Removing carID ' + str(carID) + ' from list of trackers.')
			print ('Removing carID ' + str(carID) + ' previous location.')
			print ('Removing carID ' + str(carID) + ' current location.')
			carTracker.pop(carID, None)
			carLocation1.pop(carID, None)
			carLocation2.pop(carID, None)
		
		
		'''
		1. Detecting car every 10 frames
		2. On every 10th frame, if we detect an object we extract the extreme points of the cars
		'''
	
		if not (frameCounter % 10):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
			
			for (_x, _y, _w, _h) in cars:
				x = int(_x)
				y = int(_y)
				w = int(_w)
				h = int(_h)
				
				'''
				Calculating the centre point
				'''
	
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				matchCarID = None
				
				'''
				Now loop over all the trackers and check if the centerpoint of the car is within the box of a tracker
				'''
				
				for carID in carTracker.keys():
					trackedPosition = carTracker[carID].get_position()
					
					t_x = int(trackedPosition.left())
					t_y = int(trackedPosition.top())
					t_w = int(trackedPosition.width())
					t_h = int(trackedPosition.height())
					
					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h
					
					'''
					Check if the centerpoint of the car is within the rectangle of a tracker region. Also, the centerpoint of the tracker 
					region must be within the region detected as a car. If both of these conditions hold we have a match
					'''
					
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = carID
						
				'''
				If CarID doesn't match, we create a new tracker
				'''
	
				if matchCarID is None:
					print ('Creating new tracker ' + str(currentCarID))
					
					'''
					Here we establish our dlib object tracker and provide the bounding box coordinates
					'''
					
					tracker = dlib.correlation_tracker()
					tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
					
					carTracker[currentCarID] = tracker
					carLocation1[currentCarID] = [x, y, w, h]

					currentCarID = currentCarID + 1
		
		'''
		We get all the extreme points of the cars in its final position required to calculate estimated speed
		'''
		
		for carID in carTracker.keys():
			trackedPosition = carTracker[carID].get_position()
					
			t_x = int(trackedPosition.left())
			t_y = int(trackedPosition.top())
			t_w = int(trackedPosition.width())
			t_h = int(trackedPosition.height())
			
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
			
			# speed estimation
			carLocation2[carID] = [t_x, t_y, t_w, t_h]
		
		end_time = time.time() # This is the end time used to calculate speed
		
		if not (end_time == start_time):
			fps = 1.0/(end_time - start_time)
		
		'''
		Comparing the location of same car if condition matches speed is calculated
		'''
		
		for i in carLocation1.keys():	
			if frameCounter % 1 == 0:
				[x1, y1, w1, h1] = carLocation1[i]
				[x2, y2, w2, h2] = carLocation2[i]
		
				
				carLocation1[i] = [x2, y2, w2, h2]
		
				
				if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
					if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
						speed[i] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2])

					
					if speed[i] != None and y1 >= 180:
						cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
					
		'''
		Display results
		'''
		
		cv2.imshow('result', resultImage)
		
		if cv2.waitKey(33) == 27:
			break
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	ObjectsTracking()
