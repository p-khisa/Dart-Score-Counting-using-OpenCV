#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import os
import numpy as np



# In[2]:


def blue_dart_detection(image_color):
  image_ori = image_color

  image_color = cv2.cvtColor(image_color,cv2.COLOR_BGR2HSV)

  # Blue Thrsholds
  lower_bound = np.array([90, 50, 70])
  upper_bound = np.array([128, 255, 255])

  image = image_ori

  mask = cv2.inRange(image_color, lower_bound, upper_bound)

  kernel = np.ones((3, 3), np.uint8)

  #Use erosion and dilation combination to eliminate false positives. 
  #In this case the text Q0X could be identified as circles but it is not.
  
  closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  # cv2_imshow(mask)

  contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
          cv2.CHAIN_APPROX_SIMPLE)[0]
  contours.sort(key=lambda x:cv2.boundingRect(x)[0])

  array = []

  ii = 1
  # print(len(contours))
  if len(contours) == 0:
    print("No Blue color detected.")
    centers_circle = []
    return image , centers_circle

  centers_circle = []
  for c in contours:
      (x,y),r = cv2.minEnclosingCircle(c)
      center = (int(x-30),int(y))
      r = int(r)
      # print(r)
      if r >= 12:
          cv2.circle(image,center,r,(255,0,0),2)
          centers_circle.append(center)
          font_scale = 1.5
          font = cv2.FONT_HERSHEY_PLAIN
          text = "Blue"
          # get the width and height of the text box
          (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
          # set the text start position
          text_offset_x = center[0]
          text_offset_y = center[1]
          # make the coords of the box with a small padding of two pixels
          box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
          cv2.rectangle(image,box_coords[0], box_coords[1] , (0,0,0), cv2.FILLED)
          cv2.putText(image, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255, 255, 255), thickness=1)

          break
          #cv2.putText(image, 'Blue', center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,58,238), 2)

          array.append(center)
          break
#   cv2_imshow(image)
  return image, centers_circle


# In[3]:


def yellow_dart_detection(image_color):
  image_ori = image_color

  image_color = cv2.cvtColor(image_color,cv2.COLOR_BGR2HSV)

  # Yellow Thrsholds
  lower_bound = np.array([20, 50, 70])
  upper_bound = np.array([40, 255, 255])

  # Red Thresholds
  # lower_bound = np.array([159, 50, 70])
  # upper_bound = np.array([180, 255, 255])

  image = image_ori

  mask = cv2.inRange(image_color, lower_bound, upper_bound)

  kernel = np.ones((3, 3), np.uint8)

  #Use erosion and dilation combination to eliminate false positives. 
  #In this case the text Q0X could be identified as circles but it is not.
  mask = cv2.erode(mask, kernel, iterations=3 )
  # cv2_imshow(mask)
  mask = cv2.dilate(mask, kernel, iterations=6)
  # cv2_imshow(mask)
  closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

  contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
          cv2.CHAIN_APPROX_SIMPLE)[0]
  contours.sort(key=lambda x:cv2.boundingRect(x)[0])
  array = []

  ii = 1
  # print(len(contours))
  if len(contours) == 0:
    print("No Yellow color detected.")

  centers_circle = []
  for c in contours:
      (x,y),r = cv2.minEnclosingCircle(c)
      center = (int(x),int(y+25))
      
      r = int(r)
      #print("Y" , r)
      if r >= 17:
        # print(r)qqqqq
        print(center)
        cv2.circle(image,center,r,(0,255,255),2)
        centers_circle.append(center)
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_PLAIN
        text = "Yellow"
        # get the width and height of the text box
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        # set the text start position
        text_offset_x = center[0]
        text_offset_y = center[1]
        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(image,box_coords[0], box_coords[1] , (0,0,0), cv2.FILLED)
        cv2.putText(image, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255, 255, 255), thickness=1)

        #cv2.putText(image, 'Yellow', center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (42,138,222), 2)
        #print(center)
        array.append(center)

#   cv2_imshow(image)

  return image , centers_circle


# In[ ]:





# In[4]:


def dartBoard_detection(img_color , img):
 
  # Convert the image to gray-scale
  gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
  # Find the edges in the image using canny detector
  edges = cv2.Canny(gray, 100, 1000)
  # cv2_imshow(edges)
  cropped_img = []
  c_img = img_color.copy()

  x_center = 0
  y_center = 0
 
  # Apply hough transform on the image
  circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,500,param1=1000,param2=110,minRadius=90,maxRadius=350)

  # Draw detected circles
  if circles is not None:
      circles = np.uint16(np.around(circles))
      for i in circles[0, :]:

          # Draw outer circle
          cv2.circle(c_img, (i[0], i[1]), i[2], (255, 0, 255), 7)
          w = 10
          h = 5
          # Draw inner circle
          cv2.circle(c_img, (i[0], i[1]), 2, (0, 0, 255), 3)
          # print(i[0]  ,  i[1]  , i[2])
          img1 = img_color.copy()
          img = img.copy()
          gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
          ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

          # Create mask
          height,width = img.shape
          mask = np.zeros((height,width), np.uint8)


          cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

          x_center , y_center = i[0] , i[1]
          cc = cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)

          # Copy that image using that mask
          masked_data = cv2.bitwise_and(img1, img1, mask=mask)

          # Apply Threshold
          _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

          # Find Contour
          contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
          x,y,w,h = cv2.boundingRect(cc)

          # Crop masked_data
          cropped_img = masked_data[y:y+h,x:x+w]


          
          font_scale = 1.5
          font = cv2.FONT_HERSHEY_PLAIN
          text = "DartBoard"
          # get the width and height of the text box
          (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
          # set the text start position
          text_offset_x = i[0]
          text_offset_y = i[1]
          # make the coords of the box with a small padding of two pixels
          box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
          cv2.rectangle(c_img,box_coords[0], box_coords[1] , (0,0,0), cv2.FILLED)
          cv2.putText(c_img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255, 255, 255), thickness=1)
          

  return cropped_img , x_center , y_center

# In[5]:


def inner_circle_detection(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Find the edges in the image using canny detector
  edges = cv2.Canny(gray, 100, 1000)
  # cv2_imshow(edges)

  c_img = img.copy()
  #################
  circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,250,param1=500,param2=50,minRadius=90,maxRadius=200)
  
  # Draw detected circles
  if circles is not None:
      circles = np.uint16(np.around(circles))
      for i in circles[0, :]:

          # Draw outer circle
          cv2.circle(c_img, (i[0], i[1]), i[2], (255, 0, 255), 7)
          w = 10
          h = 5
          # Draw inner circle
          cv2.circle(c_img, (i[0], i[1]), 2, (0, 0, 255), 3)
          # print(i[0]  ,  i[1]  , i[2])

          font_scale = 1.5
          font = cv2.FONT_HERSHEY_PLAIN
          text = "DartBoard"
          # get the width and height of the text box
          (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
          # set the text start position
          text_offset_x = i[0]
          text_offset_y = i[1]
          # make the coords of the box with a small padding of two pixels
          box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
          cv2.rectangle(c_img,box_coords[0], box_coords[1] , (0,0,0), cv2.FILLED)
          cv2.putText(c_img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255, 255, 255), thickness=1)
          
  return circles


# In[ ]:





# In[ ]:





# In[6]:


# input_path = '/content/drive/MyDrive/Munim_Project/Dart_Detection/7.jpg'
# output_path = '/content/drive/MyDrive/Munim_Project/Dart_Detection/output'
# image_color= cv2.imread(input_path)

# image_name = input_path[52:]
# image_color = cv2.resize(image_color, (0,0), fx=0.5, fy=0.5) 
# img = image_color.copy()
# # img = cv2.resize(img, (612,408), interpolation = cv2.INTER_AREA)
# # cv2_imshow(img)
# # Convert the image to gray-scale


# In[7]:


INT_MAX = 10000
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
	
	if ((q[0] <= max(p[0], r[0])) &
		(q[0] >= min(p[0], r[0])) &
		(q[1] <= max(p[1], r[1])) &
		(q[1] >= min(p[1], r[1]))):
		return True
		
	return False

# To find orientation of ordered triplet (p, q, r). 
# The function returns following values 
# 0 --> p, q and r are colinear 
# 1 --> Clockwise 
# 2 --> Counterclockwise 
def orientation(p:tuple, q:tuple, r:tuple) -> int:
	
	val = (((q[1] - p[1]) *
			(r[0] - q[0])) -
		((q[0] - p[0]) *
			(r[1] - q[1])))
			
	if val == 0:
		return 0
	if val > 0:
		return 1 # Collinear
	else:
		return 2 # Clock or counterclock

def doIntersect(p1, q1, p2, q2):
	
	# Find the four orientations needed for 
	# general and special cases 
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1)
	o4 = orientation(p2, q2, q1)

	# General case
	if (o1 != o2) and (o3 != o4):
		return True
	
	# Special Cases 
	# p1, q1 and p2 are colinear and 
	# p2 lies on segment p1q1 
	if (o1 == 0) and (onSegment(p1, p2, q1)):
		return True

	# p1, q1 and p2 are colinear and 
	# q2 lies on segment p1q1 
	if (o2 == 0) and (onSegment(p1, q2, q1)):
		return True

	# p2, q2 and p1 are colinear and 
	# p1 lies on segment p2q2 
	if (o3 == 0) and (onSegment(p2, p1, q2)):
		return True

	# p2, q2 and q1 are colinear and 
	# q1 lies on segment p2q2 
	if (o4 == 0) and (onSegment(p2, q1, q2)):
		return True

	return False

# Returns true if the point p lies 
# inside the polygon[] with n vertices 
def is_inside_polygon(points:list, p:tuple) -> bool:
	
	n = len(points)
	
	# There must be at least 3 vertices
	# in polygon
	if n < 3:
		return False
		
	# Create a point for line segment
	# from p to infinite
	extreme = (INT_MAX, p[1])
	count = i = 0
	
	while True:
		next = (i + 1) % n
		
		# Check if the line segment from 'p' to 
		# 'extreme' intersects with the line 
		# segment from 'polygon[i]' to 'polygon[next]' 
		if (doIntersect(points[i],
						points[next], 
						p, extreme)):
							
			# If the point 'p' is colinear with line 
			# segment 'i-next', then check if it lies 
			# on segment. If it lies, return true, otherwise false 
			if orientation(points[i], p, 
						points[next]) == 0:
				return onSegment(points[i], p, 
								points[next])
								
			count += 1
			
		i = next
		
		if (i == 0):
			break
		
	# Return true if count is odd, false otherwise 
	return (count % 2 == 1)


# In[8]:


def isInside(circle_x, circle_y, rad, x, y): 
	
	# Compare radius of circle 
	# with distance of its center 
	# from given point 
	if ((x - circle_x) * (x - circle_x) +
		(y - circle_y) * (y - circle_y) <= rad * rad): 
		return True; 
	else: 
		return False; 


# In[9]:


def calculate_score(cntrs , circles):
  label_pts = ['18' , '4' , '13' , '6' , '10' , '15' , '2' , '17' , '3' , '19' 
               , '7' , '16' , '8' , '11' , '14' , '9' , '12' , '5' , '20' , '1']

  pts_score = [ [(950,245), (835, 475), (1025,295)]
  , [(1025,295), (842, 475), (1085,370)]
  , [(1085,370), (848, 480), (1105,450)]
  , [(1105,450), (848, 490), (1110,530)]
  , [(1111,530), (848, 496), (1085,620)]
  , [(1085,621), (845, 500), (1030,700)]
  , [(1030,701),(840,505), (955, 750)]
  , [(955,750),(833,510), (870, 775)]
  , [(870,776),(825,512), (780, 775)]
  , [(779,775),(815,510), (695, 745)]
  , [(693,745),(810,503), (618, 688)]
  , [(618,688),(806,500), (565, 612)]
  , [(565,612),(805,495), (545, 530)]
  , [(545,530),(803,487), (545, 445)]
  , [(545,445),(805,480), (570, 360)]
  , [(570,360),(807,475), (630, 288)]
  , [(630,288),(813,470), (703, 243)]
  , [(703,243),(817,468), (780, 220)]
  , [(780,220),(823,470), (865, 220)]
  , [(865,220),(830,470), (948, 243)] ]
  total_score = 0
  # for i in range(len(pts_score)):
  #     cnt = cv2.polylines(image_color , np.array([pts_score[i]]) , True , (255,0,0 , 3))
  for center in cntrs:
    x = center[0]
    y = center[1]
    point_score = 0
    for i in range(len(pts_score)):
      polygon1 = pts_score[i]

      p = (x, y)

      circle_x = circles[0][0][0]
      circle_y = circles[0][0][1]
      rad = circles[0][0][2]
      if (is_inside_polygon(points = polygon1, p = p)):
        # print('Yes')
        # print("The Number is :" , int(label_pts[i]))
        if (isInside(circle_x, circle_y, rad, x, y)):

          point_score = point_score + (int(label_pts[i]) * 2)
          # print("2x Point score : " , point_score)
          total_score = total_score + point_score
          break
        else:
          point_score = point_score + int(label_pts[i])
          # print("Point score : " , point_score)
          total_score = total_score + point_score
          break
      # print("Total " ,total_score )

  return total_score



def bulls_eye_score_calculator(bulls_eye_coordinates , cntrs_blue , total_score):
    if cntrs_blue == []:
        return total_score
    else:
        if isInside(bulls_eye_coordinates[0] , bulls_eye_coordinates[1] , bulls_eye_coordinates[2] , cntrs_blue[0][0] , cntrs_blue[0][1]):
            total_score = total_score + 50
        else:
            total_score = total_score + 0
        
    return total_score

# In[10]:


video_path = 'vid.mp4'
output_path = 'output'


# In[ ]:





# In[27]:


cap= cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
i=0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print(frame_width , frame_height)
size = (1280, 720)

# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('Output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         cap.get(cv2.CAP_PROP_FPS), size , True)
while(cap.isOpened()):
    ret, frame = cap.read()
    # cv2.imshow('frame',frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if ret==True:




    

        image_color = frame.copy()
        image_mask = image_color.copy()
        pts_score = [[(950, 245), (835, 475), (1025, 295)]
            , [(1025, 295), (842, 475), (1085, 370)]
            , [(1085, 370), (848, 480), (1105, 450)]
            , [(1105, 450), (848, 490), (1110, 530)]
            , [(1111, 530), (848, 496), (1085, 620)]
            , [(1085, 621), (845, 500), (1030, 700)]
            , [(1030, 701), (840, 505), (955, 750)]
            , [(955, 750), (833, 510), (870, 775)]
            , [(870, 776), (825, 512), (780, 775)]
            , [(779, 775), (815, 510), (695, 745)]
            , [(693, 745), (810, 503), (618, 688)]
            , [(618, 688), (806, 500), (565, 612)]
            , [(565, 612), (805, 495), (545, 530)]
            , [(545, 530), (803, 487), (545, 445)]
            , [(545, 445), (805, 480), (570, 360)]
            , [(570, 360), (807, 475), (630, 288)]
            , [(630, 288), (813, 470), (703, 243)]
            , [(703, 243), (817, 468), (780, 220)]
            , [(780, 220), (823, 470), (865, 220)]
            , [(865, 220), (830, 470), (948, 243)]]
        total_score = 0
        for i in range(len(pts_score)):
            cnt = cv2.polylines(image_mask , np.array([pts_score[i]]) , True , (255,0,0 , 3))
        
        
        img = image_color[:,:,0]

        result_board , x_center , y_center = dartBoard_detection(image_color , img)

        bulls_eye_coordinates = [x_center , y_center-8 , 30]
        

        result_blue , cntrs_blue = blue_dart_detection(image_color)
        # cv2.imwrite(str(output_path)+'/' + image_name[:-4] + '_blue' + '.jpg' , result_blue)

        result_yellow , cntrs_yellow = yellow_dart_detection(image_color)
        # cv2.imwrite(str(output_path)+'/' + image_name[:-4] + '_yellow' + '.jpg' , result_yellow)

        circles = inner_circle_detection(image_color)

        default_score = 301
        score_blue = calculate_score(cntrs_blue , circles)
        score_blue = bulls_eye_score_calculator(bulls_eye_coordinates , cntrs_blue , score_blue)
        score_blue = default_score - score_blue

        score_yellow = calculate_score(cntrs_yellow , circles)
        score_yellow = bulls_eye_score_calculator(bulls_eye_coordinates , cntrs_yellow , score_yellow)
        score_yellow = default_score - score_yellow
        
        text = "Player 1 Score Is : " + str(score_yellow)
        text2 = " Player 2 Score Is : " + str(score_blue)
        

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Use putText() method for
        # inserting text on video
        cv2.putText(result_yellow, text, (50, 50), font, 1, (0, 0, 0), 2, cv2.LINE_4)
        cv2.putText(result_yellow, text2, (50, 100), font, 1, (0, 0, 0), 2, cv2.LINE_4)
        i+=1

        print(result_yellow.shape)

        #if score_yellow < 301 and score_blue < 301:
         #   print("Hey")
          #  cv2.imshow('frame2', image_mask)
           # cv2.imshow('frame', result_yellow)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            
#         out.write(result_yellow)
#         frame = result_yellow.copy()
        result.write(result_yellow)
        cv2.imshow('frame_mask', image_mask)   
        cv2.imshow('frame2', result_yellow)
        #cv2.imshow('frame',result_yellow)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    else:
        break

cap.release()
result.release()
cv2.destroyAllWindows()


# In[ ]:





# In[24]:


result_yellow.shape

