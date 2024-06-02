
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

input_path = cv2.VideoCapture (0)
output_path = 'output'
ret, image_color= input_path.read()

# Extract name for output saving
image_name = input_path

print(image_name)

def blue_dart_detection(image_color):
  image_ori = image_color

  image_color = cv2.cvtColor(image_color,cv2.COLOR_BGR2HSV)

  # Blue Thrsholds
  lower_bound = np.array([90, 50, 70])
  upper_bound = np.array([128, 255, 255])

  image = image_ori

  mask = cv2.inRange(image_color, lower_bound, upper_bound)

  # mask = cv2.adaptiveThreshold(image_ori,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
  #             cv2.THRESH_BINARY_INV,33,2)

  kernel = np.ones((3, 3), np.uint8)

  #Use erosion and dilation combination to eliminate false positives. 
  #In this case the text Q0X could be identified as circles but it is not.
  # cv2_imshow(mask)
  # mask = cv2.erode(mask, kernel, iterations=6)
  # cv2_imshow(mask)
  # mask = cv2.dilate(mask, kernel, iterations=3)
  # cv2_imshow(mask)
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
    return image
  for c in contours:
      (x,y),r = cv2.minEnclosingCircle(c)
      center = (int(x),int(y))
      r = int(r)
      #print(r)
      if r >= 20:
          cv2.circle(image,center,r,(255,0,0),2)

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


          #cv2.putText(image, 'Blue', center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,58,238), 2)

          array.append(center)

  # cv2_imshow(image)
  cv2.imshow('Blue Output', image) 
  cv2.waitKey(0) 
  cv2.destroyAllWindows()
  return image


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

  # mask = cv2.adaptiveThreshold(image_ori,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
  #             cv2.THRESH_BINARY_INV,33,2)

  kernel = np.ones((3, 3), np.uint8)

  #Use erosion and dilation combination to eliminate false positives. 
  #In this case the text Q0X could be identified as circles but it is not.
  
  closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

  contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
          cv2.CHAIN_APPROX_SIMPLE)[0]
  contours.sort(key=lambda x:cv2.boundingRect(x)[0])

  array = []

  ii = 1
  # print(len(contours))
  if len(contours) == 0:
    print("No Yellow color detected.")

  for c in contours:
      (x,y),r = cv2.minEnclosingCircle(c)
      center = (int(x),int(y))
      
      r = int(r)
      #print(r)
      if r >= 20:
          cv2.circle(image,center,r,(0,255,255),2)

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

  # cv2_imshow(image)
  cv2.imshow('Yellow Output', image) 
  cv2.waitKey(0) 
  cv2.destroyAllWindows()
  return image

def dartBoard_detection(img):
 
  # Convert the image to gray-scale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Find the edges in the image using canny detector
  edges = cv2.Canny(gray, 100, 1000)
  
    

  img= input_path   
  c_img = img
  
  # Apply hough transform on the image
  circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,500,param1=1000,param2=110,minRadius=90,maxRadius=350)
  # circles = cv2.HoughCircles(c_img,cv2.HOUGH_GRADIENT,1,20,param1=100,param2=150,minRadius=0,maxRadius=0)
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
         
  
  cv2.imshow('DartBoard Output', c_img) 
  cv2.waitKey(0) 
  cv2.destroyAllWindows()
  return c_img



# image_name = input_path[52:]
image_color = cv2.resize(image_color, (0,0), fx=0.5, fy=0.5) 

result_board = dartBoard_detection(image_color)
cv2.imwrite(str(output_path)+'/' + image_name[:-4] + '_board' + '.jpg' , result_board)

result_blue = blue_dart_detection(image_color)
cv2.imwrite(str(output_path)+'/' + image_name[:-4] + '_blue' + '.jpg' , result_blue)

result_yellow = yellow_dart_detection(image_color)
cv2.imwrite(str(output_path)+'/' + image_name[:-4] + '_yellow' + '.jpg' , result_yellow)

# dst = cv2.addWeighted(result_yellow,1,result_blue,1,0)
# cv2_imshow(dst)


