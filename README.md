# Dart-Score-Counting
This is a basic real time dart score counting algorithm, which is followed by the 301 dart rules. I tried to develop the algorithm based on color detection method. Which means this algorithm only detects Blue and Yellow colored darts. Make sure to install OpenCV version 3 !

# Required Hardware
To use the framework, you need the following hardware:
1. Raspberry Pi / windows based PC
2. HD web-cam
3. A dart board
4. A monitor

# Settings:
Adjust the camera in-front of the dart board in such way that it can capture the whole board with clear details, since you need to throw the darts to the board, so the position of the camera is very crucial. After setting up the camera properly, run the *dart_detection.py* script to detect the board and callibrate the mask for counting scores(you can perform it by looking at the screen with live video). After finishing the callibration of the mask with the dart board, run the *Dart_Detection_Final.py* script to start playing the game. 
Since, the algorithm only works with blue and yellow colored darts, so you must have to use these colored darts. 

*The algorithm is not 100% accurate, because sometimes it does not detect the color of the darts, due to the poor lighting conditions or any kind of shadows/noises.*
