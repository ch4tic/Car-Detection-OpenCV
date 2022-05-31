#!/usr/bin/python3

import cv2
import sys
import os 

car_model = cv2.CascadeClassifier('../data/haarcascade_car.xml') # loading the car detection model

# function for finding the correct coordinates of cars and drawing rectangles around them
def car_detection(frame):
    # detecting cars and drawing blue rectangles to mark them
    cars = car_model.detectMultiScale(frame, 1.15, 4) 
    for(x, y, w, h) in cars:
        color_text = (255, 20, 99)
        color_rectangle = (255, 20, 99)
        fontScale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.rectangle(frame, (x,y), (x+w, y+h), color_rectangle, thickness=2)
        cv2.putText(image, 'Car', (x, y-10), font, 0.5, color_text, 2)
    return frame

# function for loading and rendering the given video
def main():
    video_number = input("What video do you want to use(1-2)")
    video_number = int(video_number)
    video = cv2.VideoCapture('../data/cars' + str(video_number) + '.mp4') 
    if video_number > 2:
        print("Invalid number!")
        sys.exit()
    elif video_number < 1: 
        print("Invalid number!")
        sys.exit()
 
    while video.isOpened():
        ret, frame = video.read()
        exitKey = cv2.waitKey(1)
        if ret:
            cars_frame = car_detection(frame) # calling the car_detection function and drawing rectangles
            cv2.imshow('frame', cars_frame) # showing the video 
        else:
            break
        # if user presses 'q' the program ends
        if exitKey == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
