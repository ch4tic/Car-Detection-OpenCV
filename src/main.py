#!/usr/bin/python3

import cv2
import os 

car_model = cv2.CascadeClassifier('../data/haarcascade_car.xml')

# function for recieving image frames and drawing rectangles around it using the detected coordinates
def car_detection(frame):
    cars = car_model.detectMultiScale(frame, 1.15, 4)
    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), color=(255, 0, 0), thickness=2)
    return frame

# function for loading and rendering the given video
def simulator():
    video = cv2.VideoCapture('../data/cars.mp4')
    while video.isOpened():
        ret, frame = video.read()
        controlkey = cv2.waitKey(1)
        if ret:
            cars_frame = car_detection(frame)
            cv2.imshow('frame', cars_frame)
        else:
            break
        if controlkey == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    simulator()
