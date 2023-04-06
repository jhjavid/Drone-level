import cv2 as cv
import time
from pygame.locals import *
import pygame
import sys

# def on_key_release(key):
#     if key == Key.right:
#         print("Right key clicked")
#     elif key == Key.left:
#         print("Left key clicked")
#     elif key == Key.up:
#         print("Up key clicked")
#     elif key == Key.down:
#         print("Down key clicked")
#     elif key == Key.esc:
#         exit()


Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_CPU)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


cap = cv.VideoCapture('output.avi')
starting_time = time.time()
frame_counter = 0


while True:
    ret, frame = cap.read()

    window = pygame.display.set_mode((100, 100))
    frame_counter += 1
    if ret == False:
        break
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid[0]], score)
        if class_name[classid[0]] == 'person':
            cv.rectangle(frame, box, color, 1)
            cv.putText(frame, label, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
    
    
    w = int(frame.shape[0]/2)
    h = int(frame.shape[1]/2)
    cv.line(frame,(h+50,w),(h-50,w),(0, 255, 0),3)
    cv.line(frame,(h,w+50),(h,w-50),(0, 255, 0),3)
    for evenement in pygame.event.get():
        if evenement.type == QUIT or (evenement.type == KEYDOWN and evenement.key == K_ESCAPE):
            print('QUIT')
            pygame.quit()
            sys.exit()

        if evenement.type == KEYDOWN and evenement.key == K_RIGHT:
            print("Clicked on the right arrow")
            cv.line(frame,(h+60,w),(h-40,w),(0, 0, 255),3)
        if evenement.type == KEYDOWN and evenement.key == K_LEFT:
            print("Clicked on the left arrow")
            cv.line(frame,(h+40,w),(h-60,w),(0, 0, 255),3)
        if evenement.type == KEYDOWN and evenement.key == K_UP:
            print("Clicked on the up arrow")
            cv.line(frame,(h,w+40),(h,w-60),(0, 0, 255),3)
        if evenement.type == KEYDOWN and evenement.key == K_DOWN:
            print("Clicked on the down arrow")
            cv.line(frame,(h,w+60),(h,w-40),(0, 0, 255),3)
    
    cv.rectangle(frame,(frame.shape[1]-150,10),(frame.shape[1]-10,100),(0,0,255),-1)
    cv.putText(frame, f'Altitude: {frame_counter+100}', (frame.shape[1]-145,25),cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
    cv.putText(frame, f'Speed: {frame_counter}', (frame.shape[1]-145,50),cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
    cv.putText(frame, f'GPS: {frame_counter}', (frame.shape[1]-145,75),cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    cv.putText(frame, f'FPS: {int(fps)}', (10, 15),cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
