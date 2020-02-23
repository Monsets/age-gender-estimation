import face_model
import argparse
import cv2
import sys
import numpy as np
import datetime
import time


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
  size = cv2.getTextSize(label, font, font_scale, thickness)[0]
  x, y = point
  cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
  cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


cap = cv2.VideoCapture('test1.avi')

scales = [int(cap.get(3)), int(cap.get(4))]


parser = argparse.ArgumentParser(description='face model_ag test')
# general
parser.add_argument('--image-size', default='{},{}'.format(int(cap.get(3)), int(cap.get(4))), help='')
parser.add_argument('--image', default='sample-images/test1.jpg', help='')
parser.add_argument('--model_ag', default='./model_ag/m1/model_ag,0', help='path to load model_ag.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=1, type=int, help='mtcnn or essh option, 0 means mtcnn, 1 means essh')
args = parser.parse_args()


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter('result.avi', fourcc, 20, (scales[0], scales[1]), True)

model = face_model.FaceModel(args)

start = time.time()
n_frames = 0
while(cap.isOpened()):
    ret, img = cap.read()
    print("Frame {}".format(n_frames))
    n_frames += 1
    if ret == True:
        img_db, bbox, points = model.get_input(img, args)

        if img_db is not None:
            for _ in range(1):
                gender, age = model.get_ga(img_db)

            for b in bbox:
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            for p in points:
                for i in range(5):
                    cv2.circle(img, (p[i][0], p[i][1]), 1, (0, 0, 255), 2)
            for i in range(len(age)):
                label = "{}, {}".format(int(age[i]), "F" if gender[i] == 0 else "M")
             draw_label(img, (int(bbox[i,0]), int(bbox[i,1])), label)

        writer.write(img)
    else:
        break

finish = time.time()
print("{} frames per second".format(n_frames / (finish - start)))

# When everything done, release the video capture object
cap.release()
writer.release()
