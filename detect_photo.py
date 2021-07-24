import cv2
import imutils
import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import libraries.tf_lite_detect as detector
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--p", required=True,
                help="path to input directory of faces + images")
args = vars(ap.parse_args())

img = cv2.imread(args["p"])
cv2.imshow("res", img)
cv2.waitKey()

boxes = detector.detect('tflite', 'tflite_vehicle_recognition_ver2.tflite', size=416, tiny=True, model='yolov4',
                        str_input=False, image=img, pic_output=False, output='result.png', iou=0.45, score=0.25,
                        classes_file=detector.read_class_names('obj.names'))

for x in boxes:
    if (x['class'] == "Car") or (x['class'] == "Motorcycle"):
        image= img[x['c0']:x['c2'], x['c1']:x['c3']]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(gray, 300, 600)
        cv2.imshow("a",edged)
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)#grabs the contours according to opencv ver
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = []
        cv2. imshow('gray',edged)
        cv2.waitKey(0)
        for c in contours:
            peri = cv2.arcLength(c, True)#shape boundery length (true symbolises closed)
            approx = cv2.approxPolyDP(c, 0.018	* peri, True)

            if len(approx) == 4:
                screenCnt.append(approx)

        if screenCnt is []:
            detected = 0
            print ("No contour detected")
        else:
             detected = 1

        if detected == 1:
            for x in screenCnt:
                cv2.drawContours(image, [x], -1, (0, 0, 255), 3)
                cv2.imshow('img',image)
                cv2.waitKey(0)
                xs=[]
                ys=[]
                for y in x:
                    xs.append(y[0][0])
                    ys.append(y[0][1])
                xmin= np.min(xs)
                xmax = np.max(xs)
                ymin = np.min(ys)
                ymax = np.max(ys)
                cropped = image[ymin:ymax,xmin:xmax]
                cv2.imshow("cropped",cropped)
                cv2.waitKey(0)

                text = pytesseract.image_to_string(cropped, config='--psm 11')
                print("programming_fever's License Plate Recognition\n")
                print("Detected license plate Number is:",text)
                img = cv2.resize(img,(500,300))
                cropped = cv2.resize(cropped,(400,200))
                cv2.imshow('car',img)
                cv2.imshow('Cropped',cropped)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
