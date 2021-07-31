# import the necessary packages
from libraries.videostream import WebcamVideoStream as vstream
import numpy as np
import argparse
import time
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import time
import libraries.plate_detector as p_reader

# construct the argument parser and parse the arguments
original = sys.stdout
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",default="tflite_model.tflite",
	help="path to tflite trained model [STRING]")
ap.add_argument("-a", "--address",default=0,
	help="stream address\n[rtsp] = rtsp://[username]:[password]@[ipaddress]/[camera specific additions] [STRING]\n[webcam] = 0 [INT]")
ap.add_argument("-cl", "--classes",default="obj.names",
	help="path to classes file of the model [STRING]")
ap.add_argument("-l", "--log",default=True,
	help="creating a log of recognized numbers [True\False]")
ap.add_argument("-v", "--view",default=False,
	help="viewing result for debugging [True/False]")
ap.add_argument("-sl", "--sleep",default=0,
	help="amount of sleep time between frames in seconds [DOUBLE]")
ap.add_argument("-si", "--size",default=416,
	help="size of the pictures the model was trained on [INT]")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# loads tensorflow and the detector library
print("[INFO] loading detector...")
import libraries.tf_lite_detect as detector

# loads the classes file
print("[INFO] processing the classes file...")
classes = detector.read_class_names(args["classes"])

# initializing the plate reader
print("[INFO] loading plate detector...")
anpr = p_reader.plate_reader(debug=False)

print("[INFO] loading tesseract ocr...")
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = vstream(src=args["address"]).start()
time.sleep(1)
print("[INFO] Running...")
# start the FPS throughput estimator
fps = 0.0
startTime = time.time()
start = time.time()
end = time.time()

size=args["size"]
conf=args["confidence"]
model=args["model"]
view=args["view"]
sl= args["sleep"]
log= args["log"]

log_file=open("log.txt","a")

# loop over frames from the video file stream
while (True):
	# grab the frame from the threaded video stream
	frame = vs.read()
	# resize the frame to have the width and height that the model requires
	resized_frame = cv2.resize(frame, (size,size))
	# construct a blob from the image
	s=time.time()
	boxes = detector.detect(framework='tflite', weights=model, size=size, tiny=True, model='yolov4',
							str_input=False, image=resized_frame, pic_output=False, output='result.png', iou=0.45, score=conf,
							classes_file=classes)
	e=time.time()
	#print("tflite:",e-s)
	s=time.time()
	for x in boxes:
		if (x['class'] == "Car") or (x['class'] == "Motorcycle"):
			image = frame[x['c0']:x['c2'], x['c1']:x['c3']]
			cv2.imshow("car", image)
			(lpText, lpCnt) = anpr.find_and_ocr(image, psm=7,
												clearBorder=True)
			# only continue if the license plate was successfully OCR'd
			if lpText is not None and lpCnt is not None:
				print("[INFO] ", lpText)
				log_file.write(str(time.ctime())+"  --  "+lpText+"\n")
			if view and lpText is not None and lpCnt is not None :
				print(view)
				# fit a rotated bounding box to the license plate contour and
				# draw the bounding box on the license plate
				box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
				box = box.astype("int")
				cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
				# compute a normal (unrotated) bounding box for the license
				# plate and then draw the OCR'd license plate text on the
				# image
				(x, y, w, h) = cv2.boundingRect(lpCnt)
				cv2.putText(image, lpText, (x, y - 15),
							cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
				# show the output ANPR image
				cv2.imshow("Output ANPR", image)
				cv2.waitKey(0)

	e=time.time()
	#print("plates:",e-s)
	# update the FPS counter
	fps+=1.0
	# update the end time
	end = time.time()
	if(fps%100==0):
		print("[INFO] approx. FPS: {:.2f}".format(fps/(end-start)))
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	time.sleep(sl)
        
# stop the timer and display FPS information
elapsed = time.time()-startTime
fps = fps/elapsed
print("[INFO] elasped time: {:.2f}".format(elapsed))
print("[INFO] approx. FPS: {:.2f}".format(fps))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()