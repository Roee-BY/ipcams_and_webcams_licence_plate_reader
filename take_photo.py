from libraries.videostream import WebcamVideoStream as vstream
import numpy as np
import argparse
import time
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

original = sys.stdout
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--address", required=True,
	help="camera stream adress")
ap.add_argument("-o", "--out", required=True,
	help="pic output address")
args = vars(ap.parse_args())

vs = vstream(src=args["address"]).start()
frame = vs.read()
cv2.imwrite(args["out"], frame)