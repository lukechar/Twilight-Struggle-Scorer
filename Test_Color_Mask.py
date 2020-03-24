import cv2
import numpy as np
import time
import tools
import imutils

img = cv2.imread("test_images/test3.jpg")
height, width, channels = img.shape

img_wo_tracks = img.copy()

# Draw rectangle to zero turn record and space track
cv2.rectangle(img_wo_tracks, (int(width * 0.6), 0), (width, int(height * 0.33)), (255,0,0), -1)

# Draw rectangle to zero military ops track
cv2.rectangle(img_wo_tracks, (int(width * 0.275), int(height * 0.815)), (int(width * 0.465), int(height)), (255,0,0), -1)

imMan = tools.ImageManipulations(img_wo_tracks)
red_masked = imMan.apply_color_mask((175, 5), (110, 255), (0, 255))
blue_masked = imMan.apply_color_mask((100, 110), (110, 255), (0, 255))

sd = tools.ShapeDetector()

ussr_controlled = 0
us_controlled = 0

# Apply thresholding with mask result
red_thresh = cv2.threshold(red_masked, 120, 255, cv2.THRESH_BINARY)[1]
blue_thresh = cv2.threshold(blue_masked, 0, 255, cv2.THRESH_BINARY)[1]

# Convert to grayscale by keeping only the (V)alue channel
red_gs = cv2.split(red_thresh)[2]
blue_gs = cv2.split(blue_thresh)[2]

# Find coutours
cnts_red = cv2.findContours(red_gs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_red = imutils.grab_contours(cnts_red)
cnts_blue = cv2.findContours(blue_gs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_blue = imutils.grab_contours(cnts_blue)

cv2.imshow("thresh", cv2.resize(red_gs, (1280, 960)))
cv2.waitKey(0)
cv2.imshow("thresh", cv2.resize(blue_gs, (1280, 960)))
cv2.waitKey(0)

area_min = 3000
area_max = 5000

# Count Red control tokens
for c in contours_red:
    c_area = cv2.contourArea(c)
    if c_area < area_min or c_area > area_max:
        continue
    if sd.isSquare(c, tol=0.5):
        cv2.drawContours(img, [c], -1, (0,255,0), 3)
        ussr_controlled += 1
    else:
        cv2.drawContours(img, [c], -1, (255,0,0), 3)

# Count Blue control tokens
for c in contours_blue:
    c_area = cv2.contourArea(c)
    if c_area < area_min or c_area > area_max:
        continue
    if sd.isSquare(c, tol=0.5):
        cv2.drawContours(img, [c], -1, (0,255,0), 3)
        us_controlled += 1
    else:
        cv2.drawContours(img, [c], -1, (0,0,255), 3)

# Display Results
cv2.putText(img, "USSR Controlled: {}".format(ussr_controlled), (0, int(height * 0.2)), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), thickness=5)
cv2.putText(img, "US Controlled: {}".format(us_controlled), (0, int(height * 0.24)), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), thickness=5)

cv2.imshow("final", cv2.resize(img, (1280, 960)))
cv2.waitKey(0)