import cv2
import numpy as np
import time
import tools
import board
import imutils
import os

images = os.listdir("test_images/")
# images = ["test9.jpg"]

failed = []

red_means = []

for image_name in images:
    img = cv2.imread("test_images/{}".format(image_name))

    ts_board = board.TwilightStruggleBoard(img)

    ussr_controlled = ts_board.get_ussr_controlled()
    us_controlled = ts_board.get_us_controlled()

    if ussr_controlled == us_controlled == 14:
        print("PASS")
        # cv2.imshow("final - PASS", cv2.resize(img, (1280, 960)))
        # cv2.waitKey(0)
    else: 
        print("FAIL")
        failed.append(image_name)
        # cv2.imshow("final - FAIL", cv2.resize(img, (1280, 960)))
        # cv2.waitKey(0)
    print(ussr_controlled, us_controlled)
print()
if len(failed) > 0:
    print(f"Failed: {failed}")
else:
    print("All pass!")