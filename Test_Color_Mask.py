import cv2
import numpy as np
import time
import tools
import board
import imutils
import os

images = os.listdir("test_images/")
# images = ["test2.jpg"]

failed = []

red_means = []

for image_name in images:
    img = cv2.imread("test_images/{}".format(image_name))

    ts_board = board.TwilightStruggleBoard(img)

    ussr_controlled = ts_board.get_ussr_controlled()
    us_controlled = ts_board.get_us_controlled()

    print(len(ussr_controlled), len(us_controlled))

    print("Europe: " + str(ts_board.get_europe_score()))
    print("Asia: " + str(ts_board.get_asia_score()))
    print("Middle East: " + str(ts_board.get_middle_east_score()))
    print("Africa: " + str(ts_board.get_africa_score()))
    print("South America: " + str(ts_board.get_south_america_score()))
    print("Central America: " + str(ts_board.get_central_america_score()))
    print("Southeast Asia: " + str(ts_board.get_sea_score()))

    cv2.imshow("{}".format(image_name), cv2.resize(img, (1280, 960)))
    cv2.waitKey(0)