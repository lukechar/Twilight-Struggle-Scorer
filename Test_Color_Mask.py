import cv2
import numpy as np
import time
import tools
import imutils
import os

images = os.listdir("test_images/")
# images = ["test9.jpg"]

failed = []

red_means = []

for image_name in images:
    img = cv2.imread("test_images/{}".format(image_name))
    height, width, channels = img.shape

    img_wo_tracks = img.copy()

    # Draw rectangle to zero turn record and space track
    cv2.rectangle(img_wo_tracks, (int(width * 0.6), 0), (width, int(height * 0.3)), (0,0,0), -1)

    # Draw rectangle to zero military ops track
    cv2.rectangle(img_wo_tracks, (int(width * 0.275), int(height * 0.83)), (int(width * 0.5), int(height)), (0,0,0), -1)

    # Draw rectangle to zero action round track
    cv2.rectangle(img_wo_tracks, (0, 0), (int(width * 0.365), int(height * .2)), (0,0,0), -1)

    # cv2.imshow("Rects", cv2.resize(img_wo_tracks, (1280, 960)))
    # cv2.waitKey(0)

    imMan = tools.ImageManipulations(img_wo_tracks)
    red_masked = imMan.apply_color_mask((0, 7), (140, 255), (140, 255))
    blue_masked = imMan.apply_color_mask((100, 110), (180, 255), (140, 255))

    sd = tools.ShapeDetector()

    ussr_controlled = 0
    us_controlled = 0

    # Convert to grayscale by keeping only the (V)alue channel
    red_gs = cv2.split(red_masked)[2]
    blue_gs = cv2.split(blue_masked)[2]

    # Apply thresholding with mask result
    red_thresh = cv2.threshold(red_gs, 110, 255, cv2.THRESH_BINARY)[1]
    blue_thresh = cv2.threshold(blue_gs, 0, 255, cv2.THRESH_BINARY)[1]
    

    # Find coutours
    cnts_red = cv2.findContours(red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contours_red = imutils.grab_contours(cnts_red)
    cnts_blue = cv2.findContours(blue_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contours_blue = imutils.grab_contours(cnts_blue)

    red_gs_color = cv2.cvtColor(red_gs, cv2.COLOR_GRAY2BGR)
    blue_gs_color = cv2.cvtColor(blue_gs, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(red_gs_color, contours_red, -1, (0,0,255), 5)
    cv2.drawContours(blue_gs_color, contours_blue, -1, (255,0,0), 5)

    area_min = 3000
    area_max = 4500

    square_tolerance = 0.2
    square_sides_tolerance = 0
    sqaure_center_of_mass_tolerance = 40
    square_side_length_tolerance = 60

    # Count Red control tokens
    for c in contours_red:
        c = cv2.convexHull(c)
        c_area = cv2.contourArea(c)
        if c_area < area_min or c_area > area_max:
            continue
        if sd.isSquare(c, tol=square_tolerance, sides_tol=square_sides_tolerance, center_of_mass_tol=sqaure_center_of_mass_tolerance, side_length_tol=square_side_length_tolerance):
            cv2.drawContours(img, [c], -1, (0,255,0), 3)   
            ussr_controlled += 1
        else:
            cv2.drawContours(img, [c], -1, (0,0,255), 3)  # pass area, fail square --> red      

    # Count Blue control tokens
    for c in contours_blue:
        c = cv2.convexHull(c)
        c_area = cv2.contourArea(c)
        if c_area < area_min or c_area > area_max:
            continue
        if sd.isSquare(c, tol=square_tolerance, sides_tol=square_sides_tolerance, center_of_mass_tol=sqaure_center_of_mass_tolerance, side_length_tol=square_side_length_tolerance):
            # if imMan.check_mean_hue_of_contour(c, 106, 10)[0]:
            cv2.drawContours(img, [c], -1, (0,255,0), 3)
            us_controlled += 1
        else:
            cv2.drawContours(img, [c], -1, (0,0,255), 3)  # pass area, fail square --> red

    # Display Results
    cv2.rectangle(img, (0, int(height * 0.1)), (int(width * 0.25), int(height * 0.35)), (0,255,0), -1)
    cv2.putText(img, "USSR Controlled: {}".format(ussr_controlled), (0, int(height * 0.2)), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=5)
    cv2.putText(img, "US Controlled: {}".format(us_controlled), (0, int(height * 0.24)), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), thickness=5)

    if ussr_controlled == us_controlled == 14:
        print("PASS:")
        # cv2.imshow("final - PASS", cv2.resize(img, (1280, 960)))
        # cv2.waitKey(0)
    else: 
        print("FAIL:")
        failed.append(image_name)
        cv2.imshow("final - FAIL", cv2.resize(img, (1280, 960)))
        cv2.waitKey(0)
    print(ussr_controlled, us_controlled)
print()
if len(failed) > 0:
    print(f"Failed: {failed}")
else:
    print("All pass!")