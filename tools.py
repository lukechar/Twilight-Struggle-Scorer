import cv2
import numpy as np

class ShapeDetector:
    def __init__(self, *args, **kwargs):
        pass

    def isRectangle(self, c):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        return len(approx) == 4

    def isSquare(self, c, tol=0.05):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) != 4:
            return False
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        return ar >= (1.0 - tol) and ar <= (1.0 + tol)

class ImageManipulations:
    def __init__(self, img):
        if len(img) == 0:
            raise ValueError("Empty image!")
        self.img = img

    def apply_color_mask(self, hue_bounds, saturation_bounds, value_bounds):
        try:
            assert len(hue_bounds) == len(saturation_bounds) == len(value_bounds) == 2
        except AssertionError as e:
            raise ValueError("All bounds must be 2-tuples")
        # Check input ranges
        for x in hue_bounds:
            if x < 0 or x > 180:
                raise ValueError("Hue values must be between 0 and 180")
        for x in saturation_bounds:
            if x < 0 or x > 255:
                raise ValueError("Saturation values must be between 0 and 255")
        for x in value_bounds:
            if x < 0 or x > 255:
                raise ValueError("\"Value\" values must be between 0 and 255")
        # Convert image to HSV color space
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        mask = None
        # Generate color mask
        # Hue wraps-around 180 deg
        if hue_bounds[1] < hue_bounds[0]:
            lower_1 = np.array([0, saturation_bounds[0], value_bounds[0]])
            upper_1 = np.array([hue_bounds[1], saturation_bounds[1], value_bounds[1]])
            lower_2 = np.array([hue_bounds[0], saturation_bounds[0], value_bounds[0]])
            upper_2 = np.array([180, saturation_bounds[1], value_bounds[1]])
            mask = cv2.inRange(hsv, lower_1, upper_1) + cv2.inRange(hsv, lower_2, upper_2)
        # Hue range is between 0 and 180
        else:
            lower = np.array([hue_bounds[0], saturation_bounds[0], value_bounds[0]])
            upper = np.array([hue_bounds[1], saturation_bounds[1], value_bounds[1]])
            mask = cv2.inRange(hsv, lower, upper)

        return cv2.bitwise_and(self.img, self.img, mask=mask)