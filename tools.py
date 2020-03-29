import cv2
import numpy as np

class ShapeDetector:
    def __init__(self, *args, **kwargs):
        pass

    # Get center of contour
    # @param c - the contour to get the center of using bounding rectangle
    # @return [2 tuple of ints] - the center of the contour using bounding rectangle
    def get_contour_center(self, c):
        (x, y, w, h) = cv2.boundingRect(c)
        return (x + (w // 2), y + (h // 2))

    # Check if contour is rectangle
    # @param c - the contour to check
    # @return [bool] - True if contour is a rectangle, False otherwise
    def isRectangle(self, c):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        return len(approx) == 4

    # Check if contour is square
    # @param c - the contour to check
    # @return [bool] - True if contour is a square, False otherwise
    def isSquare(self, c, tol, center_of_mass_tol, side_length_tol):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        if len(approx) != 4:
            return False
        side_sum = 0
        side_lengths = []
        for vertex in range(4):
            for v in range(vertex + 1, 4):
                dist = np.linalg.norm(approx[vertex][0] - approx[v][0])
                side_lengths.append(dist)
        side_lengths = sorted(side_lengths)[:4]
        for s in side_lengths:
            side_sum += s
        avg_side_length = side_sum / len(side_lengths)
        for l in side_lengths:
            if l < avg_side_length - side_length_tol or l > avg_side_length + side_length_tol:
                return False
        # Check width to height ratio of bounding box
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # Check that center of mass coincides with bounding rectangle center (within tolerance)
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        x_diff = cx - x
        y_diff = cy - y
        return ar >= (1.0 - tol) and ar <= (1.0 + tol) and cx >= (x - center_of_mass_tol) and cx <= (x + center_of_mass_tol) and cy >= (y - center_of_mass_tol) and cy <= (y + center_of_mass_tol)

class ImageManipulations:
    def __init__(self, img):
        if len(img) == 0:
            raise ValueError("Empty image!")
        self.img = img
        # Convert image to HSV color space
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

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
        mask = None
        # Generate color mask
        # Hue wraps-around 180 deg
        if hue_bounds[1] < hue_bounds[0]:
            lower_1 = np.array([0, saturation_bounds[0], value_bounds[0]])
            upper_1 = np.array([hue_bounds[1], saturation_bounds[1], value_bounds[1]])
            lower_2 = np.array([hue_bounds[0], saturation_bounds[0], value_bounds[0]])
            upper_2 = np.array([180, saturation_bounds[1], value_bounds[1]])
            mask = cv2.inRange(self.hsv, lower_1, upper_1) + cv2.inRange(self.hsv, lower_2, upper_2)
        # Hue range is between 0 and 180
        else:
            lower = np.array([hue_bounds[0], saturation_bounds[0], value_bounds[0]])
            upper = np.array([hue_bounds[1], saturation_bounds[1], value_bounds[1]])
            mask = cv2.inRange(self.hsv, lower, upper)

        return cv2.bitwise_and(self.img, self.img, mask=mask)

    # def check_mean_hue_of_contour(self, c, mean_hue, mean_hue_tolerance):
    #     hue_channel = cv2.split(self.hsv)[0]
    #     mask = np.zeros(hue_channel.shape, np.uint8)
    #     # # Get contour center
    #     # x,y,w,h = cv2.boundingRect(c)
    #     # c_center = (x + w // 2, y + h // 2)
    #     # # Shrink contour to remove edge colors
    #     # Draw (filled-in) contour on mask
    #     cv2.drawContours(mask, [c], -1, 255, -1)
    #     # cv2.imshow("b4", cv2.resize(mask, (1280, 960)))
    #     # cv2.waitKey(0)
    #     mask = cv2.erode(mask, None, iterations=3)
    #     # cv2.imshow("after", cv2.resize(mask, (1280, 960)))
    #     # cv2.waitKey(0)
    #     # Get mean hue of masked hue channel
    #     mean_hue_measured = cv2.mean(hue_channel, mask=mask)[0]
    #     return mean_hue - mean_hue_tolerance < mean_hue_measured and mean_hue_measured < mean_hue + mean_hue_tolerance, mean_hue_measured