import cv2
import imutils

from tools import ImageManipulations, ShapeDetector

class TwilightStruggleBoard:

    CONTROL_TOKEN_AREA_MIN = 3000
    CONTROL_TOKEN_AREA_MAX = 4500
    CONTROL_TOKEN_SQUARE_TOL = 0.2
    CONTROL_TOKEN_SQUARE_COM_TOL = 40
    CONTROL_TOKEN_SQAURE_SIDE_LEN_TOL = 10

    RED_MASK_HUE_RANGE = (0, 7)
    RED_MASK_SAT_RANGE = (140, 255)
    RED_MASK_VAL_RANGE = (140, 255)

    BLUE_MASK_HUE_RANGE = (100, 110)
    BLUE_MASK_SAT_RANGE = (180, 255)
    BLUE_MASK_VAL_RANGE = (140, 255)
    
    def __init__(self, img):
        if len(img) == 0:
            raise ValueError("Empty image!")
        self.img = img
        # Get image shape properties
        self.height, self.width, self.channels = self.img.shape
        # Convert image to HSV color space
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.ussr_controlled_positions = []
        self.us_controlled_positions = []
        
        # For debugging
        self.ussr_controlled_contours = []
        self.us_controlled_contours = []

        self.process_board()

    def process_board(self):
        # Apply track masks
        img_wo_tracks = self.img.copy()
        # Draw rectangle to zero turn record and space track
        cv2.rectangle(img_wo_tracks, (int(self.width * 0.6), 0), (self.width, int(self.height * 0.3)), (0,0,0), -1)
        # Draw rectangle to zero military ops track
        cv2.rectangle(img_wo_tracks, (int(self.width * 0.275), int(self.height * 0.83)), (int(self.width * 0.5), int(self.height)), (0,0,0), -1)
        # Draw rectangle to zero action round track
        cv2.rectangle(img_wo_tracks, (0, 0), (int(self.width * 0.365), int(self.height * .2)), (0,0,0), -1)

        imMan = ImageManipulations(img_wo_tracks)
        red_masked = imMan.apply_color_mask(self.RED_MASK_HUE_RANGE, self.RED_MASK_SAT_RANGE, self.RED_MASK_VAL_RANGE)
        blue_masked = imMan.apply_color_mask(self.BLUE_MASK_HUE_RANGE, self.BLUE_MASK_SAT_RANGE, self.BLUE_MASK_VAL_RANGE)

        sd = ShapeDetector()

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

        # Find/verify Red control tokens
        for c in contours_red:
            c = cv2.convexHull(c)
            c_area = cv2.contourArea(c)
            if c_area < self.CONTROL_TOKEN_AREA_MIN or c_area > self.CONTROL_TOKEN_AREA_MAX:
                continue
            if sd.isSquare(c, self.CONTROL_TOKEN_SQUARE_TOL, self.CONTROL_TOKEN_SQUARE_COM_TOL, self.CONTROL_TOKEN_SQAURE_SIDE_LEN_TOL):
                self.ussr_controlled_positions.append(sd.get_contour_center(c))

        # Find/verify Blue control tokens
        for c in contours_blue:
            c = cv2.convexHull(c)
            c_area = cv2.contourArea(c)
            if c_area < self.CONTROL_TOKEN_AREA_MIN or c_area > self.CONTROL_TOKEN_AREA_MAX:
                continue
            if sd.isSquare(c, self.CONTROL_TOKEN_SQUARE_TOL, self.CONTROL_TOKEN_SQUARE_COM_TOL, self.CONTROL_TOKEN_SQAURE_SIDE_LEN_TOL):
                self.us_controlled_positions.append(sd.get_contour_center(c))

    # For debugging purposes
    def get_ussr_controlled_contours(self):
        return self.ussr_controlled_contours

    # For debugging purposes
    def get_us_controlled_contours(self):
        return self.us_controlled_contours

    def get_ussr_controlled(self):
        return len(self.ussr_controlled_positions)

    def get_us_controlled(self):
        return len(self.us_controlled_positions)

    def get_ussr_battlegrounds(self):
        pass

    def get_us_battlegrounds(self):
        pass

    def get_europe_score(self):
        pass

    def get_asia_score(self):
        pass

    def get_africa_score(self):
        pass

    def get_middle_east_score(self):
        pass

    def get_central_america_score(self):
        pass

    def get_south_america_score(self):
        pass
