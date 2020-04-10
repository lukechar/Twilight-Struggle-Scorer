import numpy as np
import cv2
import imutils
import os
import sys

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
        self.battleground_positions = []
        self.nonbattleground_positions = []
        self.ussr_controlled_countries = []
        self.us_controlled_countries = []
        self.template_locations = []
        self.reference_template_locations = []

        # Battleground names (bottom to top)
        self.battleground_names = [
            'Argentina', 
            'South Africa', 
            'Chile', 
            'Angola', 
            'Brazil', 
            'Zaire', 
            'Nigeria', 
            'Venezuela', 
            'Panama', 
            'Thailand', 
            'Saudi Arabia', 
            'India', 
            'Egypt', 
            'Cuba', 
            'Libya', 
            'Pakistan',
            'Mexico',
            'Israel',
            'Iraq',
            'Iran',
            'Algeria',
            'Japan',
            'South Korea',
            'Italy',
            'North Korea',
            'France',
            'West Germany',
            'East Germany',
            'Poland',
        ]
        # Battleground regions (bottom to top)
        self.battleground_regions = [
            'sa',
            'af',
            'sa',
            'af',
            'sa',
            'af',
            'af',
            'sa',
            'ca',
            'as',
            'me',
            'as',
            'me',
            'ca',
            'me',
            'as',
            'ca',
            'me',
            'me',
            'me',
            'af',
            'as',
            'as',
            'eu',
            'as',
            'eu',
            'eu',
            'eu',
            'eu',
        ]
        
        # Non-battleground names (bottom to top)
        self.nonbattleground_names = [
            'Uruguay',
            'Paraguay',
            'Botswana',
            'Australia',
            'Bolivia',
            'Zimbabwe',
            'Peru',
            'SE African States',
            'Indonesia',
            'Ecuador',
            'Kenya',
            'Cameroon',
            'Colombia',
            'Malaysia',
            'Somalia',
            'Ivory Coast',
            'Costa Rica',
            'Ethiopia',
            'Vietnam',
            'Philippines',
            'El Salvador',
            'Honduras',
            'Nicaragua',
            'Sudan',
            'Saharan States',
            'Haiti',
            'Dominican Republic',
            'West African States',
            'Laos/Cambodia',
            'Burma',
            'Guatemala',
            'Jordan',
            'Taiwan',
            'Gulf States',
            'Morocco',
            'Tunisia',
            'Afganistan',
            'Lebanon',
            'Greece',
            'Syria',
            'Spain/Portugal',
            'Turkey',
            'Yugoslavia',
            'Bulgaria',
            'Austria',
            'Hungary',
            'Romania',
            'Canada',
            'Benelux',
            'Czechoslovakia',
            'UK',
            'Denmark',
            'Sweden',
            'Norway',
            'Finland',
        ]
        # Non-battleground regions (bottom to top)
        self.nonbattleground_regions = [
            'sa',
            'sa',
            'af',
            'as',
            'sa',
            'af',
            'sa',
            'af',
            'as',
            'sa',
            'af',
            'af',
            'sa',
            'as',
            'af',
            'af',
            'ca',
            'af',
            'as',
            'as',
            'ca',
            'ca',
            'ca',
            'af',
            'af',
            'ca',
            'ca',
            'af',
            'as',
            'as',
            'ca',
            'me',
            'as',
            'me',
            'af',
            'af',
            'as',
            'me',
            'eu',
            'me',
            'eu',
            'eu',
            'eu',
            'eu',
            'eu',
            'eu',
            'eu',
            'eu',
            'eu',
            'eu',
            'eu',
            'eu',
            'eu',
            'eu',
            'eu',
        ]

        # For debugging
        self.ussr_controlled_contours = []
        self.us_controlled_contours = []

        self.process_board()

    def process_board(self):
        '''
        *************************************************
        Mask tracks on perhiphary of board 
        ***************************************************************
        '''        
        # Apply track masks
        img_wo_tracks = self.img.copy()
        img_wo_tracks_battleground_sample = cv2.imread(os.path.join("resources", "images", "battlegrounds.png"))
        img_wo_tracks_nonbattleground_sample = cv2.imread(os.path.join("resources", "images", "nonbattlegrounds.png"))

        # Draw rectangle to zero turn record and space track
        cv2.rectangle(img_wo_tracks, (int(self.width * 0.6), 0), (self.width, int(self.height * 0.3)), (0,0,0), -1)
        cv2.rectangle(img_wo_tracks_battleground_sample, (int(self.width * 0.6), 0), (self.width, int(self.height * 0.3)), (0,0,0), -1)
        cv2.rectangle(img_wo_tracks_nonbattleground_sample, (int(self.width * 0.6), 0), (self.width, int(self.height * 0.3)), (0,0,0), -1)
        # Draw rectangle to zero military ops track
        cv2.rectangle(img_wo_tracks, (int(self.width * 0.275), int(self.height * 0.83)), (int(self.width * 0.5), int(self.height)), (0,0,0), -1)
        cv2.rectangle(img_wo_tracks_battleground_sample, (int(self.width * 0.275), int(self.height * 0.83)), (int(self.width * 0.5), int(self.height)), (0,0,0), -1)
        cv2.rectangle(img_wo_tracks_nonbattleground_sample, (int(self.width * 0.275), int(self.height * 0.83)), (int(self.width * 0.5), int(self.height)), (0,0,0), -1)
        # Draw rectangle to zero action round track
        cv2.rectangle(img_wo_tracks, (0, 0), (int(self.width * 0.365), int(self.height * .2)), (0,0,0), -1)
        cv2.rectangle(img_wo_tracks_battleground_sample, (0, 0), (int(self.width * 0.365), int(self.height * .2)), (0,0,0), -1)
        cv2.rectangle(img_wo_tracks_nonbattleground_sample, (0, 0), (int(self.width * 0.365), int(self.height * .2)), (0,0,0), -1)

        imMan = ImageManipulations(img_wo_tracks)
        imMan_battleground_sample = ImageManipulations(img_wo_tracks_battleground_sample)
        imMan_nonbattleground_sample = ImageManipulations(img_wo_tracks_nonbattleground_sample)

        sd = ShapeDetector()
        '''
        *************************************************
        Get affine transformation contants
        ***************************************************************
        '''

        # Get locations of template images
        TEMPLATE_FOLDER = os.path.join("resources", "template_images")
        # templates = os.listdir(TEMPLATE_FOLDER)
        # templates = ["labrador.png", "tierra_del_fuego.png", "japan.png", "indonesia.png"]
        templates = ["labrador.png", "tierra_del_fuego.png", "japan.png"]
        for template_fn in templates:
            template = cv2.imread(f"{TEMPLATE_FOLDER}/{template_fn}", 0)  # load in grayscale
            res = cv2.matchTemplate(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), template, cv2.TM_CCORR_NORMED)
            minVal, bestFitRes, minLoc, bestFitLoc = cv2.minMaxLoc(res)  # For TM_CCORR_NORMED, the bestFitRes is the largest res (0 <= res <= 1)

            # Assert that found best template fit is at least 90%
            assert bestFitRes >= 0.9

            # Draw solid yellow circle on found template locations for debugging
            # cv2.circle(self.img, bestFitLoc, 12, (0, 255, 255), -1)

            self.template_locations.append(bestFitLoc)


        # with open("resources/reference_template_locations.dat", 'w') as f:
        #     for loc in self.template_locations:
        #         f.write(str(loc) + '\n')

        # Calculate tranformation matrix with reference matrix from file (taken from template positions of 'test2.jpg')
        # Load reference template positions
        with open(os.path.join("resources", "reference_template_locations.dat"), 'r') as f:
            pos_read = f.readlines()
        for loc in pos_read:
            stripped = loc.rstrip()
            loc_tuple = eval(stripped)
            self.reference_template_locations.append(loc_tuple)

        template_locations_mat = np.float32([self.template_locations[:3]])
        reference_template_locations_mat = np.float32([self.reference_template_locations[:3]])

        transform_mat = cv2.getAffineTransform(reference_template_locations_mat, template_locations_mat)

        '''
        *************************************************
        Find positions of all battlegrounds
        ***************************************************************
        '''
        # Get battleground positions from pink-marked (H=318.1 deg, S=100%, V=100%) sample image
        pink_mask = imMan_battleground_sample.apply_color_mask((159, 160), (255, 255), (255, 255))
        pink_gs = cv2.split(pink_mask)[0]
        pink_thresh = cv2.threshold(pink_gs, 0, 255, cv2.THRESH_BINARY)[1]
        cnts_pink = cv2.findContours(pink_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        contours_pink = imutils.grab_contours(cnts_pink)

        for contour in range(len(contours_pink)):
            contour_center = sd.get_contour_center(contours_pink[contour])
            self.battleground_positions.append((contour_center, self.battleground_names[contour], self.battleground_regions[contour]))

        # Transform battleground_positions coordinates (battleground_positions[0])
        bg_pos = []
        for pos in self.battleground_positions:
            bg_pos.append(pos[0])
        bg_pos_mat = np.float32([bg_pos])
        new_bg_pos_mat = cv2.transform(bg_pos_mat, transform_mat)
        
        for p in new_bg_pos_mat[0]:
            cv2.circle(self.img, (int(round(p[0])), int(round(p[1]))), 10, (255,0,0), thickness=-1)
        # for p in bg_pos_mat[0]:
        #     cv2.circle(self.img, (int(round(p[0])), int(round(p[1]))), 10, (0,0,255), thickness=-1)

        new_bg_pos = []
        for i in range(len(self.battleground_positions)):
            new_bg_pos.append(((int(round(new_bg_pos_mat[0][i][0])), int(round(new_bg_pos_mat[0][i][1]))), self.battleground_positions[i][1], self.battleground_positions[i][2]))
        self.battleground_positions = new_bg_pos

        # DEBUG: draw battleground positions after translation in pink
        # for pos in self.battleground_positions:
        #     cv2.circle(self.img, pos[0], 60, (255, 0, 255), thickness=5)

        '''
        *************************************************
        Find positions of all non-battlegrounds
        ***************************************************************
        '''
        # Get non-battleground positions from cyan-marked (H=168.9 deg, S=100%, V=100%) sample image (created with test2.jpg)
        cyan_mask = imMan_nonbattleground_sample.apply_color_mask((84, 85), (255, 255), (255, 255))
        cyan_gs = cv2.split(cyan_mask)[0]
        cyan_thresh = cv2.threshold(cyan_gs, 0, 255, cv2.THRESH_BINARY)[1]
        cnts_cyan = cv2.findContours(cyan_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        contours_cyan = imutils.grab_contours(cnts_cyan)

        for contour in range(len(contours_cyan)):
            contour_center = sd.get_contour_center(contours_cyan[contour])
            self.nonbattleground_positions.append((contour_center, self.nonbattleground_names[contour], self.nonbattleground_regions[contour]))

        # Transform nonbattleground_positions (nonbattleground_positions[0])
        nbg_pos = []
        for pos in self.nonbattleground_positions:
            nbg_pos.append(pos[0])
        nbg_pos_mat = np.float32([nbg_pos])
        new_nbg_pos_mat = cv2.transform(nbg_pos_mat, transform_mat)
        
        for p in new_nbg_pos_mat[0]:
            cv2.circle(self.img, (int(round(p[0])), int(round(p[1]))), 10, (0,255,0), thickness=-1)
        # for p in nbg_pos_mat[0]:
        #     cv2.circle(self.img, (int(round(p[0])), int(round(p[1]))), 10, (0,0,255), thickness=-1)

        new_nbg_pos = []
        for i in range(len(self.nonbattleground_positions)):
            new_nbg_pos.append(((int(round(new_nbg_pos_mat[0][i][0])), int(round(new_nbg_pos_mat[0][i][1]))), self.nonbattleground_positions[i][1], self.nonbattleground_positions[i][2]))
        self.nonbattleground_positions = new_nbg_pos

        # DEBUG: draw nonbattleground positions after translation in green
        # for pos in self.nonbattleground_positions:
        #     cv2.circle(self.img, pos[0], 60, (0, 255, 0), thickness=5)

        '''
        *************************************************
        Find positions of all control tokens 
        ***************************************************************
        '''
        red_masked = imMan.apply_color_mask(self.RED_MASK_HUE_RANGE, self.RED_MASK_SAT_RANGE, self.RED_MASK_VAL_RANGE)
        blue_masked = imMan.apply_color_mask(self.BLUE_MASK_HUE_RANGE, self.BLUE_MASK_SAT_RANGE, self.BLUE_MASK_VAL_RANGE)

        # Convert to grayscale by keeping only the (V)alue channel
        red_gs = cv2.split(red_masked)[0]
        blue_gs = cv2.split(blue_masked)[0]

        # Apply thresholding with mask result
        red_thresh = cv2.threshold(red_gs, 0, 255, cv2.THRESH_BINARY)[1]
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

        '''
        *************************************************
        Assign each controlled position to a country
        ***************************************************************
        '''
        VERTICAL_SHIFT = 38
        HORIZONTAL_SHIFT = 42
        vertical_shift_transformed = VERTICAL_SHIFT
        horizontal_shift_transformed = HORIZONTAL_SHIFT
        for pos in self.ussr_controlled_positions:
            self.ussr_controlled_countries.append(Country(pos, self.battleground_positions, self.nonbattleground_positions, True, vertical_shift_transformed, horizontal_shift_transformed))

            if not self.ussr_controlled_countries[-1].matched:
                cv2.circle(self.img, (pos[0], pos[1]), 70, (0,0,255), thickness=10)
                continue
            if self.ussr_controlled_countries[-1].battleground:
                cv2.circle(self.img, (pos[0] + horizontal_shift_transformed, pos[1] - vertical_shift_transformed), 10, (255,0,0), thickness=-1)
                cv2.circle(self.img, (pos[0] - horizontal_shift_transformed, pos[1] - vertical_shift_transformed), 10, (255,255,0), thickness=-1)
                cv2.circle(self.img, self.ussr_controlled_countries[-1].matched[0], 10, (255,0,255), thickness=-1)
                cv2.circle(self.img, (pos[0], pos[1]), 70, (255,0,0), thickness=10)
            else:
                cv2.circle(self.img, self.ussr_controlled_countries[-1].matched[0], 10, (255,0,255), thickness=-1)
                cv2.circle(self.img, (pos[0], pos[1]), 70, (0,255,0), thickness=10)

        for pos in self.us_controlled_positions:
            self.us_controlled_countries.append(Country(pos, self.battleground_positions, self.nonbattleground_positions, False, vertical_shift_transformed, horizontal_shift_transformed))

            if not self.us_controlled_countries[-1].matched:
                cv2.circle(self.img, (pos[0], pos[1]), 70, (0,0,255), thickness=10)
                continue
            if self.us_controlled_countries[-1].battleground:
                cv2.circle(self.img, (pos[0] + horizontal_shift_transformed, pos[1] - vertical_shift_transformed), 10, (255,0,0), thickness=-1)
                cv2.circle(self.img, (pos[0] - horizontal_shift_transformed, pos[1] - vertical_shift_transformed), 10, (255,255,0), thickness=-1)
                cv2.circle(self.img, self.us_controlled_countries[-1].matched[0], 10, (255,0,255), thickness=-1)
                cv2.circle(self.img, (pos[0], pos[1]), 70, (255,0,0), thickness=10)
            else:
                cv2.circle(self.img, self.us_controlled_countries[-1].matched[0], 10, (255,0,255), thickness=-1)
                cv2.circle(self.img, (pos[0], pos[1]), 70, (0,255,0), thickness=10)

        

    def get_ussr_controlled(self, region=None):
        if region:
            return list(filter(lambda c : c.region == region, self.ussr_controlled_countries))
        return self.ussr_controlled_countries

    def get_us_controlled(self, region=None):
        if region:
            return list(filter(lambda c : c.region == region, self.us_controlled_countries))
        return self.us_controlled_countries

    def get_ussr_battlegrounds(self, region=None):
        if region:
            return list(filter(lambda c : c.battleground and c.region == region, self.ussr_controlled_countries))
        return list(filter(lambda c : c.battleground, self.ussr_controlled_countries))

    def get_us_battlegrounds(self, region=None):
        if region:
            return list(filter(lambda c : c.battleground and c.region == region, self.us_controlled_countries))
        return list(filter(lambda c : c.battleground, self.us_controlled_countries))

    def get_ussr_non_battlegrounds(self, region=None):
        if region:
            return list(filter(lambda c : not c.battleground and c.region == region, self.ussr_controlled_countries))
        return list(filter(lambda c : not c.battleground, self.ussr_controlled_countries))

    def get_us_non_battlegrounds(self, region=None):
        if region:
            return list(filter(lambda c : not c.battleground and c.region == region, self.us_controlled_countries))
        return list(filter(lambda c : not c.battleground, self.us_controlled_countries))

    def check_ussr_control(self, name):
        return len(list(filter(lambda c : c.name == name, self.ussr_controlled_countries))) != 0

    def check_us_control(self, name):
        return len(list(filter(lambda c : c.name == name, self.us_controlled_countries))) != 0

    def get_region_score(self, scoring_region, num_region_battlegrounds, presence_score, domination_score, control_score):
        ussr_score = 0
        # Get presence/domination/control score
        if len(self.get_ussr_controlled(region=scoring_region)) > 0:
            # Control
            if len(self.get_ussr_battlegrounds(region=scoring_region)) == num_region_battlegrounds and len(self.get_ussr_controlled(region=scoring_region)) > len(self.get_us_controlled(region=scoring_region)):
                ussr_score += control_score
            # Domination
            elif len(self.get_ussr_battlegrounds(region=scoring_region)) > len(self.get_us_battlegrounds(region=scoring_region)) and len(self.get_ussr_non_battlegrounds(region=scoring_region)) > 0 and len(self.get_ussr_controlled(region=scoring_region)) > len(self.get_us_controlled(region=scoring_region)):
                ussr_score += domination_score
            # Presence
            else:
                ussr_score += presence_score
        # Add battleground score
        ussr_score += len(self.get_ussr_battlegrounds(region=scoring_region))
        # USSR has potential adjacency scoring in Europe with Canada
        if scoring_region == 'eu':
            if self.check_ussr_control('Canada'):
                ussr_score += 1
        # USSR has potential adjacency scoring in Central America with Cuba, Mexico and Panama
        if scoring_region == 'ca':
            if self.check_ussr_control('Cuba'):
                ussr_score += 1
            if self.check_ussr_control('Mexico'):
                ussr_score += 1
            if self.check_ussr_control('Panama'):
                ussr_score += 1

        us_score = 0
        # Get presence/domination/control score
        if len(self.get_us_controlled(region=scoring_region)) > 0:
            # Control
            if len(self.get_us_battlegrounds(region=scoring_region)) == num_region_battlegrounds and len(self.get_us_controlled(region=scoring_region)) > len(self.get_ussr_controlled(region=scoring_region)):
                us_score += control_score
            # Domination
            elif len(self.get_us_battlegrounds(region=scoring_region)) > len(self.get_ussr_battlegrounds(region=scoring_region)) and len(self.get_us_non_battlegrounds(region=scoring_region)) > 0 and len(self.get_us_controlled(region=scoring_region)) > len(self.get_ussr_controlled(region=scoring_region)):
                us_score += domination_score
            # Presence
            else:
                us_score += presence_score
        # Add battleground score
        us_score += len(self.get_us_battlegrounds(region=scoring_region))
        # US has potential adjacency scoring in Europe with Finland, Poland and Romania
        if scoring_region == 'eu':
            if self.check_us_control('Finland'):
                us_score += 1
            if self.check_us_control('Poland'):
                us_score += 1
            if self.check_us_control('Romania'):
                us_score += 1
        # US has potential adjacency scoring in Asia with Afganistan and North Korea
        elif scoring_region == 'as':
            if self.check_us_control('Afganistan'):
                us_score += 1
            if self.check_us_control('North Korea'):
                us_score += 1

        return ussr_score - us_score

    def get_europe_score(self):
        return self.get_region_score('eu', 5, 3, 7, 1000)

    def get_asia_score(self):
        # TODO: implement Formosan Resolution effect
        # TODO: implement Shuttle Diplomacy effect
        return self.get_region_score('as', 6, 3, 7, 9)

    def get_sea_score(self):
        score = 0
        if self.check_ussr_control('Indonesia'):
            score += 1
        elif self.check_us_control('Indonesia'):
            score -= 1
        if self.check_ussr_control('Malaysia'):
            score += 1
        elif self.check_us_control('Malaysia'):
            score -= 1
        if self.check_ussr_control('Thailand'):
            score += 2
        elif self.check_us_control('Thailand'):
            score -= 2
        if self.check_ussr_control('Vietnam'):
            score += 1
        elif self.check_us_control('Vietnam'):
            score -= 1
        if self.check_ussr_control('Philippines'):
            score += 1
        elif self.check_us_control('Philippines'):
            score -= 1
        if self.check_ussr_control('Laos/Cambodia'):
            score += 1
        elif self.check_us_control('Laos/Cambodia'):
            score -= 1
        if self.check_ussr_control('Burma'):
            score += 1
        elif self.check_us_control('Burma'):
            score -= 1
        return score

    def get_africa_score(self):
        return self.get_region_score('af', 5, 1, 4, 6)

    def get_middle_east_score(self):
        # TODO: implement Shuttle Diplomacy effect
        return self.get_region_score('me', 6, 3, 5, 7)

    def get_central_america_score(self):
        return self.get_region_score('ca', 3, 1, 3, 5)

    def get_south_america_score(self):
        return self.get_region_score('sa', 4, 2, 5, 6)

class Country:

    CONTROL_POSITION_TOLERENCE = 80

    def __init__(self, coordinates, battleground_positions, nonbattleground_positions, ussr_controlled, down_shift, horizontal_shift):
        self.coordinates = coordinates
        self.battleground = False
        self.region = None
        self.ussr_controlled = ussr_controlled
        self.name = None
        self.matched = None

        # Check if position is a battleground
        for bg_pos in battleground_positions:
            dist_left = (((self.coordinates[0] - horizontal_shift) - bg_pos[0][0])**2 + ((self.coordinates[1] - down_shift) - bg_pos[0][1])**2)**0.5
            dist_right = (((self.coordinates[0] + horizontal_shift) - bg_pos[0][0])**2 + ((self.coordinates[1] - down_shift) - bg_pos[0][1])**2)**0.5
            if dist_right <= self.CONTROL_POSITION_TOLERENCE or dist_left <= self.CONTROL_POSITION_TOLERENCE:
                self.battleground = True
                self.matched = bg_pos
                self.name = self.matched[1]
                self.region = self.matched[2]
        # Check if position is a non-battleground
        if not self.matched:
            for nbg_pos in nonbattleground_positions:
                dist_left = (((self.coordinates[0] - horizontal_shift) - nbg_pos[0][0])**2 + ((self.coordinates[1] - down_shift) - nbg_pos[0][1])**2)**0.5
                dist_right = (((self.coordinates[0] + horizontal_shift) - nbg_pos[0][0])**2 + ((self.coordinates[1] - down_shift) - nbg_pos[0][1])**2)**0.5                
                if dist_right <= self.CONTROL_POSITION_TOLERENCE or dist_left <= self.CONTROL_POSITION_TOLERENCE:
                    self.matched = nbg_pos
                    self.name = self.matched[1]
                    self.region = self.matched[2]
        # If neither, raise error
        if not self.matched:
            # raise NameError(f"Unable to find matching reference country coordinate for {self.coordinates}")
            print(f"WARNING: Unable to find matching reference country coordinate for {self.coordinates}")
