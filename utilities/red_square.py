import cv2
import numpy as np
import os
import utilities.blob_analysis as blob_analysis

class Detector():
    def __init__(self, blue_delta=15,
                 blue_mask_dilation_kernel_size=45,
                 red_delta=50,
                 red_mask_dilation_kernel_size=13,
                 debug_directory=None):
        self.blue_delta = blue_delta
        self.blue_mask_dilation_kernel_size = blue_mask_dilation_kernel_size
        self.red_delta = red_delta
        self.red_mask_dilation_kernel_size = red_mask_dilation_kernel_size
        self.debug_directory = debug_directory
        self.blob_detector = blob_analysis.BinaryBlobDetector()

    def Detect(self, image):
        blue_minus_green_img = image[:, :, 0].astype(float) - image[:, :, 1].astype(float)
        blue_minus_red_img = image[:, :, 0].astype(float) - image[:, :, 2].astype(float)
        blue_domination_img = np.minimum(blue_minus_red_img, blue_minus_green_img)
        _, blue_domination_mask = cv2.threshold(np.clip(blue_domination_img, 0, 255).astype(np.uint8), \
                                                 self.blue_delta, 255, cv2.THRESH_BINARY)
        # Dilate and erode
        blue_dilation_erosion_kernel = np.ones((self.blue_mask_dilation_kernel_size, self.blue_mask_dilation_kernel_size), dtype=np.uint8)
        blue_domination_mask = cv2.dilate(blue_domination_mask, blue_dilation_erosion_kernel)
        blue_domination_mask = cv2.erode(blue_domination_mask, blue_dilation_erosion_kernel)

        red_minus_green_img = image[:, :, 2].astype(float) - image[:, :, 1].astype(float)
        red_minus_blue_img = -blue_minus_red_img
        red_domination_img = np.minimum(red_minus_green_img, red_minus_blue_img)
        _, red_domination_mask = cv2.threshold(np.clip(red_domination_img, 0, 255).astype(np.uint8), \
                                               self.red_delta, 255, cv2.THRESH_BINARY)
        # Dilate and erode
        red_dilation_erosion_kernel = np.ones((self.red_mask_dilation_kernel_size, self.red_mask_dilation_kernel_size), dtype=np.uint8)
        red_domination_mask = cv2.dilate(red_domination_mask, red_dilation_erosion_kernel)
        red_domination_mask = cv2.erode(red_domination_mask, red_dilation_erosion_kernel)

        # Masks intersection
        red_square_mask = np.minimum(blue_domination_mask, red_domination_mask)

        # Blob analysis
        seedPoint_boundingBox_list, annotated_img = self.blob_detector.DetectBlobs(red_square_mask)
        largest_points_list = None
        highest_number_of_pixels = 0
        for seed_point, bounding_box in seedPoint_boundingBox_list:
            points_list = blob_analysis.PointsOfBlob(red_square_mask, seed_point, bounding_box)
            if points_list is None:
                number_of_pixels = 0
            else:
                number_of_pixels = len(points_list)
                if number_of_pixels > highest_number_of_pixels:
                    highest_number_of_pixels = number_of_pixels
                    largest_points_list = points_list
        if largest_points_list is None:
            center_of_mass = (-1, -1)
        else:
            center_of_mass = blob_analysis.CenterOfMass(largest_points_list)

        if self.debug_directory is not None:
            blue_domination_img_filepath = os.path.join(self.debug_directory, "Detector_detect_blueDomination.png")
            cv2.imwrite(blue_domination_img_filepath, blue_domination_img)
            blue_domination_mask_filepath = os.path.join(self.debug_directory, "Detector_detect_blueDominationMask.png")
            cv2.imwrite(blue_domination_mask_filepath, blue_domination_mask)
            red_domination_img_filepath = os.path.join(self.debug_directory, "Detector_detect_redDomination.png")
            cv2.imwrite(red_domination_img_filepath, red_domination_img)
            red_domination_mask_filepath = os.path.join(self.debug_directory, "Detector_detect_redDominationMask.png")
            cv2.imwrite(red_domination_mask_filepath, red_domination_mask)
            red_square_mask_filepath = os.path.join(self.debug_directory, "Detector_detect_redMask.png")
            cv2.imwrite(red_square_mask_filepath, red_square_mask)

        return center_of_mass