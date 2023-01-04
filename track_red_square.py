import cv2
import logging
import argparse
import ast
import os
import numpy as np
import utilities.red_square as red_square
import copy
import pickle
import stereo_vision.projection
import imageio

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
        inputImagesFilepathPrefix,
        outputDirectory,
        redSquareDetectorBlueDelta,
        redSquareDetectorBlueDilationSize,
        redSquareDetectorRedDelta,
        redSquareDetectorRedDilationSize,
        projectionMatrix1Filepath,
        projectionMatrix2Filepath,
        radialDistortion1Filepath,
        radialDistortion2Filepath
):
    logging.info("track_red_square.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Create a dictionary linking timestamp to image filepath
    timestamp_to_imageFilepathsList = TimestampToImageFilepathsList([1, 2], inputImagesFilepathPrefix)

    # Create a detector for the red square in the images
    red_square_detector = red_square.Detector(
        blue_delta=redSquareDetectorBlueDelta,
        blue_mask_dilation_kernel_size=redSquareDetectorBlueDilationSize,
        red_delta=redSquareDetectorRedDelta,
        red_mask_dilation_kernel_size=redSquareDetectorRedDilationSize,
        debug_directory=None
    )

    # Load the projection matrices
    P1, P2 = None, None
    with open(projectionMatrix1Filepath, 'rb') as proj1_file:
        P1 = pickle.load(proj1_file)
    with open(projectionMatrix2Filepath, 'rb') as proj2_file:
        P2 = pickle.load(proj2_file)
    # Create the stereo vision system
    stereo_system = stereo_vision.projection.StereoVisionSystem([P1, P2])

    # Load the radial distortion models
    radial_dist1, radial_dist2 = None, None
    with open(radialDistortion1Filepath, 'rb') as radial_dist1_file:
        radial_dist1 = pickle.load(radial_dist1_file)
    with open(radialDistortion2Filepath, 'rb') as radial_dist2_file:
        radial_dist2 = pickle.load(radial_dist2_file)
    radial_distortions = [radial_dist1, radial_dist2]

    # Keep a list of images to create an animated gif
    gif_images = []

    with open(os.path.join(outputDirectory, "red_square_coordinates.csv"), 'w') as coords_file:
        header = "timestamp"
        for camera_ID_ndx in range(1, 3):
            header += f",x_{str(camera_ID_ndx)},y_{str(camera_ID_ndx)}"
        header += "\n"
        coords_file.write(header)
        timestamps = list(timestamp_to_imageFilepathsList.keys())
        timestamps.sort()  # We want the images to be processed in chronological order
        for timestamp in timestamps:
            image_filepaths_list = timestamp_to_imageFilepathsList[timestamp]
            coords_file.write(timestamp)
            images = []
            for image_filepath in image_filepaths_list:
                image = cv2.imread(image_filepath)
                images.append(image)
            img_shapeHWC = images[0].shape
            mosaic_img = np.zeros((img_shapeHWC[0], len(images) * img_shapeHWC[1], img_shapeHWC[2]), dtype=np.uint8)
            undistorted_centers = []
            for image_ndx in range(len(images)):
                image = images[image_ndx]
                annotated_img = copy.deepcopy(image)
                center = red_square_detector.Detect(image)
                center_rounded = (round(center[0]), round(center[1]))
                cv2.line(annotated_img, (center_rounded[0] - 5, center_rounded[1]), (center_rounded[0] + 5, center_rounded[1]), (255, 0, 0),
                         thickness=3)
                cv2.line(annotated_img, (center_rounded[0], center_rounded[1] - 5),
                         (center_rounded[0], center_rounded[1] + 5), (255, 0, 0),
                         thickness=3)
                # Undistort the coordinates
                undistorted_center = radial_distortions[image_ndx].UndistortPoint(center)
                undistorted_centers.append(undistorted_center)
                mosaic_img[:, image_ndx * img_shapeHWC[1]: (image_ndx + 1) * img_shapeHWC[1], :] = annotated_img
                coords_file.write(f",{center[0]},{center[1]}")
            coords_file.write("\n")
            # Solve the 3D coordinates
            XYZ = stereo_system.SolveXYZ(undistorted_centers)
            uv = undistorted_centers[0]
            cv2.putText(mosaic_img, "({:.1f}, {:.1f}, {:.1f})".format(XYZ[0], XYZ[1], XYZ[2]),
                        (round(uv[0]) + 10, round(uv[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
            mosaic_img_filepath = os.path.join(outputDirectory, 'stereo_' + timestamp + '.png')
            cv2.imwrite(mosaic_img_filepath, mosaic_img)

            # imageio expects RGB images
            gif_images.append(cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB))
    logging.info("Creating an animated gif...")
    animated_gif_filepath = os.path.join(outputDirectory, "animation.gif")
    imageio.mimsave(animated_gif_filepath, gif_images)

def TimestampToImageFilepathsList(camera_ID_list, images_filepath_prefix):
    extensions = ['.PNG']
    images_directory = os.path.dirname(images_filepath_prefix)
    timestamp_to_imageFilepathsList = {}
    filepaths_in_directory = [os.path.join(images_directory, f) for f in os.listdir(images_directory) \
                              if os.path.isfile(os.path.join(images_directory, f))]

    image_filepaths = [filepath for filepath in filepaths_in_directory \
                       if filepath.upper()[-4:] in extensions]

    first_camera_image_filepaths = [filepath for filepath in image_filepaths \
                                    if filepath.startswith(images_filepath_prefix + str(camera_ID_list[0]) + '_')]
    first_camera_ID = camera_ID_list[0]
    first_camera_prefix = images_filepath_prefix + str(first_camera_ID) + '_'

    for filepath in first_camera_image_filepaths:
        timestamp = filepath[len(first_camera_prefix): -4]
        image_filepaths_list = [filepath]
        for other_camera_ID_ndx in range(1, len(camera_ID_list)):
            other_camera_ID = camera_ID_list[other_camera_ID_ndx]
            other_camera_filepath = images_filepath_prefix + str(other_camera_ID) + '_' + timestamp + '.png'
            if not os.path.exists(other_camera_filepath):
                raise FileNotFoundError(f"TimestampToImageFilepathsList(): Could not find file '{other_camera_filepath}'")
            image_filepaths_list.append(other_camera_filepath)
        timestamp_to_imageFilepathsList[timestamp] = image_filepaths_list
    return timestamp_to_imageFilepathsList

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputImagesFilepathPrefix', help="The filepath prefix of the input images. Default: 'red_square_images/camera_'",
                        default='red_square_images/camera_')
    parser.add_argument('--outputDirectory', help="The output directory. Defaut: './output_track_red_square'",
                        default='./output_track_red_square')
    parser.add_argument('--redSquareDetectorBlueDelta', help="For the red square detector, the blue delta. Default: 15",
                        type=int, default=15)
    parser.add_argument('--redSquareDetectorBlueDilationSize',
                        help="For the red square detector, the blue dilation size. Default: 45", type=int, default=45)
    parser.add_argument('--redSquareDetectorRedDelta', help="For the red square detector, the red delta. Default: 70",
                        type=int, default=70)
    parser.add_argument('--redSquareDetectorRedDilationSize',
                        help="For the red square detector, the red dilation size. Default: 13", type=int, default=13)
    parser.add_argument('--projectionMatrix1Filepath',
                        help="Filepath of the projection matrix 1. Defualt: './output_calibrate_stereo/camera1.projmtx'",
                        default="./output_calibrate_stereo/camera1.projmtx")
    parser.add_argument('--projectionMatrix2Filepath',
                        help="Filepath of the projection matrix 2. Default: './output_calibrate_stereo/camera2.projmtx'",
                        default="./output_calibrate_stereo/camera2.projmtx")
    parser.add_argument('--radialDistortion1Filepath',
                        help="The filepath for the radial distortion compensation model for camera 1. Default: './radial_distortion/calibration_left.pkl'",
                        default='./radial_distortion/calibration_left.pkl')
    parser.add_argument('--radialDistortion2Filepath',
                        help="The filepath for the radial distortion compensation model for camera 2. Default: './radial_distortion/calibration_right.pkl'",
                        default='./radial_distortion/calibration_right.pkl')
    args = parser.parse_args()

    main(
        args.inputImagesFilepathPrefix,
        args.outputDirectory,
        args.redSquareDetectorBlueDelta,
        args.redSquareDetectorBlueDilationSize,
        args.redSquareDetectorRedDelta,
        args.redSquareDetectorRedDilationSize,
        args.projectionMatrix1Filepath,
        args.projectionMatrix2Filepath,
        args.radialDistortion1Filepath,
        args.radialDistortion2Filepath
    )