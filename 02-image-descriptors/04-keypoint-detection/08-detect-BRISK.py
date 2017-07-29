## Set up session:
import argparse

import numpy as np
import cv2

from skimage.exposure import rescale_intensity

if __name__ == '__main__':
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Input image for GFTT keypoints detection.")
    args = vars(parser.parse_args())

    # Format input image:
    image_original = cv2.imread(args["image"])
    image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Input", image_grayscale)
    cv2.waitKey(0)

    # Detect difference of Gaussian corners:
    brisk = cv2.BRISK_create()

    # Detect:
    key_points = brisk.detect(image_grayscale)

    # Draw detected key points:
    image_with_simple_keypoints = np.zeros_like(image_original)
    cv2.drawKeypoints(
        image_original,
        key_points,
        image_with_simple_keypoints,
        color = (0, 255, 255)
    )
    cv2.imshow(
        "BRISK Simple Keypoints: {}".format(
            len(key_points)
        ),
        image_with_simple_keypoints
    )
    cv2.waitKey(0)
