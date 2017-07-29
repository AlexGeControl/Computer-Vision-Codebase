## Set up session:
import argparse

import numpy as np
import cv2

if __name__ == '__main__':
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Input image for FAST keypoints detection.")
    args = vars(parser.parse_args())

    # Format input image:
    image_original = cv2.imread(args["image"])
    image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Input", image_grayscale)
    cv2.waitKey(0)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    # Detect:
    key_points = fast.detect(image_grayscale)

    # Draw detected key points:
    image_with_keypoints = np.zeros_like(image_original)
    cv2.drawKeypoints(
        image_original,
        key_points,
        image_with_keypoints,
        color = (0, 255, 255)
    )

    cv2.imshow(
        "Keypoints Detected: {}".format(len(key_points)),
        image_with_keypoints
    )
    cv2.waitKey(0)
