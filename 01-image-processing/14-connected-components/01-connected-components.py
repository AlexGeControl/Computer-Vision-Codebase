## Set up session:
import argparse

import numpy as np
import cv2
from skimage import measure

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input image."
    )
    args = vars(parser.parse_args())

    # Read and format:
    image_original = cv2.imread(args["input"])
    # image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    # """
    image_grayscale = cv2.split(
        cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    )[2]
    # """

    # Display original image:
    cv2.imshow("Input", image_original)
    cv2.waitKey(0)

    # Threshold it:
    image_thresholded = cv2.adaptiveThreshold(
        image_grayscale,
        255,
        # 1. Mean calculation method:
        cv2.ADAPTIVE_THRESH_MEAN_C,
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        # 2. Neighborhood size:
        13,
        # 3. Constant C for T tuning:
        5
    )

    cv2.imshow("Thresholded", image_thresholded)
    cv2.waitKey(0)

    # Connected-component labeling:
    labels = measure.label(image_thresholded, neighbors=8, background=0)

    # Identify each component:
    components = np.zeros_like(image_thresholded, dtype=np.uint8)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label != 0:
            component = np.zeros_like(image_thresholded, dtype=np.uint8)
            component[labels == label] = 255

            num_pixels = cv2.countNonZero(component)
            print num_pixels

            if num_pixels >= 500 and num_pixels <= 1500:
                components = cv2.add(components, component)

    cv2.imshow("Components", components)
    cv2.waitKey(0)
