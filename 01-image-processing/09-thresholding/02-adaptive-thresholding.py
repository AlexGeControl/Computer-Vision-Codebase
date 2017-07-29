## Set up session:
import argparse

import numpy as np
import cv2

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
    image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    image_smoothed = cv2.GaussianBlur(image_grayscale, (5, 5), sigmaX=0, sigmaY=0)
    # Display original image:
    cv2.imshow("Input", image_original)
    cv2.waitKey(0)

    # Otsu's method--ptimal thresholding for bi-moddal intensity histogram:
    image_binary = cv2.adaptiveThreshold(
        image_smoothed,
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
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )
    image_closed = cv2.morphologyEx(
        image_binary,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    cv2.imshow("Mask: Adaptive", image_closed)
    cv2.waitKey(0)

    image_segmented = cv2.bitwise_and(
        image_original, image_original, mask = image_closed
    )

    cv2.imshow("Segmented", image_segmented)
    cv2.waitKey(0)
