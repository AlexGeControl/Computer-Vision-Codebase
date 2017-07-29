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
    parser.add_argument(
        "-l", "--lower",
        type=float,
        default = 90.0,
        help="Lower bound of gradient orientation."
    )
    parser.add_argument(
        "-u", "--upper",
        type=float,
        default = 180.0,
        help="Upper bound of gradient orientation."
    )
    args = vars(parser.parse_args())

    # Read and format:
    image_original = cv2.imread(args["input"])
    image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    image_smoothed = cv2.GaussianBlur(image_grayscale, (5, 5), sigmaX=0, sigmaY=0)
    # Display original image:
    cv2.imshow("Input", image_original)
    cv2.waitKey(0)

    # Gradients along X & Y:
    gradient_x = cv2.Sobel(
        image_smoothed,
        ddepth = cv2.CV_64F,
        dx = 1,
        dy = 0,
        ksize = -1
    )

    gradient_y = cv2.Sobel(
        image_smoothed,
        ddepth = cv2.CV_64F,
        dx = 0,
        dy = 1,
        ksize = -1
    )

    # Orientations:
    orientation = 180.0 / np.pi * np.arctan2(gradient_y, gradient_x)

    # Filter by orientations:
    orientation_qualified = np.where(
        (
            (args["lower"] <= orientation) &
            (orientation <= args["upper"])
        ),
        orientation,
        np.inf
    )

    # Generate mask:
    mask = np.zeros_like(image_grayscale)
    mask[~np.isinf(orientation_qualified)] = 1

    cv2.imshow(
        "Qualified",
        cv2.bitwise_and(
            image_original, image_original, mask = mask
        )
    )
    cv2.waitKey(0)
