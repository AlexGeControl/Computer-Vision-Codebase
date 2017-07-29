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

    # Display original image:
    cv2.imshow("Input", image_original)
    cv2.waitKey(0)

    # Gaussian filtering--2d multivariate Gaussian weighted average:
    kernel_sizes = (
        (3, 3),
        (5, 5),
        (7, 7),
        (9, 9)
    )
    for kernel_size in kernel_sizes:
        blurred = cv2.GaussianBlur(
            image_original,
            kernel_size,
            # Compute Gaussian kernel according to the given kernel size:
            sigmaX = 0,
            sigmaY = 0
        )

        cv2.imshow(
            "Gaussian Blurred ({}, {})".format(kernel_size[0], kernel_size[1]),
            blurred
        )
        cv2.waitKey(0)
