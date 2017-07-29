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

    # Display original image:
    cv2.imshow("Input", image_grayscale)
    cv2.waitKey(0)

    # Open it--Remove small blobs:
    kernel_sizes = (
        (3, 3),
        (5, 5),
        (7, 7)
    )
    for kernel_size in kernel_sizes:
        kernel = cv2.getStructuringElement(
            ## Structuring types
            # 1. rectangular
            cv2.MORPH_RECT,
            # 2. cross
            # cv2.MORPH_CROSS,
            # 3. eclipse
            # cv2.MORPH_ELLIPSE,
            kernel_size
        )
        opened = cv2.morphologyEx(
            image_grayscale,
            cv2.MORPH_OPEN,
            kernel
        )

        cv2.imshow("Opened ({}, {})".format(kernel_size[0], kernel_size[1]), opened)
        cv2.waitKey(0)
