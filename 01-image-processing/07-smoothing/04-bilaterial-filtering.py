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

    # Bilateral filtering
    #   1. Only neighboring pixels with similar intensity will be included in filtering;
    #   2. Details and textures will be filtered while edges will be preserved.
    kernel_params = (
        (11, 21,  7),
        (11, 41, 21),
        (11, 61, 39)
    )
    for diameter, sigma_color, sigma_space in kernel_params:
        blurred = cv2.bilateralFilter(
            image_original,
            diameter,
            sigma_color,
            sigma_space
        )

        cv2.imshow(
            "Bilateral Filtered ({}, {}, {})".format(diameter, sigma_color, sigma_space),
            blurred
        )
        cv2.waitKey(0)
