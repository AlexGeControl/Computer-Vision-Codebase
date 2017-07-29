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

    # Op white hat--difference between original image and its opening:
    kernel_sizes = (
        (3, 3),
        (5, 5),
        (7, 7),
        (9, 9)
    )
    for kernel_size in kernel_sizes:
        blurred = cv2.blur(image_original, kernel_size)

        cv2.imshow(
            "Average Blurred ({}, {})".format(kernel_size[0], kernel_size[1]),
            blurred
        )
        cv2.waitKey(0)
