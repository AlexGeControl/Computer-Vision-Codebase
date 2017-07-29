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

    # Median filtering--fill center value with existing value inside its neighborhood:
    kernel_sizes = (3, 7, 11, 15)
    for kernel_size in kernel_sizes:
        blurred = cv2.medianBlur(
            image_original,
            kernel_size
        )

        cv2.imshow(
            "Median Blurred {}".format(kernel_size),
            blurred
        )
        cv2.waitKey(0)
