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

    # Split original image into its three color-channel components:
    for (channel, component) in zip(
        ("B", "G", "R"),
        cv2.split(image_original)
    ):
        cv2.imshow("Channel {}".format(channel), component)
        cv2.waitKey(0)
