## Set up session:
import argparse

import numpy as np
import cv2

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
    image_hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)

    # Display original image:
    cv2.imshow("Input", image_original)
    cv2.waitKey(0)

    # Visualize:
    plt.figure()
    plt.title("RGB Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Frequency of Pixels")

    for (color_channel, image_splitted) in zip(
        ("b", "g", "r"),
        cv2.split(image_hsv)
    ):
        # Grayscale histogram:
        hist = cv2.calcHist(
            [image_splitted],
            [0],
            None,
            [256],
            (0, 256)
        )
        hist *= 100.0 / hist.sum()
        plt.plot(hist, color = color_channel)

    plt.xlim([0, 256])
    plt.show()
