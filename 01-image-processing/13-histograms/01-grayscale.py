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
    image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    # Display original image:
    cv2.imshow("Input", image_original)
    cv2.waitKey(0)

    # Grayscale histogram:
    hist = cv2.calcHist(
        [image_grayscale],
        [0],
        None,
        [256],
        (0, 256)
    )
    hist = 100.0 * hist / hist.sum()

    # Visualize:
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Frequency of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()
