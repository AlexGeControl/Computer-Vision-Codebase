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
    figure = plt.figure()

    # Subplot 121: H & S
    ax = figure.add_subplot(121)
    hist_h_s = cv2.calcHist(
        [image_hsv[0], image_hsv[1]],
        [0, 1],
        None,
        [30, 30],
        [0, 180, 0, 256]
    )
    p = ax.imshow(hist_h_s, interpolation = "nearest")
    ax.set_title("Histogram for H & S")
    plt.colorbar(p)

    # Subplot 122: H & S
    ax = figure.add_subplot(122)
    hist_h_v = cv2.calcHist(
        [image_hsv[0], image_hsv[2]],
        [0, 1],
        None,
        [30, 30],
        [0, 180, 0, 256]
    )
    p = ax.imshow(hist_h_v, interpolation = "nearest")
    ax.set_title("Histogram for H & V")
    plt.colorbar(p)

    plt.show()
