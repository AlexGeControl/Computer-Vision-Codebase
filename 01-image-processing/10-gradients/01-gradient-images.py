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
    image_smoothed = cv2.GaussianBlur(image_grayscale, (5, 5), sigmaX=0, sigmaY=0)
    # Display original image:
    cv2.imshow("Input", image_original)
    cv2.waitKey(0)

    # Sobel operator:
    gx = cv2.convertScaleAbs(
        cv2.Sobel(
            image_smoothed,
            ddepth = cv2.CV_64F,
            dx = 1,
            dy = 0,
            ksize = -1
        )
    )
    gy = cv2.convertScaleAbs(
        cv2.Sobel(
            image_smoothed,
            ddepth = cv2.CV_64F,
            dx = 0,
            dy = 1,
            ksize = -1
        )
    )

    # Gradient images:
    image_gradient = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)

    cv2.imshow("Sobel X", gx)
    cv2.waitKey(0)

    cv2.imshow("Sobel Y", gy)
    cv2.waitKey(0)

    cv2.imshow("Gradient Image", image_gradient)
    cv2.waitKey(0)
