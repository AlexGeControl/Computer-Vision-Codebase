## Set up session:
import argparse

import numpy as np
import cv2

def auto_canny(image_grayscale, sigma=0.33):
    """ Auto canny operator using heuristics from PyImageSearch
    """
    median = np.median(image_grayscale)

    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    edge = cv2.Canny(image_grayscale, lower, upper)

    return edge

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

    # Detect edges using auto Canny operator:
    edge = auto_canny(image_smoothed)

    cv2.imshow("Edges: Canny", edge)
    cv2.waitKey(0)
