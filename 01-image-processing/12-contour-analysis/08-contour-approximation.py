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
    image_smoothed = cv2.GaussianBlur(image_grayscale, (7, 7), sigmaX=0, sigmaY=0)

    # Display original image:
    cv2.imshow("Input", image_original)
    cv2.waitKey(0)

    # Detect edges using auto Canny operator:
    image_edge = auto_canny(image_smoothed)

    cv2.imshow("Edges: Canny", image_edge)
    cv2.waitKey(0)

    # Detect external contours:
    contours = cv2.findContours(
        image_edge,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )[1]
    print len(contours)
    canvas = image_original.copy()
    for (index, contour) in enumerate(contours):
        # Centroid:
        M = cv2.moments(contour)
        centroid_x, centroid_y = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

        # Label it:
        cv2.putText(
            canvas,
            "#{}".format(index + 1),
            (centroid_x, centroid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.618,
            (240, 0, 159),
            2
        )

        # Approximate the contour:
        tolerance = 0.05 * cv2.arcLength(contour, True)
        contour_denoised = cv2.approxPolyDP(
            contour,
            tolerance,
            True
        )

        cv2.drawContours(
            canvas,
            [contour_denoised],
            -1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Contours", canvas)
    cv2.waitKey(0)
