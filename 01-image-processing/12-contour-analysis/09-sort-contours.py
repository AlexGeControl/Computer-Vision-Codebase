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

def sort_contours(contours, method):
    def generate_method(method):
        if method == "top-to-bottom":
            return (lambda pair: pair[1][1])
        elif method == "bottom-to-top":
            return (lambda pair: -pair[1][1])
        elif method == "left-to-right":
            return (lambda pair: pair[1][0])
        elif method == "right-to-left":
            return (lambda pair: -pair[1][0])
        else:
            return (lambda pair: pair)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_centers = [(b[0] + b[2]/2, b[1] + b[3]/2) for b in bounding_boxes]

    (contours_sorted, _) = zip(
        *sorted(
            zip(contours, bounding_centers),
            key=generate_method(method)
        )
    )

    return contours_sorted

if __name__ == '__main__':
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input image."
    )
    parser.add_argument(
        "-o", "--order",
        type=str,
        required=True, choices=("top-to-bottom", "bottom-to-top", "left-to-right", "right-to-left"),
        help="Contour sorting order."
    )
    args = vars(parser.parse_args())

    # Read and format:
    image_original = cv2.imread(args["input"])

    """
    # Method 1: Direct Edge Detection
    image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    image_equalized = cv2.equalizeHist(image_grayscale)
    image_smoothed = cv2.medianBlur(image_equalized, 11)
    image_edge_acc = auto_canny(image_smoothed)
    """

    # Method 2: Accumulated Edge Detection
    image_edge = np.zeros(image_original.shape[:2], dtype=np.uint8)
    for image_channel in cv2.split(image_original):
        image_smoothed = cv2.medianBlur(image_channel, 15)
        image_edge_channel = auto_canny(image_smoothed)
        image_edge = cv2.bitwise_or(image_edge, image_edge_channel)

    cv2.imshow("Edges", image_edge)
    cv2.waitKey(0)

    # Detect external contours:
    contours = sort_contours(
        cv2.findContours(
            image_edge,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )[1],
        args["order"]
    )
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
        tolerance = 0.01 * cv2.arcLength(contour, True)
        contour_denoised = cv2.approxPolyDP(
            contour,
            tolerance,
            True
        )
        print "[From]: {}, [Down To]: {}".format(
            len(contour),
            len(contour_denoised)
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
