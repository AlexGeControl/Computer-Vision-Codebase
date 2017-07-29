## Set up session:
import argparse

import numpy as np
import cv2

def extract_contour_features(contour):
    """ Extract
            1. aspect ratio;
            2. extent;
            3. solidity
        from given contour
    """
    # Shape area:
    area = cv2.contourArea(contour)
    # Bounding box:
    (x, y, w, h) = cv2.boundingRect(contour)
    # Convex hull:
    convex_hull = cv2.convexHull(contour)

    # Aspect ratio:
    aspect_ratio = float(w) / h
    # Extent:
    extent = area / (w * h)
    # Solidity:
    solidity = area / cv2.contourArea(convex_hull)

    return (aspect_ratio, extent, solidity)

def predict_token(contour_features):
    """ Predict token based on its contour features
    """
    (aspect_ratio, extent, solidity) = contour_features

    if aspect_ratio >= 0.98 and aspect_ratio <= 1.02:
        return 'SQUARE'
    elif aspect_ratio >= 2.67:
        return 'RECT'
    elif extent < 0.65:
        return 'L-SHAPE'
    else:
        return 'Z-SHAPE'

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

    # Thresholding:
    image_thresholded = cv2.adaptiveThreshold(
        image_grayscale,
        255,
        # 1. Mean calculation method:
        cv2.ADAPTIVE_THRESH_MEAN_C,
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        # 2. Neighborhood size:
        13,
        # 3. Constant C for T tuning:
        5
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15)
    )
    image_smoothed = cv2.morphologyEx(
        image_thresholded,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Retrive all contours:
    contours = cv2.findContours(
        image_smoothed.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )[1]
    canvas = image_original.copy()
    for (index, contour) in enumerate(contours):
        # Identify centroid:
        M = cv2.moments(contour)
        centroid_x = int(M["m10"]/M["m00"])
        centroid_y = int(M["m01"]/M["m00"])

        # Extract contour features:
        contour_features = extract_contour_features(contour)
        print "#{}: ({}, {}, {})".format(
            index + 1,
            *contour_features
        )

        # Draw contour:
        cv2.drawContours(
            canvas,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        # Label it:
        cv2.putText(
            canvas,
            "{}".format(predict_token(contour_features)),
            (centroid_x - 50, centroid_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.618,
            (240, 0, 159),
            2
        )

    cv2.imshow(
        "Contour Analysis for Tetris",
        canvas
    )
    cv2.waitKey(0)
