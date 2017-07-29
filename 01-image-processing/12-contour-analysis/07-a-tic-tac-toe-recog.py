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

    if solidity < 0.33:
        return ''
    elif solidity < 0.67:
        return 'X'
    else:
        return 'O'

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

    # Retrive external contours:
    contours_external = cv2.findContours(
        image_grayscale.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )[1]
    canvas = image_original.copy()
    for (index, contour) in enumerate(contours_external):
        # Identify centroid:
        M = cv2.moments(contour)
        centroid_x = int(M["m10"]/M["m00"])
        centroid_y = int(M["m01"]/M["m00"])

        # Extract contour features:
        contour_features = extract_contour_features(contour)

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
            (centroid_x, centroid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (0, 0, 255),
            4
        )

    cv2.imshow(
        "Contour Analysis for Tic-Tac-Toe",
        canvas
    )
    cv2.waitKey(0)
