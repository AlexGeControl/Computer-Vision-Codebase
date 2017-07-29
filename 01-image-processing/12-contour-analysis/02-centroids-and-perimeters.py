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

    # Display original image:
    cv2.imshow("Input", image_original)
    cv2.waitKey(0)

    # RETR_EXTERNAL--return only the external contours
    contours_external = cv2.findContours(
        # Destructive to input image:
        image_grayscale.copy(),
        # Only external:
        cv2.RETR_EXTERNAL,
        # Approximated contours:
        cv2.CHAIN_APPROX_SIMPLE
    )[1]

    canvas = image_original.copy()
    for (index, contour) in enumerate(contours_external):
        # Moments:
        M = cv2.moments(contour)
        # Centroid:
        centroid_x = int(M["m10"]/M["m00"])
        centroid_y = int(M["m01"]/M["m00"])
        cv2.circle(
            canvas,
            (centroid_x, centroid_y),
            3,
            (0, 255, 0),
            -1
        )
        # Area:
        area = cv2.contourArea(contour)
        # Arc length:
        perimeter = cv2.arcLength(contour, True)
        # Text for properties:
        cv2.putText(
            canvas,
            "#{}".format(index + 1),
            (centroid_x, centroid_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (255, 255, 255),
            2
        )
        print "#{} (Area: {}, Peri: {})".format(index + 1, area, perimeter)

        # Contour:
        cv2.drawContours(
            canvas,
            [contour],
            -1,
            (0, 255, 0),
            # Border width:
            #  1. -1 for fill in;
            #  2. positive for border only
            1
        )

    cv2.imshow(
        "Contours and Centroids",
        canvas
    )
    cv2.waitKey(0)
