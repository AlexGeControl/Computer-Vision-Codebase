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

    # RETR_LIST--return all contours:
    contours_all = cv2.findContours(
        # Destructive to input image:
        image_grayscale.copy(),
        # All contours:
        cv2.RETR_LIST,
        # Approximated contours:
        cv2.CHAIN_APPROX_SIMPLE
    )[1]

    for (index, contour) in enumerate(contours_all):
        canvas = image_original.copy()
        cv2.drawContours(
            canvas,
            [contour],
            -1,
            (0, 255, 0),
            # Border width:
            #  1. -1 for fill in;
            #  2. positive for border only
            -1
        )
        cv2.imshow(
            "All Contours @ {}".format(index + 1),
            canvas
        )
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

    for (index, contour) in enumerate(contours_external):
        canvas = image_original.copy()
        cv2.drawContours(
            canvas,
            [contour],
            -1,
            (0, 255, 0),
            # Border width:
            #  1. -1 for fill in;
            #  2. positive for border only
            -1
        )
        cv2.imshow(
            "External Contours @ {}".format(index + 1),
            canvas
        )
        cv2.waitKey(0)

    # Contours to masks:
    for (index, contour) in enumerate(contours_external):
        mask = np.zeros_like(image_grayscale, dtype=np.uint8)
        cv2.drawContours(
            mask,
            [contour],
            -1,
            255,
            -1
        )
        cv2.imshow(
            "Mask {}".format(index + 1),
            mask
        )
        cv2.waitKey(0)
        cv2.imshow(
            "Segmented {}".format(index + 1),
            cv2.bitwise_and(image_original, image_original, mask = mask)
        )
        cv2.waitKey(0)
