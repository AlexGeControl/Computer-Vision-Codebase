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
        # Rotated bounding boxes:
        rotated_box_props = cv2.minAreaRect(contour)
        rotated_box_contour = np.int0(cv2.boxPoints(rotated_box_props))
        # Draw:
        cv2.drawContours(
            canvas,
            [rotated_box_contour],
            -1,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Rotated Bounding Boxes",
        canvas
    )
    cv2.waitKey(0)
