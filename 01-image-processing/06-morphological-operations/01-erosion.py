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
    cv2.imshow("Input", image_grayscale)
    cv2.waitKey(0)

    # Erode it--Remove small blobs and detach two objects:
    kernel_erode = np.ones((3, 3))
    for num_iterations in xrange(4):
        eroded = cv2.erode(
            # Input image:
            image_grayscale,
            # Structuring element--default is 3-by-3:
            kernel_erode,
            # Number of iterations:
            iterations = num_iterations
        )

        cv2.imshow("Eroded {} Times".format(num_iterations), eroded)
        cv2.waitKey(0)
