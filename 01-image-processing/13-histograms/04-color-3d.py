## Set up session:
import argparse

import numpy as np
import cv2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
    image_hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)

    # Display original image:
    cv2.imshow("Input", image_original)
    cv2.waitKey(0)

    # Set up 3D canvas:
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    H, S = image_hsv[0].flatten(), image_hsv[1].flatten()
    hist, xedges, yedges = np.histogram2d(
        H, S,
        bins = 4,
        range = [
            [0, 180],
            [0, 256]
        ]
    )

    # Construct arrays for the anchor positions of the bars.
    # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
    # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
    # with indexing='ij'.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    # Construct arrays with the dimensions for the 16 bars.
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

    plt.show()
