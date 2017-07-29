## Set up session:
import argparse

import numpy as np
import cv2

import pickle

class RootSIFT:
    """ RootSIFT descriptor--A simple wrapper on OpenCV SIFT descriptor
    """
    def __init__(self):
        self.SIFT = cv2.xfeatures2d.SIFT_create()

    def detectAndCompute(self, image, epsilon = 1e-9):
        """ Add epsilon to prevent dividing by zero
        """
        # Detect:
        key_points, features_SIFT = self.SIFT.detectAndCompute(image, None)

        if 0 == len(key_points):
            return ([], None)

        # Map to RootSIFT:
        features_L1_normalized = features_SIFT / (np.abs(features_SIFT).sum(axis = 1)[:, None] + epsilon)
        features_RootSIFT = np.sqrt(features_L1_normalized)

        return (key_points, features_RootSIFT)


if __name__ == '__main__':
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Input image for RootSIFT keypoint description.")
    args = vars(parser.parse_args())

    # Format input image:
    image_original = cv2.imread(args["image"])
    image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Input", image_grayscale)
    cv2.waitKey(0)

    # Initialize RootSIFT descriptor:
    root_sift = RootSIFT()

    # Detect:
    key_points, features = root_sift.detectAndCompute(image_grayscale)

    # Draw detected key points:
    image_with_full_keypoints = np.zeros_like(image_original)
    cv2.drawKeypoints(
        image_original,
        key_points,
        image_with_full_keypoints,
        color = (0, 255, 255),
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imshow(
        "RootSIFT Full Keypoints: {}".format(
            len(key_points)
        ),
        image_with_full_keypoints
    )
    cv2.waitKey(0)

    # Compute feature vector for each keypoint:
    print("[Feature Dimension]: {}".format(features.shape))

    # Save description:
    with open(
        "RootSIFT-description-{}.pkl".format(args["image"].split('/')[1]),
        "wb"
    ) as features_pkl:
        pickle.dump(features, features_pkl)
