## Set up session:
import argparse

from abc import ABCMeta, abstractmethod

import numpy as np
import cv2

## Descriptors:
class BaseDescriptor:
    """ Abstract base descriptor for keypoint description
    """
    @abstractmethod
    def __init__(self):
        self.descriptor = None

    def detectAndCompute(self, image, epsilon=1e-9):
        """ Detect keypoints, then compute RootSIFT features for each of them
        """
        # Detect keypoints and compute features:
        key_points, features = self.descriptor.detectAndCompute(image, None)

        return (key_points, features)

class SIFTDescriptor(BaseDescriptor):
    """ SIFT feature descriptor
    """
    def __init__(self):
        self.descriptor = cv2.xfeatures2d.SIFT_create()

class RootSIFTDescriptor(BaseDescriptor):
    """ RootSIFT feature descriptor:
    """
    def __init__(self):
        self.descriptor = cv2.xfeatures2d.SIFT_create()

    def detectAndCompute(self, image, epsilon=1e-9):
        """ Detect keypoints, then compute RootSIFT features for each of them
        """
        # Detect and computer SIFT keypoints and features:
        key_points, features_SIFT = self.descriptor.detectAndCompute(image, None)

        # Map to RootSIFT
        features_normalzied = features_SIFT / (np.abs(features_SIFT).sum(axis = 1)[:, None] + epsilon)
        features_RootSIFT = np.sqrt(features_normalzied)

        return (key_points, features_RootSIFT)

class SURFDescriptor(BaseDescriptor):
    """ SURF feature descriptor
    """
    def __init__(self):
        self.descriptor = cv2.xfeatures2d.SURF_create()

## Matchers:
class BaseMatcher:
    """ Abstract base matcher for keypoint matcher
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.matcher = None

    def knnMatch(self, features_query, features_train, ratio = 0.75):
        # Get raw matches:
        raw_matches = self.matcher.knnMatch(
            features_query,
            features_train,
            k = 2
        )

        # Filter false-positives using ratio test:
        good_matches = [best for best, second_best in raw_matches if best.distance < ratio * second_best.distance]

        return good_matches

class BruteForceMatcher(BaseMatcher):
    """ Brute force feature matcher:
    """
    def __init__(self):
        self.matcher = cv2.BFMatcher()

class FLANNMatcher(BaseMatcher):
    """ Fast matcher based on KD-tree
    """
    FLANN_INDEX_KDTREE = 0

    def __init__(self):
        index_params = dict(
            algorithm = FLANNMatcher.FLANN_INDEX_KDTREE,
            trees = 4
        )
        search_params = dict(
            checks = 50
        )
        self.matcher = cv2.FlannBasedMatcher(index_params,search_params)

def create_descriptor(name):
    """ Create keypoint descriptor
    """
    if "SIFT" == name:
        return SIFTDescriptor()
    elif "RootSIFT" == name:
        return RootSIFTDescriptor()
    elif "SURF" == name:
        return SURFDescriptor()
    else:
        return None

def create_matcher(name):
    """ Create keypoint matcher
    """
    if "BruteForce" == name:
        return BruteForceMatcher()
    elif "FLANN" == name:
        return FLANNMatcher()
    else:
        return None

def visualize_matches(image_X, image_Y, keypoints_X, keypoints_Y, matches, one_by_one=False):
    """ Draw matched keypoints on the two images
    """
    # Initialize canvas:
    (H_X, W_X) = image_X.shape[:2]
    (H_Y, W_Y) = image_Y.shape[:2]
    canvas = np.zeros((max(H_X, H_Y), W_X + W_Y, 3), dtype=np.uint8)
    canvas[0:H_X, 0:W_X] = image_X
    canvas[0:H_Y, W_X:] = image_Y

    # Draw matches:
    for index, match in enumerate(matches):
        color = np.random.randint(low = 0, high = 255, size = (3, ))
        keypoint_X = (
            int(keypoints_X[match.queryIdx].pt[0]), int(keypoints_X[match.queryIdx].pt[1])
        )
        keypoint_Y  = (
            W_X + int(keypoints_Y[match.trainIdx].pt[0]), int(keypoints_Y[match.trainIdx].pt[1])
        )
        cv2.line(canvas, keypoint_X, keypoint_Y, color, 2)

        # Visualize matches one by one:
        if one_by_one:
            cv2.imshow("Matched {}".format(index + 1), canvas)
            cv2.waitKey(0)

    # Finally:
    cv2.imshow("Matched--Total {}".format(len(matches)), canvas)
    cv2.waitKey(0)

if __name__ == '__main__':
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--first",
        type=str,
        required=True,
        help="Path to first object image."
    )
    parser.add_argument(
        "-s", "--second",
        type=str,
        required=True,
        help="Path to second object image."
    )
    parser.add_argument(
        "-d", "--descriptor",
        type=str,
        default="RootSIFT", choices=("SIFT", "RootSIFT", "SURF"),
        help="Feature descriptor to use."
    )
    parser.add_argument(
        "-m", "--matcher",
        type=str,
        default="BruteForce", choices=("BruteForce", "FLANN"),
        help="Feature matcher to use."
    )
    parser.add_argument('--visualize-all', dest='one_by_one', action='store_false')
    parser.add_argument('--visualize-one-by-one', dest='one_by_one', action='store_true')
    parser.set_defaults(one_by_one=False)
    args = vars(parser.parse_args())

    # Read input images:
    image_X = cv2.imread(args["first"])
    image_X_grayscale = cv2.cvtColor(image_X, cv2.COLOR_BGR2GRAY)
    image_Y = cv2.imread(args["second"])
    image_Y_grayscale = cv2.cvtColor(image_Y, cv2.COLOR_BGR2GRAY)

    # Detect and describe:
    descriptor = create_descriptor(args["descriptor"])
    keypoints_X, features_X = descriptor.detectAndCompute(image_X)
    keypoints_Y, features_Y = descriptor.detectAndCompute(image_Y)

    # Match:
    matcher = create_matcher(args["matcher"])
    matches = matcher.knnMatch(features_X, features_Y)

    # Visualze:
    visualize_matches(image_X, image_Y, keypoints_X, keypoints_Y, matches, args["one_by_one"])
