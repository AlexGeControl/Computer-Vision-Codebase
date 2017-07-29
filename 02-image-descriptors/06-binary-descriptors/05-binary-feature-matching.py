## Set up session:
import argparse

from abc import ABCMeta, abstractmethod

import numpy as np
import cv2

## Detectors:
class BaseDetector:
    """ Abstract base keypoint detector
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.detector = None

    def detect(self, image):
        return self.detector.detect(image)

class FASTDetector(BaseDetector):
    """ Fast keypoint detector
        0. Detect only at one single scale
        1. Use number of continuous bits to filter corners
    """
    def __init__(self):
        self.detector = cv2.FastFeatureDetector_create()

class BRISKDetector(BaseDetector):
    """ Brisk keypoint detector
        0. Detect across multiple scale spaces
        1. Use number of continuous bits to filter corners
    """
    def __init__(self):
        self.detector = cv2.BRISK_create()

class ORBDetector(BaseDetector):
    """ ORB keypoint detector
        0. Detect across multiple scale spaces
        1. Use number of continuous bits to filter corners
        2. Rank detected corners using Harris score and only keep top K
    """
    def __init__(self):
        self.detector = cv2.ORB_create()

## Extractors:
class BaseExtractor:
    """ Abstract base keypoint extractor
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.extractor = None

    def compute(self, image, key_points):
        return self.extractor.compute(image, key_points)

class BRIEFExtractor(BaseExtractor):
    """ BRIEF keypoint descriptor
        1. Generate binary tests using 2-d multivariate Gaussian sampling
        2. No orientation compensation
    """
    def __init__(self):
        self.extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

class ORBExtractor(BaseExtractor):
    """ ORB keypoint descriptor
        1. Greedy high-variance & low correlation binary tests generation through learning
        2. Orientation compensation using intensity centroid
    """
    def __init__(self):
        self.extractor = cv2.ORB_create()

class BRISKExtractor(BaseExtractor):
    """ BRISK keypoint descriptor
        1. Hand-crafted binary tests from short pairs
        2. Orientation compensation from long pairs
    """
    def __init__(self):
        self.extractor = cv2.BRISK_create()

class FREAKExtractor(BaseExtractor):
    """ FREAK keypoint descriptor
        1. Hand-crafted binary tests based on human retina sensing pattern
        2. Orientation compensation from hand-crafted gradient sampling pattern
    """
    def __init__(self):
        self.extractor = cv2.xfeatures2d.FREAK_create()

## Matchers:
class BaseMatcher:
    """ Abstract base matcher for keypoint matcher
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.matcher = None

    def match(self, features_query, features_train, top_k=100):
        # Get raw matches:
        raw_matches = self.matcher.match(
            features_query,
            features_train
        )

        # Filter false-positives using ratio test:
        good_matches = sorted(raw_matches, key=lambda x: x.distance)

        return good_matches[:top_k]

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
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

class FLANNMatcher(BaseMatcher):
    """ Fast matcher based on KD-tree
    """
    FLANN_INDEX_LSH = 6

    def __init__(self):
        index_params = dict(
            algorithm = FLANNMatcher.FLANN_INDEX_LSH,
            table_number = 6,
            key_size = 12,
            multi_probe_level = 1
        )
        search_params = dict(
            checks = 50
        )
        self.matcher = cv2.FlannBasedMatcher(index_params,search_params)

def create_detector(name):
    """ Create keypoint detector:
    """
    if "FAST" == name:
        return FASTDetector()
    elif "BRISK" == name:
        return BRISKDetector()
    elif "ORB" == name:
        return ORBDetector()
    else:
        return None

def create_extractor(name):
    """ Create keypoint extractor
    """
    if "BRIEF" == name:
        return BRIEFExtractor()
    elif "ORB" == name:
        return ORBExtractor()
    elif "BRISK" == name:
        return BRISKExtractor()
    elif "FREAK" == name:
        return FREAKExtractor()
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
        "-d", "--detector",
        type=str,
        default="ORB", choices=("FAST", "BRISK", "ORB"),
        help="Feature detector to use."
    )
    parser.add_argument(
        "-e", "--extractor",
        type=str,
        default="FREAK", choices=("BRIEF", "ORB", "BRISK", "FREAK"),
        help="Feature extractor to use."
    )
    parser.add_argument(
        "-m", "--matcher",
        type=str,
        default="BruteForce", choices=("BruteForce", "FLANN"),
        help="Feature matcher to use."
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=100,
        help="Number of keypoints"
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
    detector = create_detector(args["detector"])
    extractor = create_extractor(args["extractor"])
    keypoints_X = detector.detect(image_X_grayscale)
    _, features_X = extractor.compute(image_X_grayscale, keypoints_X)
    keypoints_Y = detector.detect(image_Y_grayscale)
    _, features_Y = extractor.compute(image_Y_grayscale, keypoints_Y)

    # Match:
    matcher = create_matcher(args["matcher"])
    matches = matcher.knnMatch(features_X, features_Y)

    # Visualze:
    visualize_matches(image_X, image_Y, keypoints_X, keypoints_Y, matches, args["one_by_one"])
