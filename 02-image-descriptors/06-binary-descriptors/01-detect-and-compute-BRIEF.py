## Set up session:
import argparse

import numpy as np
import cv2

import pickle

if __name__ == '__main__':
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Input image for BRIEF keypoints detection.")
    args = vars(parser.parse_args())

    # Format input image:
    image_original = cv2.imread(args["image"])
    image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Input", image_grayscale)
    cv2.waitKey(0)

    # Detect difference of Gaussian corners:
    fast = cv2.FastFeatureDetector_create()

    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    key_points = fast.detect(image_grayscale,None)

    # compute the descriptors with BRIEF
    _, features = brief.compute(image_grayscale, key_points)

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
        "BRIEF Full Keypoints: {}".format(
            len(key_points)
        ),
        image_with_full_keypoints
    )
    cv2.waitKey(0)

    # Compute feature vector for each keypoint:
    print("[Feature Dimension]: {}".format(features.shape))

    # Save description:
    with open(
        "BRIEF-description-{}.pkl".format(args["image"].split('/')[1]),
        "wb"
    ) as features_pkl:
        pickle.dump(features, features_pkl)
