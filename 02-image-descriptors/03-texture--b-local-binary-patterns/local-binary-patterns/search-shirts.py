## Set up session:
import argparse
import glob
from os.path import join

import numpy as np
import cv2
from skimage import feature
from pyimagesearch import LocalBinaryPatterns

class LocalBinaryDescriptor(object):
    """ Descriptor for local binary patterns extraction
    """
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image, method = 'uniform', epsilon=1e-6):
        lbp_image = feature.local_binary_pattern(
            image,
            self.num_points,
            self.radius,
            method = method
        )

        (hist, _) = np.histogram(
            lbp_image.flatten(),
            bins = range(0, self.num_points + 3),
            range=(0, self.num_points + 2)
        )

        # Normalize it:
        hist = hist / (hist.sum() + epsilon)

        return hist

def chi_square_distance(one, another, epsilon=1e-8):
    """ Calculate Chi-square distance for the two histograms:
    """
    return 0.50 * np.sum((one - another) ** 2 / (one + another + epsilon))

if __name__ == '__main__':
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Path to shirt dataset.")
    parser.add_argument("-q", "--query", required=True, help="Path to query image.")
    args = vars(parser.parse_args())

    # Extract features from dataset:
    database = {}
    descriptor = LocalBinaryPatterns(24,8)
    for image_filename in glob.glob(
        join(args["dataset"], "*.jpg")
    ):
        database[image_filename] = descriptor.describe(
            cv2.cvtColor(
                cv2.imread(image_filename),
                cv2.COLOR_BGR2GRAY
            )
        )

    # Select the top-3 most similar images:
    query = descriptor.describe(
        cv2.cvtColor(
            cv2.imread(args["query"]),
            cv2.COLOR_BGR2GRAY
        )
    )
    similarity_scores = sorted(
        [
            (chi_square_distance(query, data), filename) for (filename, data) in database.items()
        ]
    )[:3]

    # Display top-3 most similar images:
    cv2.imshow(
        "Query",
        cv2.imread(args["query"])
    )
    cv2.waitKey(0)
    for (rank, (_, filename)) in enumerate(similarity_scores):
        cv2.imshow(
            "Top {}".format(rank + 1),
            cv2.imread(filename)
        )
        cv2.waitKey(0)
