## Set up session:
import argparse
import glob
import re
from os.path import join

import numpy as np
import cv2
from skimage.feature import hog
from skimage.exposure import rescale_intensity

# Pre-processor:
from sklearn.preprocessing import LabelEncoder
# Cross validation:
from sklearn.model_selection import StratifiedShuffleSplit
# Classifier:
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
# Evaluation metric:
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
# Hyperparameter tuning:
from sklearn.model_selection import GridSearchCV
# Model pipeline:
from sklearn.pipeline import Pipeline

## Config:
INPUT = 'car_logos'

def get_labels_and_features(INPUT, identify_ROI=True):
    # Initialize:
    brands = []
    features = []

    # Parse:
    brand_parser = re.compile("{}/(\w+)/(\w+)_(\d+).png".format(INPUT))
    for image_filename in glob.glob("{}/*/*.png".format(INPUT)):
        # Parse brand
        brand = brand_parser.match(image_filename).group(1)

        # Extract features:
        hog_descriptor = HOGDescriptor(
            orientations = 9,
            pixels_per_cell = (16, 16),
            cells_per_block = (2, 2),
            transform_sqrt = True,
            block_norm = 'L1',
            visualise = True
        )

        (feature, _) = hog_descriptor.describe(
            cv2.imread(image_filename),
            identify_ROI = identify_ROI
        )

        brands.append(brand)
        features.append(feature)

    return (brands, features)

def canny(image, sigma = 0.25):
    """ Heuristically optimal canny edge detector

        1. Remove noise using Gaussian smoothing;
        2. Suppress non-local maximum pixel to keep only thin edges;
        3. Use hysteresis thresholding to keep only strong edges.
    """
    # compute the median of the single channel pixel intensities
    median = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(image, lower, upper)

    return edged

## Feature descriptors:
class HOGDescriptor:
    """ Histogram of Oriented Gradients feature descriptors
    """

    def __init__(
        self,
        orientations = 9,
        pixels_per_cell = (8, 8),
        cells_per_block = (3, 3),
        transform_sqrt = False,
        block_norm = 'L1',
        visualise = False
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.transform_sqrt = transform_sqrt
        self.block_norm = block_norm
        self.visualise = visualise

    def describe(self, image, identify_ROI = True):
        """ Extract HOG description
        """
        # Convert to gray scale:
        image_grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Extract ROI and scale it to canonical size:
        if identify_ROI:
            image_canny = canny(image_grayscale, sigma=0.33)
            contours = cv2.findContours(
                image_canny.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )[1]
            max_contour = max(contours, key = cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(max_contour)
            image_canonical = cv2.resize(
                image_canny[y:y+h, x:x+w],
                (256, 128)
            )
        else:
            # Remove noise:
            image_blurred = cv2.GaussianBlur(image_grayscale, (5, 5), 0)
            # Detect edges using Canny detector:
            image_canny = canny(image_blurred, sigma=0.33)

            image_canonical = cv2.resize(
                image_canny,
                (256, 128)
            )

        # cv2.imshow('Canonical', image_canonical)
        # cv2.waitKey(0)

        # Extract HOG description:
        features_hog, image_hog = hog(
            image_canonical,
            orientations = self.orientations,
            pixels_per_cell = self.pixels_per_cell,
            cells_per_block = self.cells_per_block,
            transform_sqrt = self.transform_sqrt,
            block_norm = self.block_norm,
            visualise = self.visualise
        )

        image_hog = rescale_intensity(image_hog, out_range=(0, 255))

        return (features_hog, image_hog)

def get_best_model(X_train, y_train):
    """ Get best model using XGBoost classifier
    """
    # Create cross-validation sets from the training data
    cv_sets = StratifiedShuffleSplit(n_splits = 5, test_size = 0.20, random_state = 42).split(X_train, y_train)

    # Model:
    model = Pipeline(
        [
            ('clf', KNeighborsClassifier())
        ]
    )
    # Hyperparameters:
    params = {
        # 1. Maximum depth of each booster:
        "clf__n_neighbors": (3, 5, 7),
        # 2. Weight contribution:
        "clf__weights": ('uniform', 'distance')
    }

    '''
    model = Pipeline([
        ('clf', XGBClassifier(
            nthread = 4,
            subsample = 1.0,
            colsample_bylevel = 1.0
        ))
    ])

    # Hyperparameters:
    params = {
        # 1. Maximum depth of each booster:
        "clf__max_depth": (6, 8),
        # 2. Learning rate::
        "clf__learning_rate": (0.01, 0.03, 0.1, 0.3),
        # 3. Number of boosters:
        "clf__n_estimators": (80, 100, 120),
        # 4. Regularization:
        "clf__reg_lambda": (0.1, 1.0, 10.0)
    }
    '''

    # Make an fbeta_score scoring object
    scorer = make_scorer(accuracy_score)

    # Perform grid search on the classifier using 'scorer' as the scoring method
    grid_searcher = GridSearchCV(
        estimator = model,
        param_grid = params,
        scoring = scorer,
        cv = cv_sets,
        n_jobs = 2,
        verbose = 1
    )

    # Fit the grid search object to the training data and find the optimal parameters
    grid_fitted = grid_searcher.fit(X_train, y_train)

    # Get the estimator
    best_model = grid_fitted.best_estimator_

    # Get parameters & scores:
    best_parameters, score, _ = max(grid_fitted.grid_scores_, key=lambda x: x[1])

    print "[Best Parameters]:"
    print best_parameters
    print "[Best Score]:"
    print score

    return best_model

## Main:
if __name__ == '__main__':
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--training", required=True, help="Path to training dataset.")
    parser.add_argument("-t", "--testing", required=True, help="Path to testing image.")
    args = vars(parser.parse_args())

    # Parse datasets:
    (brands_train, features_train) = get_labels_and_features(args["training"], identify_ROI=True)
    (brands_test, features_test) = get_labels_and_features(args["testing"], identify_ROI=False)

    # Format:
    brand_encoder = LabelEncoder()
    X_train, X_test = np.array(features_train), np.array(features_test)
    y_train = brand_encoder.fit_transform(brands_train)
    y_test = brand_encoder.transform(brands_test)

    # Model:
    best_model = get_best_model(X_train, y_train)

    # Train:
    best_model.fit(X_train, y_train)

    # Predict:
    y_pred = best_model.predict(X_test)

    print classification_report(y_test, y_pred)
