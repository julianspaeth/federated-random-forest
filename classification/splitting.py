import random
import numpy as np


def gini_split(X, y, n_classes, n_features, max_features, random_state):
    """
    Obtains the optimal feature index and threshold value for the split at the parent node, which are then used to decide the split of
    training examples of features/targets into smaller subsets.

    Parameters:
    ----------
    X: np.array
        Subset of all the training examples of features at the parent node.

    y: np.array
        Subset of all the training examples of targets at the parent node.

    random_state: int, default=None

    Returns:
    -------
    best_feat_id: int, None
        The feature index considered for split at parent node.

    best_threshold: float, None
        The threshold value at the feature considered for split at parent node.
    """
    m = len(y)
    if m <= 1:
        return None, None

    num_class_parent = [np.sum(y == c) for c in range(n_classes)]
    best_gini = 1.0 - sum((n / m) ** 2 for n in num_class_parent)
    if best_gini == 0:
        return None, None

    best_feat_id, best_threshold = None, None

    random.seed(random_state)
    feat_indices = random.sample(range(n_features), max_features)

    for feat_id in feat_indices:

        sorted_column = sorted(set(X[:, feat_id]))
        threshold_values = [np.mean([a, b]) for a, b in zip(sorted_column, sorted_column[1:])]

        for threshold in threshold_values:

            left_y = y[X[:, feat_id] < threshold]
            right_y = y[X[:, feat_id] > threshold]

            num_class_left = [np.sum(left_y == c) for c in range(n_classes)]
            num_class_right = [np.sum(right_y == c) for c in range(n_classes)]

            gini_left = 1.0 - sum((n / len(left_y)) ** 2 for n in num_class_left)
            gini_right = 1.0 - sum((n / len(right_y)) ** 2 for n in num_class_right)

            gini = (len(left_y) / m) * gini_left + (len(right_y) / m) * gini_right

            print(feat_id, threshold, gini)

            if gini < best_gini:
                best_gini = gini
                best_feat_id = feat_id
                best_threshold = threshold

    print("best")
    print(best_feat_id, best_threshold, best_gini)
    return best_feat_id, best_threshold
