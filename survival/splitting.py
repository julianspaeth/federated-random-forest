from lifelines.statistics import logrank_test

import numpy as np


def _find_split(node):
    """
    Find the best split for a Node.
    :param node: Node to find best split for.
    :return: score of best split, value of best split, variable to split, left indices, right indices.
    """
    score_opt = 0
    split_val_opt = None
    lhs_idxs_opt = None
    rhs_idxs_opt = None
    split_var_opt = None
    for i in node.f_idxs:
        score, split_val, lhs_idxs, rhs_idxs = _find_best_split_for_variable(node, i)

        if score > score_opt:
            score_opt = score
            split_val_opt = split_val
            lhs_idxs_opt = lhs_idxs
            rhs_idxs_opt = rhs_idxs
            split_var_opt = i

    return score_opt, split_val_opt, split_var_opt, lhs_idxs_opt, rhs_idxs_opt


def _find_best_split_for_variable(node, var_idx):
    """
    Find best split for a variable of a Node. Best split for a variable is the split with the highest log rank
    statistics. The logrank_test function of the lifelines package is used here.
    :param node: Node
    :param var_idx: Index of variable
    :return: score, split value, left indices, right indices.
    """
    x = node.x.reset_index(drop=True)
    x_feature = x.iloc[:, var_idx].to_numpy()
    y = node.y.to_numpy()
    score, split_val, lhs_idxs, rhs_idxs = logrank_statistics(y=y,
                                                              x_feature=x_feature,
                                                              min_leaf=node.min_leaf)
    return score, split_val, lhs_idxs, rhs_idxs


def logrank_statistics(y, x_feature, min_leaf):
    """
    Compute logrank_test of liflines package.
    :param y: Labels
    :param feature: Feature index
    :param min_leaf: Minimum number of leafs for each split.
    :return: best score, best split value, left indices, right indices
    """

    score_opt = 0
    split_val_opt = None
    lhs_idxs = None
    rhs_idxs = None

    for split_val in np.unique(np.sort(x_feature, kind="mergesort")):
        feature1 = np.argwhere(x_feature <= split_val).flatten().tolist()
        feature2 = np.argwhere(x_feature > split_val).flatten().tolist()
        if len(feature1) < min_leaf or len(feature2) < min_leaf:
            continue
        durations_a = y[feature1, 1]
        event_observed_a = y[feature1, 0]
        durations_b = y[feature2, 1]
        event_observed_b = y[feature2, 0]
        score = logrank_test(durations_a, durations_b, event_observed_a, event_observed_b).test_statistic
        if score > score_opt:
            score_opt = round(score, 3)
            split_val_opt = round(split_val, 3)
            lhs_idxs = feature1
            rhs_idxs = feature2

    return score_opt, split_val_opt, lhs_idxs, rhs_idxs
