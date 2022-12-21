from scipy import stats
import numpy as np
import pandas as pd

from classification.splitting import gini_split


class RandomForest:
    """
    A class that implements Random Forest algorithm from scratch.

    For more information, refer to https://towardsdatascience.com/master-machine-learning-random-forest-from-scratch-with-python-3efdd51b6d7a

    Parameters:
    ----------
    num_tree: int, default=5
        The number of voting decision tree classifiers used for classification.

    subsample_size: float, default=None
        The proportion of the total training examples used to train each decision trees.

    max_depth: int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until, all leaves are the purest.

    max_features: int, float, default=None
        For each decision tree, at each split from parent node to child nodes, consider only 'max features' to find threshold split.
        If float and <1, max_features take the proportion of the features in the dataset.

    bootstrap: bool, default=True
        Bootstrap sampling of training examples, with or without replacement.

    random_state: int, default=None
        Controls the randomness of the estimator. The features are always randomly permuted at each split in each decision tree,
        and bootstrap sampling is randomly permuted.
    """

    def __init__(self, n_estimators=100, subsample_size=None, max_depth=None, max_features=None, bootstrap=True,
                 random_state=None):
        self.num_trees = n_estimators
        self.subsample_size = subsample_size
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        # Will store individually trained decision trees
        self.decision_trees = []

    def sample(self, X, y, random_state):
        """
        Helper function used for boostrap sampling.

        Parameters:
        ----------
        X: np.array, features
        y: np.array, target
        random_state: int, random bootstrap sampling

        Returns:
        -------
        sample of features: np.array, feature bootstrapped sample
        sample of target: np.array, corresponding target bootstrapped sample
        """
        n_rows, n_cols = X.shape

        # Sample with replacement
        if self.subsample_size is None:
            sample_size = n_rows
        else:
            sample_size = int(n_rows * self.subsample_size)

        np.random.seed(random_state)
        samples = np.random.choice(a=n_rows, size=sample_size, replace=self.bootstrap)

        return X[samples], y[samples]

    def fit(self, X, y):
        """
        Instantiates a trained Random Forest classifier object, with the corresponding rules stored as attributes in the nodes in each
        decision tree.

        Parameters:
        ----------
        X: np.array or pd.core.frame.DataFrame
            The set of feature variables of the training dataset

        y: np.array or pd.core.frame.DataFrame
            The target variable of the training dataset

        Returns:
        -------
        None
        """
        # Reset
        if len(self.decision_trees) > 0:
            self.decision_trees = []

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Build each tree of the forest
        num_built = 0

        while num_built < self.num_trees:

            clf = DecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.random_state
            )

            # Obtain data sample
            _X, _y = self.sample(X, y, self.random_state)
            # Train
            clf.fit(_X, _y)
            # Save the classifier
            self.decision_trees.append(clf)

            num_built += 1

            if self.random_state is not None:
                self.random_state += 1

    def predict(self, X):
        """
        Predict class for each test example in a test set.

        Parameters:
        ----------
        X: np.array or pd.core.frame.DataFrame
            The set of feature variables of the test dataset

        Returns:
        -------
        predicted_classes: np.array
            The numpy array of predict class for each test example
        """
        # Make predictions with every tree in the forest
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))

        # Reshape so we can find the most common value
        y = np.swapaxes(y, axis1=0, axis2=1)

        # Use majority voting for the final prediction (added keepdims=True)
        predicted_classes = stats.mode(y, axis=1, keepdims=True)[0].reshape(-1)

        return predicted_classes


class DecisionTree:
    """
    A decision tree classifier.

    For more information, refer to https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775

    Parameters
    ----------
    max_depth: int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until, all leaves are the purest.

    max_features: int, float, default=None
        At each split from parent node to child nodes, consider only 'max features' to find threshold split. If float and <1,
        max_features take the proportion of the features in the dataset.

    random_state: int, default=None
        Controls the randomness of the estimator. The features are always randomly permuted at each split.
        When ``max_features < n_features``, the algorithm will select ``max_features`` at random at each split,
        before finding the best split among them. But the best found split may vary across different runs,
        even if ``max_features=n_features``. That is the case, if the improvement of the criterion is identical for several splits
        and one split has to be selected at random.

    Attributes:
    ----------
    tree: <class Node>
        The root node which obtains all other sub-nodes, which are recursively stored as attributes.
    """

    def __init__(self, max_depth=None, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None

    def fit(self, X, y):
        """
        Instantiates a trained Decision Tree Classifier object, with the corresponding rules stored as attributes in the nodes.

        Parameters:
        ----------
        X: np.array or pd.core.frame.DataFrame
            The set of feature variables of the training dataset

        y: np.array or pd.core.series.Series
            The target variable of the training dataset

        Returns:
        -------
        None
        """
        # store number of classes and features of the dataset into model object
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        if self.max_features == None:
            self.max_features = self.n_features

        if isinstance(self.max_features, float) and self.max_features <= 1:
            self.max_features = int(self.max_features * self.n_features)

        # create tree for the dataset
        self.tree = self.grow_tree(X, y, self.random_state)

    def predict(self, X):
        """
        Predict class for each test example in a test set.

        Parameters:
        ----------
        X: np.array or pd.core.frame.DataFrame
            The set of feature variables of the test dataset

        Returns:
        -------
        predicted_classes: np.array
            The numpy array of predict class for each test example
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        predicted_classes = np.array([self.predict_example(inputs) for inputs in X])

        return predicted_classes

    def grow_tree(self, X, y, random_state, depth=0):
        """
        Recursive function to continuously generate nodes. At each recursion step, a parent node is formed and recursively split
        into left child node and right child node IF the maximum depth is not reached or the parent node is less pure than
        the average gini of child nodes.

        Parameters:
        ----------
        X: np.array
            Subset of all the training examples of features at the parent node.

        y: np.array
            Subset of all the training examples of targets at the parent node.

        random_state: int, default=None

        depth: int
            The number of times a branch has split.

        Returns:
        --------
        node: <class Node>
            The instantiated Node, with its corresponding attributes.
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)

        node = Node(predicted_class=predicted_class)

        if (self.max_depth is None) or (depth < self.max_depth):
            id, thr = gini_split(X, y, self.n_classes, self.n_features, self.max_features, random_state)

            if id is not None:
                if random_state is not None:
                    random_state += 1

                indices_left = X[:, id] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]

                node.feature_index = id
                node.threshold = thr
                node.left = self.grow_tree(X_left, y_left, random_state, depth + 1)
                node.right = self.grow_tree(X_right, y_right, random_state, depth + 1)

        return node

    def predict_example(self, inputs):
        """
        Generate the predicted class of a single row of test example based on the feature indices and thresholds that have been stored in all the nodes.

        Parameters:
        ----------
        inputs: An row of test examples containing the all the features that have been trained on.

        Returns:
        --------
        node.predicted_class: int
            The stored attribute - predicted_class - of the terminal node.
        """
        node = self.tree

        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class


class Node:
    """
    Node object of the decision tree. Each node may contain other node objects as attributes, as the decision tree grows.
    An exception is when the decision tree has reached the terminal node.

    Parameters:
    ----------
    predicted_class: int
        the predicted class is specified by taking the mode of the classes in the node during training. Predicted class is an
        important information to capture in the terminal node.

    Attributes:
    ----------
    feature_index: int
        The index of the feature of the fitted data where the split will occur for the node

    threshold: float
        The value split ('less than' and 'more than') for the chosen feature

    left: <class Node>
        the left child Node that will be grown that fufills the condition 'less than' threshold

    right: <class Node>
        the right child Node that will be grown that fulfils the condition 'more than' threshold
    """

    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None