import numpy as np


class LinearRegression:

  def __init__(self, regularization=None, lam=0, learning_rate=1e-3, tol=0.05):
    """
    This class implements linear regression models
    Params:
    --------
    regularization - None for no regularization
                    'l2' for ridge regression
                    'l1' for lasso regression

    lam - lambda parameter for regularization in case of 
        Lasso and Ridge

    learning_rate - learning rate for gradient descent algorithm, 
                    used in case of Lasso

    tol - tolerance level for weight change in gradient descent
    """
    
    self.regularization = regularization 
    self.lam = lam 
    self.learning_rate = learning_rate 
    self.tol = tol
    self.weights = None
  
  def fit(self, X, y):
    
    X = np.array(X)
    # first insert a column with all 1s in the beginning
    X  = np.insert(X,0,1,axis = 1)
    if self.regularization is None:
      self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
    elif self.regularization == 'l2': # Ridge
      self.weights = np.linalg.inv(X.T @ X+np.eye((X.T @ X).shape[0])) @ X.T @ y
    elif self.regularization == 'l1': # Lasso
      # initialize random weights, for example normally distributed (you can use the np.random.randn function)
      self.weights = np.random.randn(X.shape[1])

      converged = False
      # we can store the loss values to see how fast the algorithm converges
      self.loss = []
      # just a counter of algorithm steps
      i = 0 
      while (not converged):
        i += 1
        # calculate the predictions in case of the weights in this stage
        y_pred = X @ self.weights
        # calculate the mean squared error (loss)
        self.loss.append(np.sum((y - y_pred) ** 2))
        # calculate the gradient of the objective function with respect to w
        grad = -2*X.T @ (y-X @ self.weights)+self.lam * sum(np.sign(self.weights))
        new_weights = self.weights - self.learning_rate * grad
        # Ete hin u nor weight-eri tarberutyuny tol-ic qich a, converged sarqum enq True
        converged = np.linalg.norm(self.weights-new_weights)<=self.tol
        self.weights = new_weights
      print(f'Converged in {i} steps')

  def predict(self, X):
    X = np.array(X)
    X = np.insert(X,0,1,axis=1)
    return X @ self.weights
  
class DecisionNode:

    def __init__(self, feature_id=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_id = feature_id
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

class RegressionTree:
    def __init__(self, min_tsamples_split=2, min_impurity=1e-7, max_depth=float("inf")):
        # Minimum number of data points to perform spliting
        self.min_samples_split = min_tsamples_split

        # The minimum impurity to perform spliting
        self.min_impurity = min_impurity

        # The maximum depth to grow the tree until
        self.max_depth = max_depth

        # Root node in dec. tree
        self.root = None

    def calculate_purity_gain(self, y, y1, y2):

        par_impur = np.var(y)
        left_impur = np.var(y1)*len(y1)/len(y)
        right_impur = np.var(y2)*len(y2)/len(y)
        return par_impur - (left_impur + right_impur)

    def divide_on_feature(self, X, y, feature_id, threshold):

        if isinstance(threshold, int) or isinstance(threshold, float):
            true_indices = X.T[feature_id] >= threshold
        else:
            true_indices = X.T[feature_id] == threshold

        X_1, y_1 = X[true_indices], y[true_indices]
        X_2, y_2 = X[~(true_indices)], y[~(true_indices)]

        return X_1, y_1, X_2, y_2

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, current_depth=0):

        largest_purity_gain = 0  # initial small value for purity gain
        nr_samples, nr_features = np.shape(X)

        # checking if we have reached the pre-specified limits
        if nr_samples >= self.min_samples_split and current_depth <= self.max_depth:

            # go over the features to select the one that gives more purity
            for feature_id in range(nr_features):

                unique_values = np.unique(X[:, feature_id])

                # we iterate through all unique values of feature column and
                # calculate the impurity
                for threshold in unique_values:

                    # Divide X and y according to the condition
                    # if the feature value of X at index feature_id
                    # meets the threshold
                    X1, y1, X2, y2 = self.divide_on_feature(X, y, feature_id, threshold)

                    # checking if we have samples in each subtree
                    if len(X1) > 0 and len(X2) > 0:
                        # calculate purity gain for the split
                        purity_gain = self.calculate_purity_gain(y, y1, y2)

                        # If this threshold resulted in a higher purity gain than
                        # previously thresholds store the threshold value and the
                        # corresponding feature index
                        if purity_gain > largest_purity_gain:
                            largest_purity_gain = purity_gain
                            best_feature_id = feature_id
                            best_threshold = threshold
                            best_X1 = X1  # X of right subtree (true)
                            best_y1 = y1  # y of right subtree (true)
                            best_X2 = X2  # X of left subtree (true)
                            best_y2 = y2  # y of left subtree (true)

        # if the resulting purity gain is good enough according our
        # pre-specified amount, then we continue growing subtrees using the
        # splitted dataset, we also increase the current_depth as
        # we go down the tree
        if largest_purity_gain > self.min_impurity:
            true_branch = self.grow_tree(best_X1,
                                         best_y1,
                                         current_depth + 1)

            false_branch = self.grow_tree(best_X2,
                                          best_y2,
                                          current_depth + 1)

            return DecisionNode(feature_id=best_feature_id,
                                threshold=best_threshold,
                                true_branch=true_branch,
                                false_branch=false_branch)

        leaf_value = np.mean(y)

        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        # this is a helper function for the predict method
        # it recursively goes down the tree
        # x is one instance (row) of our test dataset

        # when we don't specify the tree, we start from the root
        if tree is None:
            tree = self.root

        # if we have reached the leaf, then we just take the value of the leaf
        # as prediction
        if tree.value is not None:
            return tree.value

        # we take the feature of the current node that we are on
        # to test whether our instance satisfies the condition
        feature_value = x[tree.feature_id]

        # determine if we will follow left (false) or right (true) branch
        # down the tree
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # continue going down the tree recursively through the chosen subtree
        # this function will finish when we reach the leaves
        return self.predict_value(x, branch)

    def predict(self, X):
        # Classify samples one by one and return the set of labels
        X = np.array(X)
        y_pred = [self.predict_value(instance) for instance in X]
        return y_pred