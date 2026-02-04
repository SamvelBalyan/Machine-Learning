import numpy as np
def calculate_entropy(y):
  value, counts = np.unique(y,return_counts=True)
  probs = counts/len(y)
  entropy = 0
  for i in probs:
    entropy-= i*np.log2(i)
  return entropy


def calculate_gini(y):
  value, counts = np.unique(y,return_counts=True)
  probs = counts/len(y)
  gini = 1
  for i in probs:
    gini -= i*i
  return gini


class DecisionNode:
   
    def __init__(self, feature_id=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_id = feature_id
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class DTClassifier:
  
    def __init__(self, impurity='entropy', min_samples_split=2,
                 min_impurity=1e-7, max_depth=float("inf")):
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.impurity = self.impurity_function(impurity)
        self.root = None

    def impurity_function(self, impurity_name):
        impurity_functions = {'gini': calculate_gini,
                              'entropy': calculate_entropy}
        return impurity_functions[impurity_name]

    def calculate_purity_gain(self, y, y1, y2):
        
        # YOUR CODE HERE
        E_parent = self.impurity(y)
        E_left = self.impurity(y1)
        E_right = self.impurity(y2)
        pure_gain = E_parent - (len(y1) * E_left + len(y2) * E_right) / (len(y1) + len(y2))
        
        return pure_gain

    def divide_on_feature(self, X, y, feature_id, threshold):
        
        if isinstance(threshold, int) or isinstance(threshold, float):
            true_indices = X.T[feature_id] >= threshold
        else:
            true_indices = X.T[feature_id] == threshold
        
        X_1, y_1 = X[true_indices], y[true_indices]
        X_2, y_2 = X[~(true_indices)], y[~(true_indices)]
        
        return X_1, y_1, X_2, y_2

    def majority_vote(self, y):
        
        label = np.unique(y)[np.argmax(np.unique(y,return_counts = True)[1])]
        return label

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, current_depth=0):

        largest_purity_gain = 0  # initial small value for purity gain
        nr_samples, nr_features = np.shape(X)
        # print('Tree growing')
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
                    # print('Stex divid enq anum')
                    X1, y1, X2, y2 = self.divide_on_feature(X, y, feature_id, threshold)

                    # checking if we have samples in each subtree
                    if len(X1) > 0 and len(X2) > 0:
                        # calculate purity gain for the split
                        # print('hashvum purity gain')
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

        # If none of the above conditions are met, then we have reached the
        # leaf of the tree  and we need to store the label
        leaf_value = self.majority_vote(y)

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
