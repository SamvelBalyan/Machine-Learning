import numpy as np


class LinearRegression:

  def __init__(self, regularization=None, lam=0, learning_rate=1e-3, tol=0.05):
    self.regularization = regularization
    self.lam = lam
    self.learning_rate = learning_rate
    self.tol = tol
    self.weights = None
  
  def fit(self, X, y):
    
    X = np.array(X)
    X = np.insert(X, 0, 1, axis=1)

    if self.regularization is None:
      self.weights = np.linalg.inv(X.T @ X) @ (X.T @ y)
    elif self.regularization == 'l2':
      I = np.identity(X.shape[1])
      self.weights = np.linalg.inv(X.T @ X + self.lam * I) @ (X.T @ y)
    elif self.regularization == 'l1':
      self.weights = np.random.randn(X.shape[1])
      converged = False
      self.loss = []
      i = 0
      while (not converged):
        i += 1
        y_pred = X @ self.weights
        self.loss.append(np.mean((y-y_pred)**2))
        grad = -2 * X.T @ (y-y_pred) + self.lam * np.sign(self.weights)
        new_weights = self.weights - self.learning_rate * grad
        converged = np.linalg.norm(self.weights - new_weights) < self.tol
        self.weights = new_weights
      print(f'Converged in {i} steps')

  def predict(self, X):
    X = np.array(X)
    X = np.insert(X, 0, 1, axis=1)
    return X @ self.weights 
  

def calculate_entropy(y):
  _, counts = np.unique(y, return_counts=True)
  p = counts / np.sum(counts)
  return -np.sum(p * np.log2(p))

def calculate_gini(y):
  _, counts = np.unique(y, return_counts=True)
  p = counts / np.sum(counts)
  return 1 - np.sum(p ** 2)   
  
def impurity_function(impurity_name):
  impurity_functions = {'gini': calculate_gini,
                        'entropy': calculate_entropy}
  return impurity_functions[impurity_name]
  
def divide_on_feature(X, y, feature_id, threshold):
  true_indices = X[:, feature_id] >= threshold
  X_1, y_1 = X[true_indices], y[true_indices]
  X_2, y_2 = X[~true_indices], y[~true_indices]
  return X_1, y_1, X_2, y_2


class DecisionNode():
  def __init__(self, feature_id=None, threshold=None,
                value=None, true_branch=None, false_branch=None):                
    self.feature_id = feature_id          
    self.threshold = threshold          
    self.value = value                  
    self.true_branch = true_branch      
    self.false_branch = false_branch    

class DecisionTree:
  def __init__(self, impurity='entropy', min_samples_split=2,
   min_impurity=1e-7, max_depth=float("inf")):
    self.min_samples_split = min_samples_split
    self.min_impurity = min_impurity
    self.max_depth = max_depth
    self.impurity = impurity_function(impurity)
    self.impurity_gain = None
    self.get_leaf_value = None            
    self.root = None  # Root node in dec. tree      

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    self.root = self.build_tree(X, y)

  def build_tree(self, X, y, current_depth=0):
    largest_impurity_gain = 0

    nr_samples, nr_features = np.shape(X)

    if nr_samples >= self.min_samples_split and current_depth <= self.max_depth:
      for feature_id in range(nr_features):                      
        unique_values = np.unique(X[:, feature_id])
        thresholds = (unique_values[1:] + unique_values[:-1]) / 2
        # Iterate through all thresholds values of feature column i and
        # calculate the impurity
        for threshold in thresholds:          
          # Divide X and y depending on if the feature value of X at index feature_i
          # meets the threshold
          X1, y1, X2, y2 = divide_on_feature(X, y, feature_id, threshold)                        

          if len(X1) > 0 and len(X2) > 0:
            # Calculate impurity
            impurity_gain = self.impurity_gain(y, y1, y2)

            # If this threshold resulted in a higher information gain than previously
            # recorded save the threshold value and the feature
            # index
            if impurity_gain > largest_impurity_gain:
              largest_impurity_gain = impurity_gain
              best_feature_id = feature_id
              best_threshold = threshold
              best_X1 = X1 # X of right subtree (true)
              best_y1 = y1 # y of right subtree (true)
              best_X2 = X2 # X of left subtree (true)
              best_y2 = y2 # y of left subtree (true)

    if largest_impurity_gain > self.min_impurity:
      true_branch = self.build_tree(best_X1,
                                    best_y1,
                                    current_depth + 1)
      
      false_branch = self.build_tree(best_X2,
                                      best_y2,
                                      current_depth + 1)
      return DecisionNode(feature_id=best_feature_id,
                          threshold=best_threshold,
                          true_branch=true_branch,
                          false_branch=false_branch)

    leaf_value = self.get_leaf_value(y)

    return DecisionNode(value=leaf_value)


  def predict_value(self, x, tree=None):

    if tree is None:
        tree = self.root

    # If we have a value (i.e we're at a leaf) => return value as the prediction
    if tree.value is not None:
        return tree.value

    # Choose the feature that we will test
    feature_value = x[tree.feature_id]

    # Determine if we will follow left or right branch
    branch = tree.false_branch
    if feature_value >= tree.threshold:
      branch = tree.true_branch
    
    return self.predict_value(x, branch)

  def predict(self, X):
      """ Classify samples one by one and return the set of labels """
      X = np.array(X)
      y_pred = [self.predict_value(instance) for instance in X]
      return y_pred

class RegressionTree(DecisionTree):
  def calculate_variance_reduction(self, y, y1, y2):
    var_tot = np.var(y)
    var_1 = np.var(y1)
    var_2 = np.var(y2)
    frac_1 = len(y1) / len(y)
    frac_2 = len(y2) / len(y)

    variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

    return variance_reduction

  def mean_of_y(self, y):
    return np.mean(y)
    # value = np.mean(y)
    # return value if len(value) > 1 else value[0]

  def fit(self, X, y):
    self.impurity_gain = self.calculate_variance_reduction
    self.get_leaf_value = self.mean_of_y
    super().fit(X, y)

class DTClassifier(DecisionTree):
  def calculate_impurity_gain(self, y, y1, y2):
    size_right = len(y1) 
    size_left = len(y2)
    parent_impurity = self.impurity(y)
    imp_gain = parent_impurity - \
    (size_right * self.impurity(y1) + size_left * self.impurity(y2)) / (size_left+size_right)
    return imp_gain

  def majority_vote(self, y):
    uniques, counts = np.unique(y, return_counts=True)
    return uniques[np.argmax(counts)]
  
  def fit(self, X, y):
    self.impurity_gain = self.calculate_impurity_gain
    self.get_leaf_value = self.majority_vote
    super().fit(X, y)