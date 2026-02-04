import numpy as np

# Instead of using a Decision Tree with one level
# we can create another object for Decision Stump
# which will work faster since it will not compute impurity
# to decide on which feature to make a split

# after implementing this version, create a different Adaboost
# that uses decision trees with one level and check that it is 
# more inefficient compared to the below implementation.

class DecisionStump():
  def __init__(self):
    # we will use this attribute to convert the predictions
    # in case the error > 50%
    self.flip = 1
    # the feature index on which the split was made
    self.feature_index = None
    # the threshold based on which the split was made
    self.threshold = None
    # the confidence of the model (see the pseudocode from the lecture slides)
    self.alpha = None

class Adaboost():
  # this implementation supports only -1,1 label encoding
  def __init__(self, nr_estimators=5):
    # number of weak learners (Decision Stumps) to use
    self.nr_estimators = nr_estimators

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    y[y == 0] = -1
    nr_samples, nr_features = np.shape(X)

    # initialize the uniform weights for each training instance
    # YOUR CODE HERE
    w = np.full(nr_samples, (1 / nr_samples))
    
    self.models = []

    for i in range(self.nr_estimators):
        model = DecisionStump()

        min_error = 1 

        for feature_id in range(nr_features):
          cur_X = X[:, feature_id]
          unique_values = np.unique(cur_X)
          thresholds = (unique_values[1:] + unique_values[:-1]) / 2
          for threshold in thresholds:
              # setting an intial value for the flip
              flip = 1
              # setting all the predictions as 1
              prediction = np.ones(nr_samples)
              # if the feature has values less than the fixed threshold
              # then it's prediction should be manually put as -1
              prediction[cur_X < threshold] = -1

              # compute the weighted error (epsilon_t) for the resulting prediction
              # error = np.sum(w[y != prediction])
              error = w @ (prediction == y)
              
              # if the model is worse than random guessing
              # then we need to set the flip variable to -1 
              # so that we can use it later, we also modify the error
              # accordingly
              if error > 0.5:
                error = 1 - error
                flip = -1

              # if this feature and threshold were the one giving 
              # the smallest error, then we store it's info in the 'model' object
              if error < min_error:
                model.flip = flip
                model.threshold = threshold
                model.feature_index = feature_id
                min_error = error
        
        # compute alpha based on the error of the 'best' decision stump
        model.alpha = 0.5*np.log((1-min_error + 1e-10)/(min_error + 1e-10))
        # YOUR CODE HERE
        nr_samples = X.shape[0]
        prediction = np.ones(nr_samples)
        cur_X = X[:, model.feature_index]
        if model.flip == -1:  prediction[ cur_X >= model.threshold] = -1
        else:  prediction[ cur_X <  model.threshold] = -1
       
        # compute the weights and normalize them
        w *= np.exp(-model.alpha * y * prediction) 
        w /= np.sum(w)

        # store the decision stump of the current iteration for later
        self.models.append(model)

  def predict(self, X):
    X = np.array(X)
    nr_samples = np.shape(X)[0]
    y_pred = np.zeros(nr_samples)
    # y_pred = [mod.alpha * mod.pred(mod,X) for mod in self.models]
    for i in range(X.shape[0]):
      H = 0
      for model in self.models:
        p = 1 if X[i][model.feature_index] < model.threshold else -1
        if model.flip == -1: 
          p *= -1
          H += model.alpha * p
      y_pred[i] = 1 if H > 0 else 0
    
    return y_pred


from regression_methods import RegressionTree

class GradientBoostingRegressor:
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_impurity=1e-7, max_depth=4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth

        # write the square loss function as in the lectures
        def square_loss(y, y_pred): return sum((1 / 2) * (y - y_pred) ** 2)

        # write the gradient of square loss as in the lectures
        def square_loss_gradient(y, y_pred): return y_pred - y

        self.loss = square_loss
        self.loss_gradient = square_loss_gradient

    def fit(self, X, y):
        self.trees = []  # we will store the regression trees per iteration
        self.train_loss = []  # we will store the loss values per iteration

        # initialize the predictions (f(x) in the lectures)
        # with the mean values of y
        # hint: you may want to use the np.full function here
        self.mean_y = np.mean(y)
        y_pred = np.full(len(y), self.mean_y)
        for i in range(self.n_estimators):
            tree = RegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                max_depth=self.max_depth)  # this is h(x) from our lectures
            # get the loss when comparing y_pred with true y
            # and store the values in self.train_loss
            # YOUR CODE HERE
            self.train_loss.append(self.square_loss(y, y_pred))
            # get the pseudo residuals
            residuals = self.square_loss_gradient(y, y_pred)
            tree.fit(X, residuals)  # fit the tree on the residuals
            # update the predictions y_pred using the tree predictions on X
            # YOUR CODE HERE
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)  # stor the tree model

    def predict(self, X):
        # start with initial predictions as vector of
        # the mean values of y_train (self.mean_y)
        y_pred = self.mean_y
        # iterate over the regression trees and apply the same gradient updates
        # as in the fitting process, but using test instances
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


def loss_grad(y, y_pred):   return y / y_pred - (1 - y) / (1 - y_pred)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def loss_entropy(y, y_pred):    return sum(-y * np.log(y_pred) - y * np.log(1 - y_pred))

class GradientBoostingClassifier:
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_impurity=1e-7, max_depth=4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        # YOUR CODE HERE
        # Hint: You should change the loss function and its gradient
        #       you will need cross entropy loss
        #       Also, you will need to apply sigmoid function on the regression output
        #       and apply some threshold (e.g. 0.5) to get classes.

    def fit(self, X, y):
        self.trees = []

        self.mean_y = np.mean(y)
        y_pred = np.full(len(y), 0.5)
        for i in range(self.n_estimators):
            tree = RegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                max_depth=self.max_depth)

            residuals = loss_grad(y, y_pred)
            tree.fit(X, residuals)
            a = self.learning_rate * np.array(tree.predict(X))
            y_pred = y_pred + a
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], 0.5)
        for tree in self.trees:
            a = np.array(tree.predict(X))
            y_pred += self.learning_rate * a
        for i in range(len(y_pred)):
            y_pred[i] = int(sigmoid(y_pred[i]) > 0.5)
        return y_pred
