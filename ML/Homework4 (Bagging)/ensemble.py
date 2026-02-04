import numpy as np
from scipy import stats
# from decision_tree import DTClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import progressbar

widgets = ['Model Training: ', progressbar. Percentage(),
            progressbar. Bar (marker="-", left="[", right="]"),
             ' ', progressbar.ETA()]

def get_bootstrap_samples(X, y, nr_bootstraps, nr_samples=None):
  if nr_samples is None:
    nr_samples = np.shape (X)[0]

  bootstrap_samples = []
  for i in range(nr_bootstraps):
    idx = np.random.choice (range (np.shape (X)[0]),
                            size=nr_samples,
                            replace =True)
    bootstrap_samples.append([X[idx,:], y[idx]])
  return bootstrap_samples

class Bagging:
  def __init__(self, base_estimator, nr_estimators=10):
    self.nr_estimators = nr_estimators
    self.progressbar = progressbar. ProgressBar (widgets=widgets)
    self.base_estimator = base_estimator

  def fit(self,X,y):
    X = np.array(X)
    y = np.array(y)
    bootstrap_samples = get_bootstrap_samples(X,y,
                                              nr_bootstraps = self.nr_estimators)
    self.models = []
    for i in self.progressbar(range(self.nr_estimators)):
      model = self.base_estimator()
      X_boot, y_boot = bootstrap_samples[i]
      model.fit(X_boot,y_boot)
      self.models.append(model)

  def predict(self,X):
    X =  np.array(X)
    nr_estimators = self.nr_estimators
    y_preds = np.zeros((X.shape[0],nr_estimators))
    for i in range(nr_estimators):
      y_preds[:,i] = self.models[i].predict(X)
    return stats.mode(y_preds, axis=1)[0]

class RandomForest:
  def __init__(self,nr_estimators=100,max_features=None, min_samples_split=2,min_gain=0,max_depth=float("inf")):
    self.nr_estimators = nr_estimators
    self.max_features = max_features
    self.min_samples_split = min_samples_split
    self.min_gain = min_gain
    self.max_depth = max_depth
    self.progressbar = progressbar.ProgressBar(widgets = widgets)

  def fit(self,X,y):
    X = np.array(X)
    y = np.array(y)
    nr_features = np.shape(X)[1]
    if not self.max_features:
      self.max_features - int(np.sqrt(nr_features))

    bootstrap_samples = get_bootstrap_samples(X,y,
                                              self.nr_estimators)

    self.trees = []
    for i in self.progressbar(range(self.nr_estimators)):
      tree = DecisionTreeClassifier(
                min_samples_split = self.min_samples_split,
                min_impurity = self.min_gain,
                max_depth = self.max_depth)
      X_boot,y_boot = bootstrap_samples[i]
      idx = np.random.choice(range(nr_features),
                            size=self.max_features,
                            replace = False)
      tree.feature_indicies = idx
      tree.fit(X_boot[:,idx],y_boot)
      self.trees.append(tree)
      
def predict(self,X):
  X = np.array(X)
  nr_estimators = self.nr_estimators
  y_preds = np.zeros((X.shape[0],nr_estimators))
  for i,tree in enumerate(self.trees):
    idx = tree.feature_indicies
    y_preds[:,i] = tree.predict(X[:,idx])
  return stats.mode(y_preds,axis=1)[0]







