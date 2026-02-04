import numpy as np
from pandas.core.common import random_state

# something useful for tracking algorithm's iterations
import progressbar

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode

widgets = ['Model Training: ', progressbar.Percentage(), ' ',
            progressbar.Bar(marker="-", left="[", right="]"),
            ' ', progressbar.ETA()]

def get_bootstrap_samples(X, y, nr_bootstraps, nr_samples=None):
  
  if nr_samples is None:nr_samples = X.shape[0]

  bootstrap_samples,bootstrap_sample_idx = [],[]

  for i in range(nr_bootstraps):
    indexes = np.random.choice (range (X.shape[0]),size=nr_samples,replace =True)
    bootstrap_samples.append([X[indexes], y[indexes]])
    bootstrap_sample_idx.append(indexes)

  return bootstrap_samples, bootstrap_sample_idx

class Bagging:
  def __init__(self, base_estimator, nr_estimators=10, oob_score=False):
    # number of models in the ensemble
    self.nr_estimators = nr_estimators
    
    # this can be any object that has 'fit', 'predict' methods
    self.base_estimator = base_estimator
    
    # whether we want to calculate the OOB score after training
    self.oob_score = oob_score
  
  def fit(self, X, y):
    # this method will fit a separate model (self.base_estimator)
    # on each bootstrap sample and each model should be stored
    # in order to use it in 'predict' method
  
    X = np.array(X)
    y = np.array(y)
    self.progressbar = progressbar.ProgressBar(widgets=widgets)
    bootstrap_samples, bootstrap_sample_ids = get_bootstrap_samples(X, y, nr_bootstraps=self.nr_estimators)
    self.models = []

    for i in self.progressbar(range(self.nr_estimators)):
      model = self.base_estimator()
      cur_x,cur_y = bootstrap_samples[i]
      model.fit(cur_x,cur_y) 
      self.models.append(model)

    if self.oob_score:

      # this part is for calculating the OOB score
      oob_preds, temp = [],[]

      indexes = np.arange(340)
      # YOUR CODE HERE
      for i in range(len(bootstrap_samples)):

        oob_ids = np.array(list(set(indexes) - set(bootstrap_sample_ids[i])))
        oob_preds = model.predict(X[oob_ids])
        temp.append(1 - accuracy_score(y[np.array(oob_ids)], oob_preds))

      # oob_error 
      self.oob_score = np.mean(temp)
  
  def predict(self, X):
    
    # this method will predict the labels for a given test dataset
    # get the majority 'vote' for each test instance from the ensemble
    # Hint: you may want to use 'mode' method from scipy.stats
    # YOUR CODE HERE
    X =  np.array(X)
    nr_estimators = self.nr_estimators
    y_preds = np.zeros((X.shape[0],nr_estimators))
    for i in range(nr_estimators):
      y_preds[:,i] = self.models[i].predict(X)
  
    return mode(y_preds, axis=1)[0]



class RandomForest:
  def __init__(self, nr_estimators=100, max_features=None, oob_score=False):
    # number of trees in the forest
    self.nr_estimators = nr_estimators   
    
    # this is the number of features to use for each tree
    # if not specified this should be set to sqrt(initial number of features) 
    self.max_features = max_features    
    
    self.oob_score = oob_score

  def fit(self, X, y):
    # this method will fit a separate tree
    # on each bootstrap sample and subset of features
    # each tree should be stored
    # in order to use it in 'predict' method
    self.progressbar = progressbar.ProgressBar(widgets=widgets)

    X = np.array(X)
    y = np.array(y)
    nr_features = np.shape(X)[1]
    
    bootstrap_samples, bootstrap_sample_ids = get_bootstrap_samples(X, y,
                                              self.nr_estimators)

    # If max_features is not given                                          
    if not self.max_features :
      self.max_features = np.sqrt(nr_features)

    self.trees = []
    for i in self.progressbar(range(self.nr_estimators)):
      tree = DecisionTreeClassifier()
      cur_x, cur_y = bootstrap_samples[i]
      # YOUR CODE HERE
      indexes = np.random.choice(range(nr_features),
                            size = int(self.max_features),
                            replace = False)
      # indexes hat feature unena cary                            
      tree.feature_indicies = indexes
      # fit enq anum
      tree.fit(cur_x[:,indexes],cur_y)
      self.trees.append(tree)
    

    if self.oob_score:
      # this part is for calculating the OOB score
      oob_preds = []
      temp = []

      indexes = np.arange(340)
      # YOUR CODE HERE

      for model in self.trees:
        for i in range(len(bootstrap_samples)):
          oob_ids = np.array(list(set(indexes) - set(bootstrap_sample_ids[i])))    
          oob_preds = model.predict(X[oob_ids])
          temp.append(1 - accuracy_score(y[np.array(oob_ids)], oob_preds))

      # oob_error 
      self.oob_score = np.mean(temp)
    


  def predict(self, X):
    # this method will predict the labels for a given test dataset
    # get the majority 'vote' for each test instance from the forest
    # Hint: you may want to use 'mode' method from scipy.stats
    # besides the individual trees, you will also need the feature indices
    # it was trained on 
    # YPUR CODE HERE
    X = np.array(X)
    y_preds = np.zeros((X.shape[0],self.nr_estimators))
    for i,tree in enumerate(self.trees):
      indexes = tree.feature_indicies
      y_preds[:,i] = tree.predict(X[:,indexes])
    return mode(y_preds,axis=1)[0]