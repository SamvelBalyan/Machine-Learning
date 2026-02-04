import numpy as np

class NaiveBayes:
  def __init__(self, smoothing=False):
      # initialize Laplace smoothing parameter
      self.smoothing = smoothing
    
  def fit(self, X_train, y_train):
      # use this method to learn the model
      # if you feel it is easier to calculate priors 
      # and likelihoods at the same time
      # then feel free to change this method
      self.X_train = X_train
      self.y_train = y_train
      # self.priors_likel = self.calculate_distance()
      
  def predict(self, X_test):

      ##### YOUR CODE STARTS HERE ##### 
      dists=self.calculate_distance(X_test)
      prediction=np.array([])
            
      return np.unique(self.y_train)[np.argmax(dists,axis=1)]
      ##### YOUR CODE ENDS HERE ##### 
      
  def calculate_distance(self,X_test):

      ##### YOUR CODE STARTS HERE #####  

      # This algorithm takes like 50% longer, but uses around 2 times 
      # less memory: tested with tracemalloc

      # def hash2(x,label,eps):
      #   like=1
      #   for i in range(X_train.shape[1]):
      #     like *= len(X_train[ (y_train==label) & ( X_train.T[i]==X_test[x][i])]) + eps
      #     like/=float(len(y_train[y_train==label])) + eps*len(np.unique(X_train.T[i][y_train==label]))

      #   return (len(y_train[y_train==label])*like/float(self.X_train.shape[0]))

      def cal_dist(x,label,eps):
        like=1
        current=self.X_train[self.y_train==label]

        for i in range(self.X_train.shape[1]):
          sample=current.T[i]
          like *= len(sample[sample==X_test[x][i]]) + eps
          like/=float(len(self.y_train[self.y_train==label])) + eps*len(np.unique(sample))

        return (len(self.y_train[self.y_train==label])*like/float(self.X_train.shape[0]))

      # Check smoothing
      epsilion = 1 if self.smoothing else 0

      dists=np.array([])
      for i in range(X_test.shape[0]):
        for j in np.unique(self.y_train):
          dists=np.append(dists,cal_dist(i,j,epsilion))
      dists=dists.reshape(X_test.shape[0],len(np.unique(self.y_train)))

      ##### YOUR CODE ENDS HERE #####         
      return dists
  