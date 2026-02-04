import numpy as np

def dist(x,y): return np.sqrt(np.sum((x-y)**2))

class KMeans:
  def __init__(self, k=2, max_iterations=500, tol=0.5):
    # number of clusters
    self.k = k
    # maximum number of iterations to perform
    # for updating the centroids
    self.max_iterations = max_iterations
    # tolerance level for centroid change after each iteration
    self.tol = tol
    # we will store the computed centroids 
    self.centroids = None

  def init_centroids(self, X):
  
    all_inds = np.arange(X.shape[0]) 
    rand_inds = np.random.choice(all_inds, self.k)
    centroids = X[rand_inds]

    return centroids

  def closest_centroid(self, X):
    # this function computes the distance (euclidean) between 
    # each point in the dataset from the centroids filling the values
    # in a distance matrix (dist_matrix) of size n x k
    # Hint: you may want to remember how we solved the warm-up exercise
    # in Programming module (Python_Numpy2 file)
    dist_matrix = np.linalg.norm(self.centroids - X[:,np.newaxis, :],axis=2)
    # after constructing the distance matrix, you should return
    # the index of minimal value per row
    # Hint: you may want to use np.argmin function
    return np.argmin(dist_matrix, axis=1)

  def update_centroids(self, X, label_ids):
    # this function updates the centroids (there are k centroids)
    # by taking the average over the values of X for each label (cluster)
    # here label_ids are the indices returned by closest_centroid function
    
    # YOUR CODE HERE
    new_centroids = np.empty(self.centroids.shape)
    for i,label in enumerate(np.unique(label_ids)):
      X_label = X[label_ids == label]
      new_centroid = X_label.mean()
      new_centroids[i] = new_centroid
    return new_centroids

  def fit(self, X):
    # this is the main method of this class
    X = np.array(X)
    # we start by random centroids from our data
    self.centroids = self.init_centroids(X)
    
    not_converged = True
    i = 1 # keeping track of the iterations
    while not_converged and (i < self.max_iterations):
      current_labels = self.closest_centroid(X)
      new_centroids = self.update_centroids(X, current_labels)

      # count the norm between new_centroids and self.centroids
      # to measure the amount of change between 
      # old cetroids and updated centroids
      norm = np.linalg.norm(self.centroids - new_centroids)
      not_converged = norm > self.tol
      self.centroids = new_centroids
      i += 1
    self.labels = current_labels
    print(f'Converged in {i} steps')


class HierarchicalClustering:
  def __init__(self, nr_clusters, diss_func, linkage='single', distance_threshold=None):
    # nr_clusters is the number of clusters to find from the data
    # if distance_treshold is None, nr_clusters should be provided
    # and if distance_threshold is provided, then we stop 
    # forming clusters when we reach the specified threshold 
    # diss_func is the dissimilarity measure to compute the 
    # dissimilarity/distance between two data points
    # linkage method should be one of the following {single, complete, average}
    # YOUR CODE HERE
    pass
  
  def fit(self, X):
    # YOUR CODE HERE
    X = np.array(X)
    while True:
      dist_matrix = []
      for cluster1 in X:
        for cluster2 in X:
          dist = cluster1.distance(cluster2, self.diss_func, self.linkage)
          dist_matrix.append(dist)

      dsit_matrix = np.array(dist_matrix)
      dist_matrix = dist_matrix.reshape(nr_samples, -1)
      dist_matrix += (np.eye(nr_samples)* np.max(dist_matrix)).astype('int')
      min_dist = np.min(dist_matrix)
      coord = np.argwhere(dist_matrix == min_dist)[0]
      clos1 = X[coord[0]]
      clos2 = X[coord[1]]

      if self.thresh is not None:
        if min_dist > self.thresh:
          break

      clos1.merge(clos2)
      X.remove(clos2)

      nr_samples -= 1
      self.clusters = X         

# class DBSCAN:
#   def __init__(self, diss_func, epsilon=0.5, min_points=5):
#     # epsilon is the maximum distance/dissimilarity between two points 
#     # to be considered as in the neighborhood of each other
#     # min_ponits is the number of points in a neighborhood for 
#     # a point to be considered as a core point (a member of a cluster). 
#     # This includes the point itself.
#     # diss_func is the dissimilarity measure to compute the 
#     # dissimilarity/distance between two data points 
#     # YOUR CODE HERE

#   def fit(self, X):
#     # noise should be labeled as "-1" cluster
#     # YOUR CODE HERE
