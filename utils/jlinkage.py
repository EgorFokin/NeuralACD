from scipy.spatial.distance import pdist, squareform
import numpy as np
import random

class JLinkage:

    def __init__(self,sigma=1, num_samples=10000, threshold=0.1):
        self.sigma = sigma
        self.num_samples = num_samples
        self.threshold = threshold
        pass

    def _sample_points(self,points):
        if self.sigma == -1:
            p1, p2, p3 = np.random.choice(points.shape[0], 3, replace=False)
            return points[p1], points[p2], points[p3]


        p1 = np.random.choice(points.shape[0])
        p2 = np.random.choice(points.shape[0], p=self.sample_probs[p1])

        prob3 = self.sample_probs[p1] * self.sample_probs[p2]
        prob3 /= prob3.sum()

        p3 = np.random.choice(points.shape[0], p=prob3)

        p1, p2, p3 = np.random.choice(points.shape[0], 3, replace=False)
        return points[p1], points[p2], points[p3]


    def _calculate_distances(self, points):
        D = squareform(pdist(points, metric='sqeuclidean'))
        np.fill_diagonal(D, np.inf)
        K = np.exp(-D / self.sigma**2)
        K /= K.sum(axis=1, keepdims=True)
        self.sample_probs = K

    def _calculate_preference_sets(self, points):
        self.preference_set = np.zeros((points.shape[0],self.num_samples), dtype = bool)
        for i in range(self.num_samples):
            p1, p2, p3 = self._sample_points(points)

            # Compute the plane parameters
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) == 0:
                continue  # skip degenerate triplets

            normal = normal / np.linalg.norm(normal)
            a, b, c = normal
            d = -np.dot(normal, p1)

            # Compute distance of all points to the plane
            distances = np.abs((points @ normal) + d)
            inliers = distances < self.threshold

            self.preference_set[inliers, i] = True
    
    

    def _planes_from_clusters(self, clusters, points):
        planes = []
        for cluster in clusters:
            if len(cluster) < 70:
                continue

            centroid = np.mean(points[cluster], axis=0)
            centered = points[cluster] - centroid

            # SVD
            _, _, vh = np.linalg.svd(centered)
            normal = vh[-1]  # normal vector to the plane

            # Plane equation: n • (X - p) = 0 ⇒ n • X + d = 0
            a, b, c = normal
            d = -normal @ centroid
            planes.append((a, b, c, d))
        return planes


    def _get_j_distance(self, model1, model2):
        intersection = np.sum(np.logical_and(model1, model2))
        union = np.sum(np.logical_or(model1, model2))
        return 1 - intersection / (union+ 1e-8) 
            

    def get_best_planes(self,points):
        self._calculate_distances(points)
        self._calculate_preference_sets(points)

        dist_matrix = squareform(pdist(self.preference_set, metric='jaccard'))
        dist_matrix = np.where(np.eye(len(dist_matrix), dtype=bool), np.inf, dist_matrix)

        

        clusters = [[i] for i in range(points.shape[0])]

        while True:
            # Find the closest pair
            
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

            if dist_matrix[i, j] >= 1:
                break
            # print("-"*20)
            print(len(clusters),dist_matrix[i, j])
            # print(list(map(float,dist_matrix[i])))
            # print(list(map(float,dist_matrix[j])))
            # Merge j into i
            self.preference_set[i] = np.logical_and(self.preference_set[i], self.preference_set[j])
            self.preference_set = np.delete(self.preference_set, j, axis=0)
            dist_matrix = np.delete(dist_matrix, j, axis=0)
            dist_matrix = np.delete(dist_matrix, j, axis=1)
            clusters[i].extend(clusters[j])
            clusters.pop(j)

            # Update distances from i to others
            for k in range(len(dist_matrix)):
                if k != i:
                    dist_matrix[i, k] = dist_matrix[k, i] = self._get_j_distance(self.preference_set[i], self.preference_set[k])

            # print(list(map(float,dist_matrix[i])))
            # print(dist_matrix[i])
            

            

        best_planes = self._planes_from_clusters(clusters, points)
        print(best_planes)
        return best_planes, clusters
                        







            
        
        

        
