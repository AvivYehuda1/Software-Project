import numpy as np
import sys
import symnmf
from sklearn.metrics import silhouette_score
from kmeans import k_means_clustering

np.random.seed(0)

def read_input(file_name):
    vectors = []
    with open(file_name, 'r') as file:
        for line in file:
            values = line.split(',')
            vector = list(map(float, values))
            vectors.append(vector)
    return np.array(vectors)

def initialize_H(W, k):
    m = np.mean(W)
    H = np.random.uniform(0, 2 * np.sqrt(m / k), (W.shape[0], k))
    return H

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 analysis.py <k> <file_name>")
        sys.exit(1)

    k = int(sys.argv[1])
    file_name = sys.argv[2]

    X = read_input(file_name)

    # Perform custom kmeans clustering
    max_iterations = 300  
    _, centroids = k_means_clustering(k, X.shape[0], X.shape[1], max_iterations, file_name)
    kmeans_labels = [np.argmin([np.linalg.norm(x - c) for c in centroids]) for x in X]
    kmeans_score = silhouette_score(X, kmeans_labels)

    # Perform Symnmf
    A = symnmf.sym(X)
    D = symnmf.ddg(A)
    W = symnmf.norm(A, D)
    H = initialize_H(W, k)
    max_iter = 300
    epsilon = 1e-4
    H_final = symnmf.symnmf(W, H, max_iter, epsilon)
    
    if H_final is not None:
        H_labels = np.argmax(H_final, axis=1)
        unique_labels = np.unique(H_labels)
        if len(unique_labels) > 1:
            symnmf_score = silhouette_score(X, H_labels)
            print(f"nmf: {symnmf_score:.4f}")
        else:
            print("nmf: N/A (single cluster)")
    else:
        print("nmf: N/A (H_final is None)")
    
    print(f"kmeans: {kmeans_score:.4f}")

if __name__ == "__main__":
    main()



