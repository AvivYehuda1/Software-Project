import numpy as np
import sys

def k_means_clustering(K, n, d, max_iterations, data_file):
    vectors = []
    with open(data_file, 'r') as file:
        for line in file:
            values = line.split(',')
            vector = list(map(float, values))
            vectors.append(vector)

    centroids = np.array(vectors[:K], dtype=float)
    for iteration in range(max_iterations):
        clusters = [[] for _ in range(K)]
        for point in vectors:
            distances = [np.linalg.norm(np.array(point) - centroid) for centroid in centroids]
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(point)

        new_centroids = [np.mean(cluster, axis=0) if cluster else centroid for centroid, cluster in zip(centroids, clusters)]

        if np.linalg.norm(np.array(new_centroids) - centroids) < 0.001:
            break

        centroids = np.array(new_centroids)

    return clusters, centroids

def main():
    if len(sys.argv)<5:
        print("Usage: python kmeans.py <K> <n> <d> <iter> <input_file>")
        sys.exit(1)
    if len(sys.argv)>6:
        print("Usage: python kmeans.py <K> <n> <d> <iter> <input_file>")
        sys.exit(1)
    if len(sys.argv)==6:
        K = float(sys.argv[1])
        n =float(sys.argv[2])
        d = float(sys.argv[3])
        iterr = float(sys.argv[4])
        input_file = sys.argv[5]
    if len(sys.argv)==5:
        K = float(sys.argv[1])
        n = float(sys.argv[2])
        d = float(sys.argv[3])
        iterr = 200
        input_file = sys.argv[4]



    # Validate N
    if not (n > 1 and n%1 == 0):
        print("Error: Invalid number of points!")
        sys.exit(1)

    if not  (1 < K < n and K%1 ==0):
        print("Error: Invalid number of clusters!")
        sys.exit(1)
        
    # Validate d
    if not (d%1 ==0 and d>1):
        print("Error: Invalid dimension of point!")
        sys.exit(1)

    # Validate iter
    if not (1 < iterr < 1000 and iterr%1==0):
        print("Error: Invalid maximum iteration!")
        sys.exit(1)

    try:
        clusters, centroids = k_means_clustering(int(K), int(n), int(d), int(iterr), input_file)
    except ZeroDivisionError as e:
        print("An Error Has Occurred")
        sys.exit(1)
    for centroid in centroids:
        formatted_centroid = ', '.join(['%.4f' % val for val in centroid])
        print(formatted_centroid)

if __name__ == "__main__":
    main()
