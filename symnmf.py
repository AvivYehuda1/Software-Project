import numpy as np
import sys
import symnmf

np.random.seed(0)

def read_input(file_name):
    try:
        data = np.loadtxt(file_name, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    except Exception as e:
        print(f"An Error Has Occurred: {e}")
        sys.exit(1)

def initialize_H(W, k):
    try:
        m = np.mean(W)
        H = np.random.uniform(0, 2 * np.sqrt(m / k), (W.shape[0], k))
        return H
    except Exception as e:
        print(f"An Error Has Occurred: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 4:
        print("An Error Has Occurred: Invalid number of arguments.")
        sys.exit(1)
    
    try:
        k = int(sys.argv[1])
        if k <= 0:
            raise ValueError("k must be a positive integer.")
    except ValueError as e:
        print(f"An Error Has Occurred: {e}")
        sys.exit(1)
    
    goal = sys.argv[2]
    file_name = sys.argv[3]
    
    X = read_input(file_name)
    
    if goal == 'sym':
        A = symnmf.sym(X)
        np.savetxt(sys.stdout, A, delimiter=',', fmt='%.4f')
    elif goal == 'ddg':
        A = symnmf.sym(X)
        D = symnmf.ddg(A)
        np.savetxt(sys.stdout, D, delimiter=',', fmt='%.4f')
    elif goal == 'norm':
        A = symnmf.sym(X)
        D = symnmf.ddg(A)
        W = symnmf.norm(A, D)
        np.savetxt(sys.stdout, W, delimiter=',', fmt='%.4f')
    elif goal == 'symnmf':
        A = symnmf.sym(X)
        D = symnmf.ddg(A)
        W = symnmf.norm(A, D)
        H = initialize_H(W, k)
        max_iter = 300
        epsilon = 1e-4
        H_final = symnmf.symnmf(W, H, max_iter, epsilon)
        if H_final is not None:
            np.savetxt(sys.stdout, H_final, delimiter=',', fmt='%.4f')
        else:
            print("An Error Has Occurred")
    else:
        print("An Error Has Occurred: Invalid goal.")
        sys.exit(1)

if __name__ == "__main__":
    main()

