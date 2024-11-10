# **SymNMF Clustering Algorithm Software Project**

An efficient implementation of the **Symmetric Non-negative Matrix Factorization (SymNMF)** algorithm, designed for clustering data represented by symmetric, non-negative matrices. SymNMF is valuable in applications like text mining, image processing, and community detection in networks, where symmetric, non-negative data structures are commonly used.

## **Features**

- Implements Symmetric Non-negative Matrix Factorization (SymNMF) for clustering.
- Provides comparison with the K-Means clustering algorithm.
- Includes analysis tools for clustering quality and performance comparison.

## **Table of Contents**
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)

## **Installation**

Clone this repository and install any necessary dependencies:

```bash
git clone https://github.com/AvivYehuda1/SymNMF-clustering-algorithm-Software-Project.git
cd SymNMF-clustering-algorithm-Software-Project
pip install -r requirements.txt
```

## **Usage**

### SymNMF Clustering

Run SymNMF clustering with:

```bash
python symnmf.py <k> <goal> <input_file>
```

- `k`: Number of clusters (less than the data points count).
- `goal`: Operation:
  - `symnmf`: Run SymNMF clustering.
  - `sym`: Compute similarity matrix.
  - `ddg`: Compute diagonal degree matrix.
  - `norm`: Compute normalized similarity matrix.
- `input_file`: Path to input data file (format: lines of floats separated by commas, last line empty).

### K-Means Clustering

Run K-Means clustering with:

```bash
python kmeans.py <k> <input_file>
```

- `k`: Number of clusters.
- `input_file`: Path to input data file.

### Performance Analysis

To compare SymNMF and K-Means clustering performance, run:

```bash
python analysis.py <k> <input_file>
```

The script applies both methods to the dataset and reports silhouette scores to indicate clustering quality.

## **Project Structure**

- `symnmf.py`: Python interface for the SymNMF algorithm.
- `kmeans.py`: K-Means clustering implementation.
- `analysis.py`: Comparison of clustering performance for SymNMF and K-Means.
- `symnmf.c`: Core SymNMF functions in C.
- `symnmf.h`: Header file for the C implementation.
- `symnmfmodule.c`: Python C API wrapper for SymNMF functions.
- `setup.py`: Build script for compiling the C extension.
- `Makefile`: Builds the SymNMF executable.
- `README.md`: Project documentation.

## **Examples**

Run clustering analysis on a dataset:

```bash
python analysis.py 5 input_k5_d7.txt
```

This compares SymNMF and K-Means clustering on the dataset `input_k5_d7.txt` with 5 clusters and reports the silhouette score for each.
