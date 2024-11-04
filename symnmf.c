#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "symnmf.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#define INITIAL_CAPACITY 10

void calculate_similarity_matrix(double **matrix, int n);
void calculate_diagonal_degree_matrix(double **matrix, int n);
void calculate_normalized_similarity_matrix(double **matrix, int n);

void sym(double *X, size_t n, size_t d, double *A) {
    size_t i, j, k;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            if (i == j) {
                A[i * n + j] = 0.0;
            } else {
                double sum = 0.0;
                for (k = 0; k < d; ++k) {
                    double diff = X[i * d + k] - X[j * d + k];
                    sum += diff * diff;
                }
                A[i * n + j] = exp(-sum / 2.0);
            }
        }
    }
}

void ddg(double *A, size_t n, double *D) {
    size_t i, j;
    for (i = 0; i < n; ++i) {
        double sum = 0.0;
        for (j = 0; j < n; ++j) {
            sum += A[i * n + j];
        }
        D[i * n + i] = sum;
    }
}

void norm(double *A, double *D, size_t n, double *W) {
    size_t i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            double denom_i = sqrt(D[i * n + i]);
            double denom_j = sqrt(D[j * n + j]);
            if (denom_i > 0 && denom_j > 0) {
                W[i * n + j] = A[i * n + j] / (denom_i * denom_j);
            } else {
                W[i * n + j] = 0;
            }
        }
    }
}

void update_H(double *H, double *W, size_t n, size_t k) {
    double *WH = (double *)malloc(n * k * sizeof(double));
    double *HHTH = (double *)malloc(n * k * sizeof(double));

    size_t i, j, l, m;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            WH[i * k + j] = 0;
            for (l = 0; l < n; l++) {
                WH[i * k + j] += W[i * n + l] * H[l * k + j];
            }
        }
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            HHTH[i * k + j] = 0;
            for (l = 0; l < k; l++) {
                for (m = 0; m < n; m++) {
                    HHTH[i * k + j] += H[i * k + l] * H[m * k + l] * H[m * k + j];
                }
            }
        }
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            if (HHTH[i * k + j] != 0) {
                H[i * k + j] *= (1 - 0.5 + 0.5 * WH[i * k + j] / HHTH[i * k + j]);
            }
        }
    }

    free(WH);
    free(HHTH);
}

int check_convergence(double *H_old, double *H_new, size_t n, size_t k, double epsilon) {
    double diff = 0;
    size_t i;
    for (i = 0; i < n * k; i++) {
        diff += (H_new[i] - H_old[i]) * (H_new[i] - H_old[i]);
    }
    return diff < epsilon;
}

void symnmf(double *W, double *H, size_t n, size_t k, size_t max_iter, double epsilon) {
    double *H_old = (double *)malloc(n * k * sizeof(double));
    size_t iter = 0;
    size_t i;
    while (iter < max_iter) {
        for (i = 0; i < n * k; i++) {
            H_old[i] = H[i];
        }
        update_H(H, W, n, k);
        if (check_convergence(H_old, H, n, k, epsilon)) {
            break;
        }
        iter++;
    }
    free(H_old);
}

void compute_similarity_matrix(double *X, double *A, size_t n, size_t d) {
    size_t i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i * n + j] = 0;
            } else {
                double sum = 0;
                for (k = 0; k < d; k++) {
                    double diff = X[i * d + k] - X[j * d + k];
                    sum += diff * diff;
                }
                A[i * n + j] = exp(-sum / 2);
            }
        }
    }
}

void compute_diagonal_degree_matrix(double *A, double *D, size_t n) {
    size_t i, j;
    for (i = 0; i < n; i++) {
        double sum = 0;
        for (j = 0; j < n; j++) {
            sum += A[i * n + j];
        }
        D[i * n + i] = sum;
    }
}

void compute_normalized_similarity_matrix(double *A, double *D, double *W, size_t n) {
    size_t i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double denom_i = sqrt(D[i * n + i]);
            double denom_j = sqrt(D[j * n + j]);
            if (denom_i > 0 && denom_j > 0) {
                W[i * n + j] = A[i * n + j] / (denom_i * denom_j);
            } else {
                W[i * n + j] = 0;
            }
        }
    }
}

char** read_file_lines(const char *filename, size_t *num_lines) {
    FILE *file;
    char buffer[1024];
    size_t line_count = 0;
    size_t capacity = INITIAL_CAPACITY;
    size_t i;
    char **temp;
    char **lines;
    size_t len;
    file = fopen(filename, "r");
    if (file == NULL) {
        perror("An Error Has Occurred");
        return NULL;
    }

    lines = malloc(capacity * sizeof(char *));
    if (lines == NULL) {
        perror("An Error Has Occurred");
        fclose(file);
        return NULL;
    }

    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        if (line_count >= capacity) {
            capacity *= 2;
            temp = realloc(lines, capacity * sizeof(char *));
            if (temp == NULL) {
                perror("An Error Has Occurred");
                for (i = 0; i < line_count; i++) {
                    free(lines[i]);
                }
                free(lines);
                fclose(file);
                return NULL;
            }
            lines = temp;
        }

        len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }

        lines[line_count] = malloc((strlen(buffer) + 1) * sizeof(char));
        if (lines[line_count] == NULL) {
            perror("An Error Has Occurred");
            for (i = 0; i < line_count; i++) {
                free(lines[i]);
            }
            free(lines);
            fclose(file);
            return NULL;
        }
        strcpy(lines[line_count], buffer);
        line_count++;
    }

    fclose(file);

    *num_lines = line_count;

    return lines;
}


void print_lines(char **lines, size_t num_lines) {
    size_t i;
    for ( i = 0; i < num_lines; i++) {
        printf("%s", lines[i]);
        if (i < num_lines - 1) {
            printf("\n"); 
        }
    }
}

size_t count_columns(const char *line) {
    size_t num_columns = 0;
    char *line_copy = strdup(line);
    if (line_copy) {
        char *token = strtok(line_copy, ",");
        while (token != NULL) {
            num_columns++;
            token = strtok(NULL, ",");
        }
        free(line_copy);
    }
    return num_columns;
}

int main(int argc, char *argv[]) {
    size_t i, j;
    double *X;
    double *A;
    double *D;
    double *W;
    char *token;
    char **lines;
    const char *task;
    const char *filename;
    size_t num_lines = 0;
    size_t num_columns = 0;

    if (argc != 3) {
        printf("An Error Has Occurred\n");
        return EXIT_FAILURE;
    }
    task = argv[1];
    filename = argv[2];

    lines = read_file_lines(filename, &num_lines);
  
    if (strcmp(task, "sym") != 0 && strcmp(task, "ddg") != 0 && strcmp(task, "norm") != 0) {
        fprintf(stderr, "An Error Has Occurred\n");
        for (i = 0; i < num_lines; i++) {
            free(lines[i]);
        }
        free(lines);
        return EXIT_FAILURE;
    }

    if (lines == NULL) {
        return EXIT_FAILURE;
    }

    if (num_lines > 0) {
        num_columns = count_columns(lines[0]);
    }
    X = (double*)malloc(num_lines * num_columns * sizeof(double));

    if (X == NULL) {
        perror("An Error Has Occurred");
        for (i = 0; i < num_lines; i++) {
            free(lines[i]);
        }
        free(lines);
        return EXIT_FAILURE;
    }

    for (i = 0; i < num_lines; i++) {
        token = strtok(lines[i], ",");
        for (j = 0; j < num_columns; j++) {
            if (token != NULL) {
                X[i * num_columns + j] = atof(token);
                token = strtok(NULL, ",");
            } else {
                X[i * num_columns + j] = 0.0; 
            }
        }
    }

    A = (double*)malloc(num_lines * num_lines * sizeof(double));
    if (A == NULL) {
        perror("An Error Has Occurred");
        free(X);
        for (i = 0; i < num_lines; i++) {
            free(lines[i]);
        }
        free(lines);
        return EXIT_FAILURE;
    }

    D = (double*)malloc(num_lines * num_lines * sizeof(double));
    if (D == NULL) {
        perror("An Error Has Occurred");
        free(X);
        free(A);
        for (i = 0; i < num_lines; i++) {
            free(lines[i]);
        }
        free(lines);
        return EXIT_FAILURE;
    }

    W = (double*)malloc(num_lines * num_lines * sizeof(double));
    if (W == NULL) {
        perror("An Error Has Occurred");
        free(X);
        free(A);
        free(D);
        for (i = 0; i < num_lines; i++) {
            free(lines[i]);
        }
        free(lines);
        return EXIT_FAILURE;
    }

    memset(A, 0, num_lines * num_lines * sizeof(double));
    memset(D, 0, num_lines * num_lines * sizeof(double));
    memset(W, 0, num_lines * num_lines * sizeof(double));

    sym(X, num_lines, num_columns, A);
    ddg(A, num_lines, D);
    norm(A, D, num_lines, W);

    if (strcmp(task, "sym") == 0) {
        for (i = 0; i < num_lines; i++) {
            for (j = 0; j < num_lines; j++) {
                if (j > 0) {
                    printf(",");
                }
                printf("%.4f", A[i * num_lines + j]);
            }
            printf("\n");
        }
    } else if (strcmp(task, "ddg") == 0) {
        for (i = 0; i < num_lines; i++) {
            for (j = 0; j < num_lines; j++) {
                if (j > 0) {
                    printf(",");
                }
                printf("%.4f", D[i * num_lines + j]);
            }
            printf("\n");
        }
    } else if (strcmp(task, "norm") == 0) {
        for (i = 0; i < num_lines; i++) {
            for (j = 0; j < num_lines; j++) {
                if (j > 0) {
                    printf(",");
                }
                printf("%.4f", W[i * num_lines + j]);
            }
            printf("\n");
        }
    } else {
        free(X);
        free(A);
        free(D);
        free(W);
        for (i = 0; i < num_lines; i++) {
            free(lines[i]);
        }
        free(lines);
        return EXIT_FAILURE;
    }

    free(X);
    free(D);
    free(A);
    free(W);
    for (i = 0; i < num_lines; i++) {
        free(lines[i]);
    }
    free(lines);

    return EXIT_SUCCESS;
}
