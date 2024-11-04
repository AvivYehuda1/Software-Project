#ifndef SYMNMF_H
#define SYMNMF_H

void sym(double *X, size_t n, size_t d, double *A);
void ddg(double *A, size_t n, double *D);
void norm(double *A, double *D, size_t n, double *W);
void symnmf(double *W, double *H, size_t n, size_t k, size_t max_iter, double epsilon);

#endif
