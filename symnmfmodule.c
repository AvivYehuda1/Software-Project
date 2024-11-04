#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "symnmf.h"

static PyObject* py_sym(PyObject* self, PyObject* args) {
    PyObject* X_obj;
    if (!PyArg_ParseTuple(args, "O", &X_obj)) {
        return NULL;
    }
    PyObject* X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (X_array == NULL) {
        return NULL;
    }
    double* X = (double*)PyArray_DATA((PyArrayObject*)X_array);
    int n = (int)PyArray_DIM((PyArrayObject*)X_array, 0);
    int d = (int)PyArray_DIM((PyArrayObject*)X_array, 1);
    
    npy_intp dims[2] = {n, n};
    PyObject* A_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (A_obj == NULL) {
        Py_DECREF(X_array);
        return NULL;
    }
    double* A = (double*)PyArray_DATA((PyArrayObject*)A_obj);
    sym(X, n, d, A);
    Py_DECREF(X_array);
    return A_obj;
}

static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject* A_obj;
    if (!PyArg_ParseTuple(args, "O", &A_obj)) {
        return NULL;
    }
    PyObject* A_array = PyArray_FROM_OTF(A_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (A_array == NULL) {
        return NULL;
    }
    double* A = (double*)PyArray_DATA((PyArrayObject*)A_array);
    int n = (int)PyArray_DIM((PyArrayObject*)A_array, 0);
    
    npy_intp dims[2] = {n, n};
    PyObject* D_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (D_obj == NULL) {
        Py_DECREF(A_array);
        return NULL;
    }
    double* D = (double*)PyArray_DATA((PyArrayObject*)D_obj);
    ddg(A, n, D);
    Py_DECREF(A_array);
    return D_obj;
}

static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyObject* A_obj;
    PyObject* D_obj;
    if (!PyArg_ParseTuple(args, "OO", &A_obj, &D_obj)) {
        return NULL;
    }
    PyObject* A_array = PyArray_FROM_OTF(A_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (A_array == NULL) {
        return NULL;
    }
    PyObject* D_array = PyArray_FROM_OTF(D_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (D_array == NULL) {
        Py_DECREF(A_array);
        return NULL;
    }
    double* A = (double*)PyArray_DATA((PyArrayObject*)A_array);
    double* D = (double*)PyArray_DATA((PyArrayObject*)D_array);
    int n = (int)PyArray_DIM((PyArrayObject*)A_array, 0);
    
    npy_intp dims[2] = {n, n};
    PyObject* W_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (W_obj == NULL) {
        Py_DECREF(A_array);
        Py_DECREF(D_array);
        return NULL;
    }
    double* W = (double*)PyArray_DATA((PyArrayObject*)W_obj);
    norm(A, D, n, W);
    Py_DECREF(A_array);
    Py_DECREF(D_array);
    return W_obj;
}

static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject* W_obj;
    PyObject* H_obj;
    int max_iter;
    double epsilon;
    if (!PyArg_ParseTuple(args, "OOid", &W_obj, &H_obj, &max_iter, &epsilon)) {
        return NULL;
    }
    PyObject* W_array = PyArray_FROM_OTF(W_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (W_array == NULL) {
        return NULL;
    }
    PyObject* H_array = PyArray_FROM_OTF(H_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    if (H_array == NULL) {
        Py_DECREF(W_array);
        return NULL;
    }
    double* W = (double*)PyArray_DATA((PyArrayObject*)W_array);
    double* H = (double*)PyArray_DATA((PyArrayObject*)H_array);
    int n = (int)PyArray_DIM((PyArrayObject*)W_array, 0);
    int k = (int)PyArray_DIM((PyArrayObject*)H_array, 1);
    
    symnmf(W, H, n, k, max_iter, epsilon);
    
    Py_DECREF(W_array);
    PyArray_ResolveWritebackIfCopy((PyArrayObject*)H_array);
    Py_INCREF(H_array); 
    return H_array;
}

static PyMethodDef symnmf_methods[] = {
    {"sym", py_sym, METH_VARARGS, "Calculate the similarity matrix"},
    {"ddg", py_ddg, METH_VARARGS, "Calculate the diagonal degree matrix"},
    {"norm", py_norm, METH_VARARGS, "Calculate the normalized similarity matrix"},
    {"symnmf", py_symnmf, METH_VARARGS, "Perform symmetric NMF"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmf_module = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    NULL,
    -1,
    symnmf_methods
};

PyMODINIT_FUNC PyInit_symnmf(void) {
    import_array(); // For numpy initialization
    return PyModule_Create(&symnmf_module);
}
