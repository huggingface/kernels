#define Py_LIMITED_API 0x03090000
#include <Python.h>

#include "utils.h"

static PyObject *cuda_error(int error) {
  PyErr_SetString(PyExc_RuntimeError, error_string(error));
  return NULL;
}

static int parse_args(PyObject *args, long *a, long *b) {
  if (!PyArg_ParseTuple(args, "ll", a, b)) {
    return 0;
  }
  if (*a < 0 || *b <= 0) {
    PyErr_SetString(PyExc_ValueError, "requires a >= 0 and b > 0");
    return 0;
  }
  return 1;
}

static PyObject *py_runtime_version(PyObject *self, PyObject *args) {
  int version;
  int error = runtime_version(&version);
  return error == 0 ? PyLong_FromLong(version) : cuda_error(error);
}

static PyObject *py_device_count(PyObject *self, PyObject *args) {
  int count;
  int error = device_count(&count);
  return error == 0 ? PyLong_FromLong(count) : cuda_error(error);
}

static PyObject *py_cdiv(PyObject *self, PyObject *args) {
  long a, b;
  if (!parse_args(args, &a, &b)) {
    return NULL;
  }
  return PyLong_FromLong(cdiv(a, b));
}

static PyObject *py_round_up(PyObject *self, PyObject *args) {
  long x, multiple;
  if (!parse_args(args, &x, &multiple)) {
    return NULL;
  }
  return PyLong_FromLong(round_up(x, multiple));
}

static PyObject *py_cdiv_on_device(PyObject *self, PyObject *args) {
  long a, b, result;
  if (!parse_args(args, &a, &b)) {
    return NULL;
  }
  int error = cdiv_on_device(a, b, &result);
  return error == 0 ? PyLong_FromLong(result) : cuda_error(error);
}

static PyMethodDef methods[] = {
    {"runtime_version", py_runtime_version, METH_NOARGS,
     "Version of the CUDA runtime."},
    {"device_count", py_device_count, METH_NOARGS, "Number of CUDA devices."},
    {"cdiv", py_cdiv, METH_VARARGS, "Ceiling division."},
    {"round_up", py_round_up, METH_VARARGS, "Round up to a multiple."},
    {"cdiv_on_device", py_cdiv_on_device, METH_VARARGS,
     "Ceiling division on the GPU."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_utils", NULL, -1,
                                    methods};

PyMODINIT_FUNC PyInit__utils(void) { return PyModule_Create(&module); }
