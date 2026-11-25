#define Py_LIMITED_API 0x03090000
#include <Python.h>

static long cdiv_impl(long a, long b) { return (a + b - 1) / b; }

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

static PyObject *cdiv(PyObject *self, PyObject *args) {
  long a, b;
  if (!parse_args(args, &a, &b)) {
    return NULL;
  }
  return PyLong_FromLong(cdiv_impl(a, b));
}

static PyObject *round_up(PyObject *self, PyObject *args) {
  long x, multiple;
  if (!parse_args(args, &x, &multiple)) {
    return NULL;
  }
  return PyLong_FromLong(cdiv_impl(x, multiple) * multiple);
}

static PyMethodDef methods[] = {
    {"cdiv", cdiv, METH_VARARGS, "Ceiling division."},
    {"round_up", round_up, METH_VARARGS, "Round up to a multiple."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_utils", NULL, -1,
                                    methods};

PyMODINIT_FUNC PyInit__utils(void) { return PyModule_Create(&module); }
