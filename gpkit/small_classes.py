"""Miscellaneous small classes"""
from operator import xor
from functools import reduce
import numpy as np
from .units import Quantity, qty  # pylint: disable=unused-import

Strings = (str,)
Numbers = (int, float, np.number, Quantity)


class FixedScalarMeta(type):
    "Metaclass to implement instance checking for fixed scalars"
    def __instancecheck__(cls, obj):
        return getattr(obj, "hmap", None) and len(obj.hmap) == 1 and not obj.vks


class FixedScalar(metaclass=FixedScalarMeta):  # pylint: disable=no-init
    "Instances of this class are scalar Nomials with no variables"


class Count:
    "Like python 2's itertools.count, for Python 3 compatibility."
    def __init__(self):
        self.count = -1

    def next(self):
        "Increment self.count and return it"
        self.count += 1
        return self.count


def matrix_converter(name):
    "Generates conversion function."
    def to_(self):  # used in tocoo, tocsc, etc below
        "Converts to another type of matrix."
        # pylint: disable=unused-variable
        return getattr(self.tocsr(), "to"+name)()
    return to_


class CootMatrix:
    "A very simple sparse matrix representation."
    def __init__(self, row, col, data):
        self.row, self.col, self.data = row, col, data
        self.shape = [(max(self.row) + 1) if self.row else 0,
                      (max(self.col) + 1) if self.col else 0]

    def __eq__(self, other):
        return (self.row == other.row and self.col == other.col
                and self.data == other.data and self.shape == other.shape)

    tocoo = matrix_converter("coo")
    tocsc = matrix_converter("csc")
    todia = matrix_converter("dia")
    todok = matrix_converter("dok")
    todense = matrix_converter("dense")

    def tocsr(self):
        "Converts to a Scipy sparse csr_matrix"
        from scipy.sparse import csr_matrix
        return csr_matrix((self.data, (self.row, self.col)))

    def dot(self, arg):
        "Returns dot product with arg."
        return self.tocsr().dot(arg)


class SolverLog(list):
    "Adds a `write` method to list so it's file-like and can replace stdout."
    def __init__(self, output=None, *, verbosity=0):
        list.__init__(self)
        self.verbosity = verbosity
        self.output = output

    def write(self, writ):
        "Append and potentially write the new line."
        if writ[:2] == "b'":
            writ = writ[2:-1]
        if writ != "\n":
            self.append(writ.rstrip("\n"))
        if self.verbosity > 0:  # pragma: no cover
            self.output.write(writ)


class DictOfLists(dict):
    "A hierarchy of dicionaries, with lists at the bottom."

    def append(self, sol):
        "Appends a dict (of dicts) of lists to all held lists."
        if not hasattr(self, "initialized"):
            _enlist_dict(sol, self)
            self.initialized = True  # pylint: disable=attribute-defined-outside-init
        else:
            _append_dict(sol, self)

    def atindex(self, i):
        "Indexes into each list independently."
        return self.__class__(_index_dict(i, self, self.__class__()))

    def to_arrays(self):
        "Converts all lists into array."
        _enray(self, self)


def _enlist_dict(d_in, d_out):
    "Recursively copies d_in into d_out, placing non-dict items into lists."
    for k, v in d_in.items():
        if isinstance(v, dict):
            d_out[k] = _enlist_dict(v, v.__class__())
        else:
            d_out[k] = [v]
    assert set(d_in.keys()) == set(d_out.keys())
    return d_out


def _append_dict(d_in, d_out):
    "Recursively travels dict d_out and appends items found in d_in."
    for k, v in d_in.items():
        if isinstance(v, dict):
            d_out[k] = _append_dict(v, d_out[k])
        else:
            d_out[k].append(v)
    return d_out


def _index_dict(idx, d_in, d_out):
    "Recursively travels dict d_in, placing items at idx into dict d_out."
    for k, v in d_in.items():
        if isinstance(v, dict):
            d_out[k] = _index_dict(idx, v, v.__class__())
        else:
            try:
                d_out[k] = v[idx]
            except (IndexError, TypeError):  # if not an array, return as is
                d_out[k] = v
    return d_out


def _enray(d_in, d_out):
    "Recursively turns lists into numpy arrays."
    for k, v in d_in.items():
        if isinstance(v, dict):
            d_out[k] = _enray(v, v.__class__())
        else:
            if len(v) == 1:
                v, = v
            else:
                v = np.array(v)
            d_out[k] = v
    return d_out


class HashVector(dict):
    """A simple, sparse, string-indexed vector. Inherits from dict.

    The HashVector class supports element-wise arithmetic:
    any undeclared variables are assumed to have a value of zero.

    Arguments
    ---------
    arg : iterable

    Example
    -------
    >>> x = gpkit.nomials.Monomial("x")
    >>> exp = gpkit.small_classes.HashVector({x: 2})
    """
    hashvalue = None

    def __hash__(self):
        "Allows HashVectors to be used as dictionary keys."
        if self.hashvalue is None:
            self.hashvalue = reduce(xor, map(hash, self.items()), 0)
        return self.hashvalue

    def copy(self):
        "Return a copy of this"
        hv = self.__class__(self)
        hv.hashvalue = self.hashvalue
        return hv

    def __pow__(self, other):
        "Accepts scalars. Return Hashvector with each value put to a power."
        if isinstance(other, Numbers):
            return self.__class__({k: v**other for (k, v) in self.items()})
        return NotImplemented

    def __mul__(self, other):
        """Accepts scalars and dicts. Returns with each value multiplied.

        If the other object inherits from dict, multiplication is element-wise
        and their key's intersection will form the new keys."""
        try:
            return self.__class__({k: v*other for (k, v) in self.items()})
        except:  # pylint: disable=bare-except
            return NotImplemented

    def __add__(self, other):
        """Accepts scalars and dicts. Returns with each value added.

        If the other object inherits from dict, addition is element-wise
        and their key's union will form the new keys."""
        if isinstance(other, Numbers):
            return self.__class__({k: v + other for (k, v) in self.items()})
        if isinstance(other, dict):
            sums = self.copy()
            for key, value in other.items():
                if key in sums:
                    svalue = sums[key]
                    if value == -svalue:
                        del sums[key]  # remove zeros created by addition
                    else:
                        sums[key] = value + svalue
                else:
                    sums[key] = value
            sums.hashvalue = None
            return sums
        return NotImplemented

    # pylint: disable=multiple-statements
    def __neg__(self): return -1*self
    def __sub__(self, other): return self + -other
    def __rsub__(self, other): return other + -self
    def __radd__(self, other): return self + other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def __rmul__(self, other): return self * other


EMPTY_HV = HashVector()
