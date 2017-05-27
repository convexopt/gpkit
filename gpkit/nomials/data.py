"""Machinery for exps, cs, varlocs data -- common to nomials and programs"""
from functools import reduce as functools_reduce
from operator import add
import numpy as np
from ..small_classes import HashVector
from ..keydict import KeySet, KeyDict
from ..small_scripts import mag
from .map import NomialMap
from ..repr_conventions import _repr


class NomialData(object):
    """Object for holding cs, exps, and other basic 'nomial' properties.

    cs: array (coefficient of each monomial term)
    exps: tuple of {VarKey: float} (exponents of each monomial term)
    varlocs: {VarKey: list} (terms each variable appears in)
    units: pint.UnitsContainer
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, hmap):
        self.hmap = hmap

        self.vks = {}
        for exp in self.hmap:
            for vk in exp:
                if vk not in self.vks:
                    self.vks[vk] = None
        self.units = self.hmap.units
        self.any_nonpositive_cs = any(c <= 0 for c in self.hmap.values())
        self._reset()

    def _reset(self):
        for attr in "hashvalue varlocs exps cs varkeys values".split():
            setattr(self, "_"+attr, None)

    @property
    def varlocs(self):
        if self._varlocs is None:
            self._varlocs = {}
            for i, exp in enumerate(self.exps):
                for var in exp:
                    if var not in self._varlocs:
                        self._varlocs[var] = []
                    self._varlocs[var].append(i)
        return self._varlocs

    @property
    def exps(self):
        if self._exps is None:
            self._exps = tuple(self.hmap.keys())
        return self._exps

    @property
    def cs(self):
        if self._cs is None:
            self._cs = np.array(self.hmap.values())
            if self.hmap.units:
                self._cs = self._cs*self.hmap.units
        return self._cs

    __repr__ = _repr

    def __hash__(self):
        if self._hashvalue is None:
            self._hashvalue = hash(hash(self.hmap) + hash(str(self.hmap.units)))
        return self._hashvalue

    @property
    def varkeys(self):
        "The NomialData's varkeys, created when necessary for a substitution."
        if self._varkeys is None:
            self._varkeys = KeySet(self.vks)
        return self._varkeys

    @property
    def values(self):
        "The NomialData's values, created when necessary."
        if self._values is None:
            self._values = KeyDict({k: k.descr["value"] for k in self.vks
                                    if "value" in k.descr})
        return self._values

    def diff(self, var):
        """Derivative of this with respect to a Variable

        Arguments
        ---------
        var (Variable):
            Variable to take derivative with respect to

        Returns
        -------
        NomialData
        """
        varset = self.varkeys[var]
        if len(varset) > 1:
            raise ValueError("multiple variables %s found for key %s"
                             % (list(varset), var))
        elif len(varset) == 0:
            hmap = NomialMap({HashVector(): 0})
            hmap.set_units(None)
        else:
            var, = varset
            hmap = self.hmap.diff(var)
        return NomialData(hmap)

    def __eq__(self, other):
        "Equality test"
        if not hasattr(other, "hmap"):
            return NotImplemented
        if self.hmap != other.hmap:
            return False
        if self.units != other.units:
            return False
        return True
