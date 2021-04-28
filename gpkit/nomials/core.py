"The shared non-mathematical backbone of all Nomials"
from .data import NomialData
from ..small_classes import Numbers, FixedScalar
from ..repr_conventions import MUL, UNICODE_EXPONENTS


def nomial_latex_helper(c, pos_vars, neg_vars):
    """Combines (varlatex, exponent) tuples,
    separated by positive vs negative exponent, into a single latex string."""
    pvarstrs = ['%s^{%.2g}' % (varl, x) if "%.2g" % x != "1" else varl
                for (varl, x) in pos_vars]
    nvarstrs = ['%s^{%.2g}' % (varl, -x) if "%.2g" % -x != "1" else varl
                for (varl, x) in neg_vars]
    pvarstr = " ".join(sorted(pvarstrs))
    nvarstr = " ".join(sorted(nvarstrs))
    cstr = "%.2g" % c
    if pos_vars and cstr in ["1", "-1"]:
        cstr = cstr[:-1]
    else:
        cstr = "%.4g" % c
        if "e" in cstr:  # use exponential notation
            idx = cstr.index("e")
            cstr = "%s \\times 10^{%i}" % (cstr[:idx], int(cstr[idx+1:]))

    if pos_vars and neg_vars:
        return "%s\\frac{%s}{%s}" % (cstr, pvarstr, nvarstr)
    if neg_vars and not pos_vars:
        return "\\frac{%s}{%s}" % (cstr, nvarstr)
    if pos_vars:
        return "%s%s" % (cstr, pvarstr)
    return "%s" % cstr


class Nomial(NomialData):
    "Shared non-mathematical properties of all nomials"
    sub = None

    def str_without(self, excluded=()):
        "String representation, excluding fields ('units', varkey attributes)"
        units = "" if "units" in excluded else self.unitstr(" [%s]")
        if hasattr(self, "key"):
            return self.key.str_without(excluded) + units  # pylint: disable=no-member
        if self.ast:
            return self.parse_ast(excluded) + units
        mstrs = []
        for exp, c in self.hmap.items():
            pvarstrs, nvarstrs = [], []
            for (var, x) in sorted(exp.items(),
                                   key=lambda vx: (vx[1], str(vx[0]))):
                if not x:
                    continue
                if x > 0:
                    varstrlist = pvarstrs
                else:
                    x = -x
                    varstrlist = nvarstrs
                varstr = var.str_without(excluded)
                if UNICODE_EXPONENTS and int(x) == x and 2 <= x <= 9:
                    x = int(x)
                    if x in (2, 3):
                        varstr += chr(176+x)
                    elif x in (4, 5, 6, 7, 8, 9):
                        varstr += chr(8304+x)
                elif x != 1:
                    varstr += "^%.2g" % x
                varstrlist.append(varstr)
            numerator_strings = pvarstrs
            cstr = "%.3g" % c
            if cstr == "-1":
                cstr = "-"
            if numerator_strings and cstr == "1":
                mstr = MUL.join(pvarstrs)
            else:
                mstr = MUL.join([cstr] + pvarstrs)
            if nvarstrs:
                mstr = mstr + "/" + "/".join(nvarstrs)
            mstrs.append(mstr)
        return " + ".join(sorted(mstrs)) + units

    def latex(self, excluded=()):  # TODO: add ast parsing here
        "Latex representation, parsing `excluded` just as .str_without does"
        mstrs = []
        for exp, c in self.hmap.items():
            pos_vars, neg_vars = [], []
            for var, x in exp.items():
                if x > 0:
                    pos_vars.append((var.latex(excluded), x))
                elif x < 0:
                    neg_vars.append((var.latex(excluded), x))
            mstrs.append(nomial_latex_helper(c, pos_vars, neg_vars))

        if "units" in excluded:
            return " + ".join(sorted(mstrs))
        units = self.unitstr(r"\mathrm{~\left[ %s \right]}", ":L~")
        units_tf = units.replace("frac", "tfrac").replace(r"\cdot", r"\cdot ")
        return " + ".join(sorted(mstrs)) + units_tf

    @property
    def value(self):
        """Self, with values substituted for variables that have values

        Returns
        -------
        float, if no symbolic variables remain after substitution
        (Monomial, Posynomial, or Nomial), otherwise.
        """
        if isinstance(self, FixedScalar):
            return self.cs[0]
        p = self.sub({k: k.value for k in self.vks if "value" in k.descr})  # pylint: disable=not-callable
        return p.cs[0] if isinstance(p, FixedScalar) else p

    def __eq__(self, other):
        "True if self and other are algebraically identical."
        if isinstance(other, Numbers):
            return isinstance(self, FixedScalar) and self.value == other
        return super().__eq__(other)

    __hash__ = NomialData.__hash__
    # pylint: disable=multiple-statements
    def __ne__(self, other): return not Nomial.__eq__(self, other)
    def __radd__(self, other): return self.__add__(other, rev=True)   # pylint: disable=no-member
    def __rmul__(self, other): return self.__mul__(other, rev=True)   # pylint: disable=no-member

    def prod(self):
        "Return self for compatibility with NomialArray"
        return self

    sum = prod
