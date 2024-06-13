"""Defines SolutionArray class"""
import sys
import re
import json
import difflib
from operator import sub
import warnings as pywarnings
import pickle
import gzip
import json
import pickletools
from collections import defaultdict
import numpy as np
from .nomials import NomialArray
from .small_classes import DictOfLists, Strings, SolverLog
from .small_scripts import mag, try_str_without
from .repr_conventions import unitstr, lineagestr, UNICODE_EXPONENTS
from .breakdowns import Breakdowns


CONSTRSPLITPATTERN = re.compile(r"([^*]\*[^*])|( \+ )|( >= )|( <= )|( = )")

VALSTR_REPLACES = [
    ("+nan", " nan"),
    ("-nan", " nan"),
    ("nan%", "nan "),
    ("nan", " - "),
]


class SolSavingEnvironment:
    """Temporarily removes construction/solve attributes from constraints.

    This approximately halves the size of the pickled solution.
    """

    def __init__(self, solarray, saveconstraints):
        self.solarray = solarray
        self.attrstore = {}
        self.saveconstraints = saveconstraints
        self.constraintstore = None


    def __enter__(self):
        if "sensitivities" not in self.solarray:
            pass
        elif self.saveconstraints:
            for constraint_attr in ["bounded", "meq_bounded", "vks",
                                    "v_ss", "unsubbed", "varkeys"]:
                store = {}
                for constraint in self.solarray["sensitivities"]["constraints"]:
                    if getattr(constraint, constraint_attr, None):
                        store[constraint] = getattr(constraint, constraint_attr)
                        delattr(constraint, constraint_attr)
                self.attrstore[constraint_attr] = store
        else:
            self.constraintstore = \
                self.solarray["sensitivities"].pop("constraints")

    def __exit__(self, type_, val, traceback):
        if self.saveconstraints:
            for constraint_attr, store in self.attrstore.items():
                for constraint, value in store.items():
                    setattr(constraint, constraint_attr, value)
        elif self.constraintstore:
            self.solarray["sensitivities"]["constraints"] = self.constraintstore

def msenss_table(data, _, **kwargs):
    "Returns model sensitivity table lines"
    if "models" not in data.get("sensitivities", {}):
        return ""
    data = sorted(data["sensitivities"]["models"].items(),
                  key=lambda i: ((i[1] < 0.1).all(),
                                 -np.max(i[1]) if (i[1] < 0.1).all()
                                 else -round(np.mean(i[1]), 1), i[0]))
    lines = ["Model Sensitivities", "-------------------"]
    if kwargs["sortmodelsbysenss"]:
        lines[0] += " (sorts models in sections below)"
    previousmsenssstr = ""
    for model, msenss in data:
        if not model:  # for now let's only do named models
            continue
        if (msenss < 0.1).all():
            msenss = np.max(msenss)
            if msenss:
                msenssstr = "%6s" % ("<1e%i" % max(-3, np.log10(msenss)))
            else:
                msenssstr = "  =0  "
        else:
            meansenss = round(np.mean(msenss), 1)
            msenssstr = "%+6.1f" % meansenss
            deltas = msenss - meansenss
            if np.max(np.abs(deltas)) > 0.1:
                deltastrs = ["%+4.1f" % d if abs(d) >= 0.1 else "  - "
                             for d in deltas]
                msenssstr += " + [ %s ]" % "  ".join(deltastrs)
        if msenssstr == previousmsenssstr:
            msenssstr = " "*len(msenssstr)
        else:
            previousmsenssstr = msenssstr
        lines.append("%s : %s" % (msenssstr, model))
    return lines + [""] if len(lines) > 3 else []


def senss_table(data, showvars=(), title="Variable Sensitivities", **kwargs):
    "Returns sensitivity table lines"
    if "variables" in data.get("sensitivities", {}):
        data = data["sensitivities"]["variables"]
    if showvars:
        data = {k: data[k] for k in showvars if k in data}
    return var_table(data, title, sortbyvals=True, skipifempty=True,
                     valfmt="%+-.2g  ", vecfmt="%+-8.2g",
                     printunits=False, minval=1e-3, **kwargs)


def topsenss_table(data, showvars, nvars=5, **kwargs):
    "Returns top sensitivity table lines"
    data, filtered = topsenss_filter(data, showvars, nvars)
    title = "Most Sensitive Variables"
    if filtered:
        title = "Next Most Sensitive Variables"
    return senss_table(data, title=title, hidebelowminval=True, **kwargs)


def topsenss_filter(data, showvars, nvars=5):
    "Filters sensitivities down to top N vars"
    if "variables" in data.get("sensitivities", {}):
        data = data["sensitivities"]["variables"]
    mean_abs_senss = {k: np.abs(s).mean() for k, s in data.items()
                      if not np.isnan(s).any()}
    topk = [k for k, _ in sorted(mean_abs_senss.items(), key=lambda l: l[1])]
    filter_already_shown = showvars.intersection(topk)
    for k in filter_already_shown:
        topk.remove(k)
        if nvars > 3:  # always show at least 3
            nvars -= 1
    return {k: data[k] for k in topk[-nvars:]}, filter_already_shown


def insenss_table(data, _, maxval=0.1, **kwargs):
    "Returns insensitivity table lines"
    if "constants" in data.get("sensitivities", {}):
        data = data["sensitivities"]["variables"]
    data = {k: s for k, s in data.items() if np.mean(np.abs(s)) < maxval}
    return senss_table(data, title="Insensitive Fixed Variables", **kwargs)


def tight_table(self, _, ntightconstrs=5, tight_senss=1e-2, **kwargs):
    "Return constraint tightness lines"
    title = "Most Sensitive Constraints"
    if len(self) > 1:
        title += " (in last sweep)"
        data = sorted(((-float("%+6.2g" % abs(s[-1])), str(c)),
                       "%+6.2g" % abs(s[-1]), id(c), c)
                      for c, s in self["sensitivities"]["constraints"].items()
                      if s[-1] >= tight_senss)[:ntightconstrs]
    else:
        data = sorted(((-float("%+6.2g" % abs(s)), str(c)),
                       "%+6.2g" % abs(s), id(c), c)
                      for c, s in self["sensitivities"]["constraints"].items()
                      if s >= tight_senss)[:ntightconstrs]
    return constraint_table(data, title, **kwargs)

def loose_table(self, _, min_senss=1e-5, **kwargs):
    "Return constraint tightness lines"
    title = "Insensitive Constraints |below %+g|" % min_senss
    if len(self) > 1:
        title += " (in last sweep)"
        data = [(0, "", id(c), c)
                for c, s in self["sensitivities"]["constraints"].items()
                if s[-1] <= min_senss]
    else:
        data = [(0, "", id(c), c)
                for c, s in self["sensitivities"]["constraints"].items()
                if s <= min_senss]
    return constraint_table(data, title, **kwargs)


# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def constraint_table(data, title, sortbymodel=True, showmodels=True, **_):
    "Creates lines for tables where the right side is a constraint."
    # TODO: this should support 1D array inputs from sweeps
    excluded = {"units"} if showmodels else {"units", "lineage"}
    models, decorated = {}, []
    for sortby, openingstr, _, constraint in sorted(data):
        model = lineagestr(constraint) if sortbymodel else ""
        if model not in models:
            models[model] = len(models)
        constrstr = try_str_without(
            constraint, {":MAGIC:"+lineagestr(constraint)}.union(excluded))
        if " at 0x" in constrstr:  # don't print memory addresses
            constrstr = constrstr[:constrstr.find(" at 0x")] + ">"
        decorated.append((models[model], model, sortby, constrstr, openingstr))
    decorated.sort()
    previous_model, lines = None, []
    for varlist in decorated:
        _, model, _, constrstr, openingstr = varlist
        if model != previous_model:
            if lines:
                lines.append(["", ""])
            if model or lines:
                lines.append([("newmodelline",), model])
            previous_model = model
        minlen, maxlen = 25, 80
        segments = [s for s in CONSTRSPLITPATTERN.split(constrstr) if s]
        constraintlines = []
        line = ""
        next_idx = 0
        while next_idx < len(segments):
            segment = segments[next_idx]
            next_idx += 1
            if CONSTRSPLITPATTERN.match(segment) and next_idx < len(segments):
                segments[next_idx] = segment[1:] + segments[next_idx]
                segment = segment[0]
            elif len(line) + len(segment) > maxlen and len(line) > minlen:
                constraintlines.append(line)
                line = "  "  # start a new line
            line += segment
            while len(line) > maxlen:
                constraintlines.append(line[:maxlen])
                line = "  " + line[maxlen:]
        constraintlines.append(line)
        lines += [(openingstr + " : ", constraintlines[0])]
        lines += [("", l) for l in constraintlines[1:]]
    if not lines:
        lines = [("", "(none)")]
    maxlens = np.max([list(map(len, line)) for line in lines
                      if line[0] != ("newmodelline",)], axis=0)
    dirs = [">", "<"]  # we'll check lengths before using zip
    assert len(list(dirs)) == len(list(maxlens))
    fmts = ["{0:%s%s}" % (direc, L) for direc, L in zip(dirs, maxlens)]
    for i, line in enumerate(lines):
        if line[0] == ("newmodelline",):
            linelist = [fmts[0].format(" | "), line[1]]
        else:
            linelist = [fmt.format(s) for fmt, s in zip(fmts, line)]
        lines[i] = "".join(linelist).rstrip()
    return [title] + ["-"*len(title)] + lines + [""]


def warnings_table(self, _, **kwargs):
    "Makes a table for all warnings in the solution."
    title = "WARNINGS"
    lines = ["~"*len(title), title, "~"*len(title)]
    if "warnings" not in self or not self["warnings"]:
        return []
    for wtype in sorted(self["warnings"]):
        data_vec = self["warnings"][wtype]
        if len(data_vec) == 0:
            continue
        if not hasattr(data_vec, "shape"):
            data_vec = [data_vec]  # not a sweep
        else:
            all_equal = True
            for data in data_vec[1:]:
                eq_i = (data == data_vec[0])
                if hasattr(eq_i, "all"):
                    eq_i = eq_i.all()
                if not eq_i:
                    all_equal = False
                    break
            if all_equal:
                data_vec = [data_vec[0]]  # warnings identical across sweeps
        for i, data in enumerate(data_vec):
            if len(data) == 0:
                continue
            data = sorted(data, key=lambda l: l[0])  # sort by msg
            title = wtype
            if len(data_vec) > 1:
                title += " in sweep %i" % i
            if wtype == "Unexpectedly Tight Constraints" and data[0][1]:
                data = [(-int(1e5*relax_sensitivity),
                         "%+6.2g" % relax_sensitivity, id(c), c)
                        for _, (relax_sensitivity, c) in data]
                lines += constraint_table(data, title, **kwargs)
            elif wtype == "Unexpectedly Loose Constraints" and data[0][1]:
                data = [(-int(1e5*rel_diff),
                         "%.4g %s %.4g" % tightvalues, id(c), c)
                        for _, (rel_diff, tightvalues, c) in data]
                lines += constraint_table(data, title, **kwargs)
            else:
                lines += [title] + ["-"*len(wtype)]
                lines += [msg for msg, _ in data] + [""]
    if len(lines) == 3:  # just the header
        return []
    lines[-1] = "~~~~~~~~"
    return lines + [""]

def bdtable_gen(key):
    "Generator for breakdown tablefns"

    def bdtable(self, _showvars, **_):
        "Cost breakdown plot"
        bds = Breakdowns(self)
        original_stdout = sys.stdout
        try:
            sys.stdout = SolverLog(original_stdout, verbosity=0)
            bds.plot(key)
        finally:
            lines = sys.stdout.lines()
            sys.stdout = original_stdout
        return lines

    return bdtable


TABLEFNS = {"sensitivities": senss_table,
            "top sensitivities": topsenss_table,
            "insensitivities": insenss_table,
            "model sensitivities": msenss_table,
            "tightest constraints": tight_table,
            "loose constraints": loose_table,
            "warnings": warnings_table,
            "model sensitivities breakdown": bdtable_gen("model sensitivities"),
            "cost breakdown": bdtable_gen("cost")
           }

def unrolled_absmax(values):
    "From an iterable of numbers and arrays, returns the largest magnitude"
    finalval, absmaxest = None, 0
    for val in values:
        absmaxval = np.abs(val).max()
        if absmaxval >= absmaxest:
            absmaxest, finalval = absmaxval, val
    if getattr(finalval, "shape", None):
        return finalval[np.unravel_index(np.argmax(np.abs(finalval)),
                                         finalval.shape)]
    return finalval


def cast(function, val1, val2):
    "Relative difference between val1 and val2 (positive if val2 is larger)"
    with pywarnings.catch_warnings():  # skip those pesky divide-by-zeros
        pywarnings.simplefilter("ignore")
        if hasattr(val1, "shape") and hasattr(val2, "shape"):
            if val1.ndim == val2.ndim:
                return function(val1, val2)
            lessdim, dimmest = sorted([val1, val2], key=lambda v: v.ndim)
            dimdelta = dimmest.ndim - lessdim.ndim
            add_axes = (slice(None),)*lessdim.ndim + (np.newaxis,)*dimdelta
            if dimmest is val1:
                return function(dimmest, lessdim[add_axes])
            if dimmest is val2:
                return function(lessdim[add_axes], dimmest)
        return function(val1, val2)


def diff_retrieval(self, other, sharedvks, showvars=None, *, jsondiff=False,
        senssdiff=False, absdiff=False, reldiff=False):
    """A helper function for generalized diff method
       - retreives svars and ovars,
    """
    svars = self["variables"]
    ovars = other["variables"]
    # get the type of diffs
    diff_dict = {}
    if jsondiff == False:
        if reldiff:
            rel_diff = {vk: 100*(cast(np.divide, svars[vk], ovars[vk]) - 1)
                    for vk in sharedvks}
            diff_dict['rel'] = rel_diff
        if absdiff:
            abs_diff = {vk: cast(sub, svars[vk], ovars[vk]) for vk in sharedvks}
            diff_dict['abs'] = abs_diff
        if senssdiff:
            ssenss = self["sensitivities"]["variables"]
            osenss = other["sensitivities"]["variables"]
            senss_delta = {vk: cast(sub, ssenss[vk], osenss[vk])
                    for vk in sharedvks}
            diff_dict['sens'] = senss_delta
    else:
        if reldiff:
            rel_diff = {}
            for vk in sharedvks:
                val = 100*(cast(np.divide, svars[vk], ovars[vk]) - 1)
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                rel_diff[str(vk)] = val
            diff_dict['rel'] = rel_diff
        if absdiff:
            abs_diff = {}
            for vk in sharedvks:
                val = cast(sub, svars[vk], ovars[vk])
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                abs_diff[str(vk)] = val
            diff_dict['abs'] = abs_diff
        if senssdiff:
            sense_delta = {}
            ssenss = self["sensitivities"]["variables"]
            osenss = other["sensitivities"]["variables"]
            for vk in sharedvks:
                val = cast(sub, ssenss[vk], osenss[vk])
                sense_delta[str(vk)] = val
            diff_dict['sens'] = senss_delta 
    return diff_dict


class SolutionArray(DictOfLists):
    """A dictionary (of dictionaries) of lists, with convenience methods.

    Items
    -----
    cost : array
    variables: dict of arrays
    sensitivities: dict containing:
        monomials : array
        posynomials : array
        variables: dict of arrays
    localmodels : NomialArray
        Local power-law fits (small sensitivities are cut off)

    Example
    -------
    >>> import gpkit
    >>> import numpy as np
    >>> x = gpkit.Variable("x")
    >>> x_min = gpkit.Variable("x_{min}", 2)
    >>> sol = gpkit.Model(x, [x >= x_min]).solve(verbosity=0)
    >>>
    >>> # VALUES
    >>> values = [sol(x), sol.subinto(x), sol["variables"]["x"]]
    >>> assert all(np.array(values) == 2)
    >>>
    >>> # SENSITIVITIES
    >>> senss = [sol.sens(x_min), sol.sens(x_min)]
    >>> senss.append(sol["sensitivities"]["variables"]["x_{min}"])
    >>> assert all(np.array(senss) == 1)
    """
    modelstr = ""
    _name_collision_varkeys = None
    _lineageset = False
    table_titles = {"choicevariables": "Choice Variables",
                    "sweepvariables": "Swept Variables",
                    "freevariables": "Free Variables",
                    "constants": "Fixed Variables",  # TODO: change everywhere
                    "variables": "Variables"}

    def set_necessarylineage(self, clear=False):  # pylint: disable=too-many-branches
        "Returns the set of contained varkeys whose names are not unique"
        if self._name_collision_varkeys is None:
            self._name_collision_varkeys = {}
            self["variables"].update_keymap()
            keymap = self["variables"].keymap
            name_collisions = defaultdict(set)
            for key in keymap:
                if hasattr(key, "key"):
                    if len(keymap[key.name]) == 1:  # very unique
                        self._name_collision_varkeys[key] = 0
                    else:
                        shortname = key.str_without(["lineage", "vec"])
                        if len(keymap[shortname]) > 1:
                            name_collisions[shortname].add(key)
            for varkeys in name_collisions.values():
                min_namespaced = defaultdict(set)
                for vk in varkeys:
                    *_, mineage = vk.lineagestr().split(".")
                    min_namespaced[(mineage, 1)].add(vk)
                while any(len(vks) > 1 for vks in min_namespaced.values()):
                    for key, vks in list(min_namespaced.items()):
                        if len(vks) <= 1:
                            continue
                        del min_namespaced[key]
                        mineage, idx = key
                        idx += 1
                        for vk in vks:
                            lineages = vk.lineagestr().split(".")
                            submineage = lineages[-idx] + "." + mineage
                            min_namespaced[(submineage, idx)].add(vk)
                for (_, idx), vks in min_namespaced.items():
                    vk, = vks
                    self._name_collision_varkeys[vk] = idx
        if clear:
            self._lineageset = False
            for vk in self._name_collision_varkeys:
                del vk.descr["necessarylineage"]
        else:
            self._lineageset = True
            for vk, idx in self._name_collision_varkeys.items():
                vk.descr["necessarylineage"] = idx

    def __len__(self):
        try:
            return len(self["cost"])
        except TypeError:
            return 1
        except KeyError:
            return 0

    def __call__(self, posy):
        posy_subbed = self.subinto(posy)
        return getattr(posy_subbed, "c", posy_subbed)

    def almost_equal(self, other, reltol=1e-3):
        "Checks for almost-equality between two solutions"
        svars, ovars = self["variables"], other["variables"]
        svks, ovks = set(svars), set(ovars)
        if svks != ovks:
            return False
        for key in svks:
            reldiff = np.max(abs(cast(np.divide, svars[key], ovars[key]) - 1))
            if reldiff >= reltol:
                return False
        return True

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def diff(self, other, showvars=None, *, jsondiff=False, filename="solution.json"
             ,constraintsdiff=True, senssdiff=False, sensstol=0.1,
             absdiff=False, abstol=0.1, reldiff=True, reltol=1.0,
             sortmodelsbysenss=True, **tableargs):
        """Outputs differences between this solution and another

        Arguments
        ---------
        other : solution or string
            strings will be treated as paths to pickled solutions
        senssdiff : boolean
            if True, show sensitivity differences
        sensstol : float
            the smallest sensitivity difference worth showing
        absdiff : boolean
            if True, show absolute differences
        abstol : float
            the smallest absolute difference worth showing
        reldiff : boolean
            if True, show relative differences
        reltol : float
            the smallest relative difference worth showing

        Returns
        -------
        str
        """
        if sortmodelsbysenss:
            tableargs["sortmodelsbysenss"] = self["sensitivities"]["models"]
        else:
            tableargs["sortmodelsbysenss"] = False
        tableargs.update({"hidebelowminval": True, "sortbyvals": True,
                          "skipifempty": False})
        if isinstance(other, Strings):
            if other[-4:] == ".pgz":
                other = SolutionArray.decompress_file(other)
            else:
                other = pickle.load(open(other, "rb"))
        svars, ovars = self["variables"], other["variables"]
        lines = ["Solution Diff",
                 "=============",
                 "(argument is the baseline solution)", ""]
        svks, ovks = set(svars), set(ovars)
        if showvars:
            lines[0] += " (for selected variables)"
            lines[1] += "========================="
            showvars = self._parse_showvars(showvars)
            svks = {k for k in showvars if k in svars}
            ovks = {k for k in showvars if k in ovars}
        if constraintsdiff and other.modelstr and self.modelstr:
            if self.modelstr == other.modelstr:
                lines += ["** no constraint differences **", ""]
            else:
                cdiff = ["Constraint Differences",
                         "**********************"]
                cdiff.extend(list(difflib.unified_diff(
                    other.modelstr.split("\n"), self.modelstr.split("\n"),
                    lineterm="", n=3))[2:])
                cdiff += ["", "**********************", ""]
                lines += cdiff
        if svks - ovks:
            lines.append("Variable(s) of this solution"
                         " which are not in the argument:")
            lines.append("\n".join("  %s" % key for key in svks - ovks))
            lines.append("")
        if ovks - svks:
            lines.append("Variable(s) of the argument"
                         " which are not in this solution:")
            lines.append("\n".join("  %s" % key for key in ovks - svks))
            lines.append("")
        sharedvks = svks.intersection(ovks)

        # retrieve diff data
        diff_dict = diff_retrieval(self, other, sharedvks, showvars=showvars,
                                  jsondiff=jsondiff, senssdiff=senssdiff,
                                  absdiff=absdiff, reldiff=reldiff)
        if jsondiff:
            with open(filename, "w") as f:
                json.dump(diff_dict, f)
            return diff_dict

        if reldiff:
            rel_diff = diff_dict['rel']
            lines += var_table(rel_diff,
                               "Relative Differences |above %g%%|" % reltol,
                               valfmt="%+.1f%%  ", vecfmt="%+6.1f%% ",
                               minval=reltol, printunits=False, **tableargs)
            if lines[-2][:10] == "-"*10:  # nothing larger than reltol
                lines.insert(-1, ("The largest is %+g%%."
                                  % unrolled_absmax(rel_diff.values())))
        if absdiff:
            abs_diff = diff_dict['abs']
            lines += var_table(abs_diff,
                               "Absolute Differences |above %g|" % abstol,
                               valfmt="%+.2g", vecfmt="%+8.2g",
                               minval=abstol, **tableargs)
            if lines[-2][:10] == "-"*10:  # nothing larger than abstol
                lines.insert(-1, ("The largest is %+g."
                                  % unrolled_absmax(abs_diff.values())))
        if senssdiff:
            senss_delta = diff_dict['sens']
            lines += var_table(senss_delta,
                               "Sensitivity Differences |above %g|" % sensstol,
                               valfmt="%+-.2f  ", vecfmt="%+-6.2f",
                               minval=sensstol, printunits=False, **tableargs)
            if lines[-2][:10] == "-"*10:  # nothing larger than sensstol
                lines.insert(-1, ("The largest is %+g."
                                  % unrolled_absmax(senss_delta.values())))
        return "\n".join(lines)

    def save(self, filename="solution.pkl",
             *, saveconstraints=True, **pickleargs):
        """Pickles the solution and saves it to a file.

        Solution can then be loaded with e.g.:
        >>> import pickle
        >>> pickle.load(open("solution.pkl"))
        """
        with SolSavingEnvironment(self, saveconstraints):
            pickle.dump(self, open(filename, "wb"), **pickleargs)

    def save_compressed(self, filename="solution.pgz",
                        *, saveconstraints=True, **cpickleargs):
        "Pickle a file and then compress it into a file with extension."
        with gzip.open(filename, "wb") as f:
            with SolSavingEnvironment(self, saveconstraints):
                pickled = pickle.dumps(self, **cpickleargs)
            f.write(pickletools.optimize(pickled))

    @staticmethod
    def decompress_file(file):
        "Load a gzip-compressed pickle file"
        with gzip.open(file, "rb") as f:
            return pickle.Unpickler(f).load()

    def varnames(self, showvars, exclude):
        "Returns list of variables, optionally with minimal unique names"
        if showvars:
            showvars = self._parse_showvars(showvars)
        self.set_necessarylineage()
        names = {}
        for key in showvars or self["variables"]:
            for k in self["variables"].keymap[key]:
                names[k.str_without(exclude)] = k
        self.set_necessarylineage(clear=True)
        return names

    def savemat(self, filename="solution.mat", *, showvars=None,
                excluded=("vec")):
        "Saves primal solution as matlab file"
        from scipy.io import savemat
        savemat(filename,
                {name.replace(".", "_"): np.array(self["variables"][key], "f")
                 for name, key in self.varnames(showvars, excluded).items()})

    def todataframe(self, showvars=None, excluded=("vec")):
        "Returns primal solution as pandas dataframe"
        import pandas as pd  # pylint:disable=import-error
        rows = []
        cols = ["Name", "Index", "Value", "Units", "Label",
                "Lineage", "Other"]
        for _, key in sorted(self.varnames(showvars, excluded).items(),
                             key=lambda k: k[0]):
            value = self["variables"][key]
            if key.shape:
                idxs = []
                it = np.nditer(np.empty(value.shape), flags=['multi_index'])
                while not it.finished:
                    idx = it.multi_index
                    idxs.append(idx[0] if len(idx) == 1 else idx)
                    it.iternext()
            else:
                idxs = [None]
            for idx in idxs:
                row = [
                    key.name,
                    "" if idx is None else idx,
                    value if idx is None else value[idx]]
                rows.append(row)
                row.extend([
                    key.unitstr(),
                    key.label or "",
                    key.lineage or "",
                    ", ".join("%s=%s" % (k, v) for (k, v) in key.descr.items()
                              if k not in ["name", "units", "unitrepr",
                                           "idx", "shape", "veckey",
                                           "value", "vecfn",
                                           "lineage", "label"])])
        return pd.DataFrame(rows, columns=cols)

    def savetxt(self, filename="solution.txt", *, printmodel=True, **kwargs):
        "Saves solution table as a text file"
        with open(filename, "w") as f:
            if printmodel:
                f.write(self.modelstr + "\n")
            f.write(self.table(**kwargs))

    def savejson(self, filename="solution.json", showvars=None):
        "Saves solution table as a json file"
        sol_dict = {}
        if self._lineageset:
            self.set_necessarylineage(clear=True)
        data = self["variables"]
        if showvars:
            showvars = self._parse_showvars(showvars)
            data = {k: data[k] for k in showvars if k in data}
        # add appropriate data for each variable to the dictionary
        for k, v in data.items():
            key = str(k)
            if isinstance(v, np.ndarray):
                val = {"v": v.tolist(), "u": k.unitstr()}
            else:
                val = {"v": v, "u": k.unitstr()}
            sol_dict[key] = val
        with open(filename, "w") as f:
            json.dump(sol_dict, f)

    def savecsv(self, filename="solution.csv", *, valcols=5, showvars=None):
        "Saves primal solution as a CSV sorted by modelname, like the tables."
        data = self["variables"]
        if showvars:
            showvars = self._parse_showvars(showvars)
            data = {k: data[k] for k in showvars if k in data}
        # if the columns don't capture any dimensions, skip them
        minspan, maxspan = None, 1
        for v in data.values():
            if getattr(v, "shape", None) and any(di != 1 for di in v.shape):
                minspan_ = min((di for di in v.shape if di != 1))
                maxspan_ = max((di for di in v.shape if di != 1))
                if minspan is None or minspan_ < minspan:
                    minspan = minspan_
                if maxspan is None or maxspan_ > maxspan:
                    maxspan = maxspan_
        if minspan is not None and minspan > valcols:
            valcols = 1
        if maxspan < valcols:
            valcols = maxspan
        lines = var_table(data, "", rawlines=True, maxcolumns=valcols,
                          tables=("cost", "sweepvariables", "freevariables",
                                  "constants", "sensitivities"))
        with open(filename, "w") as f:
            f.write("Model Name,Variable Name,Value(s)" + ","*valcols
                    + "Units,Description\n")
            for line in lines:
                if line[0] == ("newmodelline",):
                    f.write(line[1])
                elif not line[1]:  # spacer line
                    f.write("\n")
                else:
                    f.write("," + line[0].replace(" : ", "") + ",")
                    vals = line[1].replace("[", "").replace("]", "").strip()
                    for el in vals.split():
                        f.write(el + ",")
                    f.write(","*(valcols - len(vals.split())))
                    f.write((line[2].replace("[", "").replace("]", "").strip()
                             + ","))
                    f.write(line[3].strip() + "\n")

    def subinto(self, posy):
        "Returns NomialArray of each solution substituted into posy."
        if posy in self["variables"]:
            return self["variables"](posy)

        if not hasattr(posy, "sub"):
            raise ValueError("no variable '%s' found in the solution" % posy)

        if len(self) > 1:
            return NomialArray([self.atindex(i).subinto(posy)
                                for i in range(len(self))])

        return posy.sub(self["variables"], require_positive=False)

    def _parse_showvars(self, showvars):
        showvars_out = set()
        for k in showvars:
            k, _ = self["variables"].parse_and_index(k)
            keys = self["variables"].keymap[k]
            showvars_out.update(keys)
        return showvars_out

    def summary(self, showvars=(), **kwargs):
        "Print summary table, showing no sensitivities or constants"
        return self.table(showvars,
                          ["cost breakdown", "model sensitivities breakdown",
                           "warnings", "sweepvariables", "freevariables"],
                          **kwargs)

    def table(self, showvars=(),
              tables=("cost breakdown", "model sensitivities breakdown",
                      "warnings", "sweepvariables", "freevariables",
                      "constants", "sensitivities", "tightest constraints"),
              sortmodelsbysenss=False, **kwargs):
        """A table representation of this SolutionArray

        Arguments
        ---------
        tables: Iterable
            Which to print of ("cost", "sweepvariables", "freevariables",
                               "constants", "sensitivities")
        fixedcols: If true, print vectors in fixed-width format
        latex: int
            If > 0, return latex format (options 1-3); otherwise plain text
        included_models: Iterable of strings
            If specified, the models (by name) to include
        excluded_models: Iterable of strings
            If specified, model names to exclude

        Returns
        -------
        str
        """
        if sortmodelsbysenss and "sensitivities" in self:
            kwargs["sortmodelsbysenss"] = self["sensitivities"]["models"]
        else:
            kwargs["sortmodelsbysenss"] = False
        varlist = list(self["variables"])
        has_only_one_model = True
        for var in varlist[1:]:
            if var.lineage != varlist[0].lineage:
                has_only_one_model = False
                break
        if has_only_one_model:
            kwargs["sortbymodel"] = False
        self.set_necessarylineage()
        showvars = self._parse_showvars(showvars)
        strs = []
        for table in tables:
            if "breakdown" in table:
                if (len(self) > 1 or not UNICODE_EXPONENTS
                        or "sensitivities" not in self):
                    # no breakdowns for sweeps or no-unicode environments
                    table = table.replace(" breakdown", "")
            if "sensitivities" not in self and ("sensitivities" in table or
                                                "constraints" in table):
                continue
            if table == "cost":
                cost = self["cost"]  # pylint: disable=unsubscriptable-object
                if kwargs.get("latex", None):  # cost is not printed for latex
                    continue
                strs += ["\n%s\n------------" % "Optimal Cost"]
                if len(self) > 1:
                    costs = ["%-8.3g" % c for c in mag(cost[:4])]
                    strs += [" [ %s %s ]" % ("  ".join(costs),
                                             "..." if len(self) > 4 else "")]
                else:
                    strs += [" %-.4g" % mag(cost)]
                strs[-1] += unitstr(cost, into=" [%s]", dimless="")
                strs += [""]
            elif table in TABLEFNS:
                strs += TABLEFNS[table](self, showvars, **kwargs)
            elif table in self:
                data = self[table]
                if showvars:
                    showvars = self._parse_showvars(showvars)
                    data = {k: data[k] for k in showvars if k in data}
                strs += var_table(data, self.table_titles[table], **kwargs)
        if kwargs.get("latex", None):
            preamble = "\n".join(("% \\documentclass[12pt]{article}",
                                  "% \\usepackage{booktabs}",
                                  "% \\usepackage{longtable}",
                                  "% \\usepackage{amsmath}",
                                  "% \\begin{document}\n"))
            strs = [preamble] + strs + ["% \\end{document}"]
        self.set_necessarylineage(clear=True)
        return "\n".join(strs)

    def plot(self, posys=None, axes=None):
        "Plots a sweep for each posy"
        if len(self["sweepvariables"]) != 1:
            print("SolutionArray.plot only supports 1-dimensional sweeps")
        if not hasattr(posys, "__len__"):
            posys = [posys]
        import matplotlib.pyplot as plt
        from .interactive.plot_sweep import assign_axes
        from . import GPBLU
        (swept, x), = self["sweepvariables"].items()
        posys, axes = assign_axes(swept, posys, axes)
        for posy, ax in zip(posys, axes):
            y = self(posy) if posy not in [None, "cost"] else self["cost"]
            ax.plot(x, y, color=GPBLU)
        if len(axes) == 1:
            axes, = axes
        return plt.gcf(), axes


# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def var_table(data, title, *, printunits=True, latex=False, rawlines=False,
              varfmt="%s : ", valfmt="%-.4g ", vecfmt="%-8.3g",
              minval=0, sortbyvals=False, hidebelowminval=False,
              included_models=None, excluded_models=None, sortbymodel=True,
              maxcolumns=5, skipifempty=True, sortmodelsbysenss=None, **_):
    """
    Pretty string representation of a dict of VarKeys
    Iterable values are handled specially (partial printing)

    Arguments
    ---------
    data : dict whose keys are VarKey's
        data to represent in table
    title : string
    printunits : bool
    latex : int
        If > 0, return latex format (options 1-3); otherwise plain text
    varfmt : string
        format for variable names
    valfmt : string
        format for scalar values
    vecfmt : string
        format for vector values
    minval : float
        skip values with all(abs(value)) < minval
    sortbyvals : boolean
        If true, rows are sorted by their average value instead of by name.
    included_models : Iterable of strings
        If specified, the models (by name) to include
    excluded_models : Iterable of strings
        If specified, model names to exclude
    """
    if not data:
        return []
    decorated, models = [], set()
    for i, (k, v) in enumerate(data.items()):
        if np.isnan(v).all() or np.nanmax(np.abs(v)) <= minval:
            continue  # no values below minval
        if minval and hidebelowminval and getattr(v, "shape", None):
            v[np.abs(v) <= minval] = np.nan
        model = lineagestr(k.lineage) if sortbymodel else ""
        if not sortmodelsbysenss:
            msenss = 0
        else:  # sort should match that in msenss_table above
            msenss = -round(np.mean(sortmodelsbysenss.get(model, 0)), 4)
        models.add(model)
        b = bool(getattr(v, "shape", None))
        s = k.str_without(("lineage", "vec"))
        if not sortbyvals:
            decorated.append((msenss, model, b, (varfmt % s), i, k, v))
        else:  # for consistent sorting, add small offset to negative vals
            val = np.nanmean(np.abs(v)) - (1e-9 if np.nanmean(v) < 0 else 0)
            sort = (float("%.4g" % -val), k.name)
            decorated.append((model, sort, msenss, b, (varfmt % s), i, k, v))
    if not decorated and skipifempty:
        return []
    if included_models:
        included_models = set(included_models)
        included_models.add("")
        models = models.intersection(included_models)
    if excluded_models:
        models = models.difference(excluded_models)
    decorated.sort()
    previous_model, lines = None, []
    for varlist in decorated:
        if sortbyvals:
            model, _, msenss, isvector, varstr, _, var, val = varlist
        else:
            msenss, model, isvector, varstr, _, var, val = varlist
        if model not in models:
            continue
        if model != previous_model:
            if lines:
                lines.append(["", "", "", ""])
            if model:
                if not latex:
                    lines.append([("newmodelline",), model, "", ""])
                else:
                    lines.append(
                        [r"\multicolumn{3}{l}{\textbf{" + model + r"}} \\"])
            previous_model = model
        label = var.descr.get("label", "")
        units = var.unitstr(" [%s] ") if printunits else ""
        if not isvector:
            valstr = valfmt % val
        else:
            last_dim_index = len(val.shape)-1
            horiz_dim, ncols = last_dim_index, 1  # starting values
            for dim_idx, dim_size in enumerate(val.shape):
                if ncols <= dim_size <= maxcolumns:
                    horiz_dim, ncols = dim_idx, dim_size
            # align the array with horiz_dim by making it the last one
            dim_order = list(range(last_dim_index))
            dim_order.insert(horiz_dim, last_dim_index)
            flatval = val.transpose(dim_order).flatten()
            vals = [vecfmt % v for v in flatval[:ncols]]
            bracket = " ] " if len(flatval) <= ncols else ""
            valstr = "[ %s%s" % ("  ".join(vals), bracket)
        for before, after in VALSTR_REPLACES:
            valstr = valstr.replace(before, after)
        if not latex:
            lines.append([varstr, valstr, units, label])
            if isvector and len(flatval) > ncols:
                values_remaining = len(flatval) - ncols
                while values_remaining > 0:
                    idx = len(flatval)-values_remaining
                    vals = [vecfmt % v for v in flatval[idx:idx+ncols]]
                    values_remaining -= ncols
                    valstr = "  " + "  ".join(vals)
                    for before, after in VALSTR_REPLACES:
                        valstr = valstr.replace(before, after)
                    if values_remaining <= 0:
                        spaces = (-values_remaining
                                  * len(valstr)//(values_remaining + ncols))
                        valstr = valstr + "  ]" + " "*spaces
                    lines.append(["", valstr, "", ""])
        else:
            varstr = "$%s$" % varstr.replace(" : ", "")
            if latex == 1:  # normal results table
                lines.append([varstr, valstr, "$%s$" % var.latex_unitstr(),
                              label])
                coltitles = [title, "Value", "Units", "Description"]
            elif latex == 2:  # no values
                lines.append([varstr, "$%s$" % var.latex_unitstr(), label])
                coltitles = [title, "Units", "Description"]
            elif latex == 3:  # no description
                lines.append([varstr, valstr, "$%s$" % var.latex_unitstr()])
                coltitles = [title, "Value", "Units"]
            else:
                raise ValueError("Unexpected latex option, %s." % latex)
    if rawlines:
        return lines
    if not latex:
        if lines:
            maxlens = np.max([list(map(len, line)) for line in lines
                              if line[0] != ("newmodelline",)], axis=0)
            dirs = [">", "<", "<", "<"]
            # check lengths before using zip
            assert len(list(dirs)) == len(list(maxlens))
            fmts = ["{0:%s%s}" % (direc, L) for direc, L in zip(dirs, maxlens)]
        for i, line in enumerate(lines):
            if line[0] == ("newmodelline",):
                line = [fmts[0].format(" | "), line[1]]
            else:
                line = [fmt.format(s) for fmt, s in zip(fmts, line)]
            lines[i] = "".join(line).rstrip()
        lines = [title] + ["-"*len(title)] + lines + [""]
    else:
        colfmt = {1: "llcl", 2: "lcl", 3: "llc"}
        lines = (["\n".join(["{\\footnotesize",
                             "\\begin{longtable}{%s}" % colfmt[latex],
                             "\\toprule",
                             " & ".join(coltitles) + " \\\\ \\midrule"])] +
                 [" & ".join(l) + " \\\\" for l in lines] +
                 ["\n".join(["\\bottomrule", "\\end{longtable}}", ""])])
    return lines
