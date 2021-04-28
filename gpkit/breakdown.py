# pylint: skip-file
import string
from collections import defaultdict, namedtuple
from gpkit.nomials import Monomial, Posynomial
from gpkit.nomials.map import NomialMap
from gpkit.small_scripts import mag
from gpkit.small_classes import FixedScalar, HashVector
from gpkit.exceptions import DimensionalityError
from gpkit.repr_conventions import unitstr as get_unitstr
import numpy as np


Transform = namedtuple("Transform", ["factor", "power", "origkey"])
Tree = namedtuple("Tree", ["key", "value", "branches"])

def get_free_vks(posy, solution):
    "Returns all free vks of a given posynomial for a given solution"
    return set(vk for vk in posy.vks if vk not in solution["constants"])

def get_model_breakdown(solution):
    breakdowns = {"|sensitivity|": 0}
    for modelname, senss in solution["sensitivities"]["models"].items():
        senss = abs(senss)  # for those monomial equalities
        *namespace, name = modelname.split(".")
        subbd = breakdowns
        subbd["|sensitivity|"] += senss
        for parent in namespace:
            if parent not in subbd:
                subbd[parent] = {parent: {}}
            subbd = subbd[parent]
            if "|sensitivity|" not in subbd:
                subbd["|sensitivity|"] = 0
            subbd["|sensitivity|"] += senss
        subbd[name] = {"|sensitivity|": senss}
    print(breakdowns["HyperloopSystem"]["|sensitivity|"])
    breakdowns = {"|sensitivity|": 0}
    for constraint, senss in solution["sensitivities"]["constraints"].items():
        senss = abs(senss)  # for those monomial
        if senss <= 1e-5:
            continue
        subbd = breakdowns
        subbd["|sensitivity|"] += senss
        for parent in constraint.lineagestr().split("."):
            if parent not in subbd:
                subbd[parent] = {}
            subbd = subbd[parent]
            if "|sensitivity|" not in subbd:
                subbd["|sensitivity|"] = 0
            subbd["|sensitivity|"] += senss
        # treat vectors as namespace
        subbd[constraint] = {"|sensitivity|": senss}
    for vk, senss in solution["sensitivities"]["variables"].items():
        senss = abs(senss)
        if hasattr(senss, "shape"):
            senss = np.nansum(senss)
        if senss <= 1e-5:
            continue
        subbd = breakdowns
        subbd["|sensitivity|"] += senss
        for parent in vk.lineagestr().split("."):
            if parent not in subbd:
                subbd[parent] = {}
            subbd = subbd[parent]
            if "|sensitivity|" not in subbd:
                subbd["|sensitivity|"] = 0
            subbd["|sensitivity|"] += senss
        # treat vectors as namespace (indexing vectors above)
        subbd[vk] = {"|sensitivity|": senss}
    print(breakdowns["|sensitivity|"])
    return breakdowns

def crawl_modelbd(bd, tree=None, name="Model"):
    if tree is None:
        tree = []
    subtree = []
    tree.append(Tree(name, bd.pop("|sensitivity|"), subtree))
    for key, _ in sorted(bd.items(), key=lambda kv: kv[1]["|sensitivity|"], reverse=True):
        crawl_modelbd(bd[key], subtree, key)
    return tree[0]

def get_breakdowns(solution):
    """Returns {key: (lt, gt, constraint)} for breakdown constrain in solution.

    A breakdown constraint is any whose "gt" contains a single free variable.

    (At present, monomial constraints check both sides as "gt")
    """
    breakdowns = defaultdict(list)
    for constraint, senss in solution["sensitivities"]["constraints"].items():
        if abs(senss) <= 1e-5:  # only tight-ish ones
            continue
        if constraint.oper == ">=":
            gt, lt = (constraint.left, constraint.right)
        elif constraint.oper == "<=":
            lt, gt = (constraint.left, constraint.right)
        elif constraint.oper == "=":
            if senss > 0:  # l_over_r is more sensitive - see nomials/math.py
                lt, gt = [constraint.left, constraint.right]
            else:  # r_over_l is more sensitive - see nomials/math.py
                gt, lt = [constraint.left, constraint.right]
        if lt.any_nonpositive_cs:
            continue  # no signomials
        freegt_vks = get_free_vks(gt, solution)
        if len(freegt_vks) != 1:
            continue  # not a breakdown constraint
        brokendownvk, = freegt_vks
        if gt.exp[brokendownvk] < 0:
            if constraint.oper == "=" or len(lt.hmap) != 1:
                continue
            # We can try flipping gt/lt to make a breakdown.
            freelt_vks = get_free_vks(lt, solution)
            if len(lt.hmap) != 1 or len(freelt_vks) != 1:
                continue
            brokendownvk, = freelt_vks
            if lt.exp[brokendownvk] > 0:
                continue  # not a breakdown constraint after transformation
            gt, lt = 1/lt, 1/gt
        breakdowns[brokendownvk].append((lt, gt, constraint))
    breakdowns = dict(breakdowns)  # remove the defaultdict-ness
    for key, bds in breakdowns.items():
        # TODO: do multiple if sensitivities are quite close? right now we have to break ties!
        if len(bds) > 1:
            bds.sort(key=lambda lgc: (abs(solution["sensitivities"]["constraints"][lgc[2]]), str(lgc[0])), reverse=True)

    prevlen = None
    while len(BASICALLY_FIXED_VARIABLES) != prevlen:
        prevlen = len(BASICALLY_FIXED_VARIABLES)
        for key in breakdowns:
            if key not in BASICALLY_FIXED_VARIABLES:
                get_fixity(key, breakdowns, solution, BASICALLY_FIXED_VARIABLES)
    return breakdowns

BASICALLY_FIXED_VARIABLES = set()


def get_fixity(key, bd, solution, basically_fixed=set(), visited=set()):
    lt, gt, _ = bd[key][0]
    free_vks = get_free_vks(lt, solution).union(get_free_vks(gt, solution))
    for vk in free_vks:
        if vk not in bd:
            return
        if vk in BASICALLY_FIXED_VARIABLES:
            continue
        if vk is key:
            continue
        if vk in visited:  # been here before and it's not me
            return
        visited.add(key)
        get_fixity(vk, bd, solution, basically_fixed, visited)
        if vk not in BASICALLY_FIXED_VARIABLES:
            return
    basically_fixed.add(key)

def crawl(key, bd, solution, basescale=1, permissivity=2, verbosity=0,
          visited_bdkeys=None):
    "Returns the tree of breakdowns of key in bd, sorting by solution's values"
    if key in bd:
        # TODO: do multiple if sensitivities are quite close?
        composition, keymon, constraint = bd[key][0]
    elif isinstance(key, Posynomial):
        composition = key
        keymon = None
    else:
        raise TypeError("the `key` argument must be a VarKey or Posynomial.")

    if visited_bdkeys is None:
        visited_bdkeys = set()
    if verbosity:
        indent = verbosity-1  # HACK: a bit of overloading, here
        keyvalstr = "%s (%s)" % (key.str_without(["lineage", "units"]),
                                 get_valstr(key, solution))
        print("  "*indent + keyvalstr + ", which breaks down further")
        indent += 1
    orig_subtree = subtree = []
    tree = Tree(key, basescale, subtree)
    visited_bdkeys.add(key)
    if keymon is None:
        scale = solution(key)/basescale
    else:
        free_vks = get_free_vks(keymon, solution)
        subkey, = free_vks
        power = keymon.exp[subkey]
        fixed_vks = set(keymon.vks) - free_vks
        scale = solution(key)**power/basescale
        # TODO: make method that can handle both kinds of transforms
        if power != 1 or fixed_vks or mag(keymon.c) != 1 or keymon.units != key.units:
            units = 1
            exp = HashVector()
            for vk in free_vks:
                exp[vk] = keymon.exp[vk]
                if vk.units:
                    units *= vk.units**keymon.exp[vk]
            subhmap = NomialMap({exp: 1})
            subhmap.units = None if units == 1 else units
            freemon = Monomial(subhmap)
            factor = Monomial(keymon/freemon)
            scale = scale * solution(factor)
            if factor != 1:
                factor = factor**(-1/power)  # invert the transform
                factor.ast = None
                if verbosity:
                    keyvalstr = "%s (%s)" % (factor.str_without(["lineage", "units"]),
                                             get_valstr(factor, solution))
                    print("  "*indent + "(with a factor of " + keyvalstr + " )")
                subsubtree = []
                transform = Transform(factor, 1, keymon)
                orig_subtree.append(Tree(transform, basescale, subsubtree))
                orig_subtree = subsubtree
            if power != 1:
                if verbosity:
                    print("  "*indent + "(with a power of %.2g )" % power)
                subsubtree = []
                transform = Transform(1, 1/power, keymon)  # inverted bc it's on the gt side
                orig_subtree.append(Tree(transform, basescale, subsubtree))
                orig_subtree = subsubtree
    if verbosity:
        if keymon is not None:
            print("  "*indent + "in: " + constraint.str_without(["units", "lineage"]))
        print("  "*indent + "by:")
        indent += 1

    # TODO: use ast_parsing instead of chop?
    monsols = [solution(mon) for mon in composition.chop()]
    parsed_monsols = [getattr(mon, "value", mon) for mon in monsols]
    monvals = [float(mon/scale) for mon in parsed_monsols]
    # sort by value, preserving order in case of value tie
    sortedmonvals = sorted(zip(monvals, range(len(monvals)),
                               composition.chop()), reverse=True)
    for scaledmonval, _, mon in sortedmonvals:
        if not scaledmonval:
            continue
        scaledmonval = min(1, scaledmonval)  # clip it
        subtree = orig_subtree  # revert back to the original subtree

        # time for some filtering
        free_vks = get_free_vks(mon, solution)
        basically_fixed_vks = {vk for vk in free_vks
                               if vk in BASICALLY_FIXED_VARIABLES}
        if free_vks - basically_fixed_vks:  # don't remove the last one
            free_vks = free_vks - basically_fixed_vks

        if scaledmonval > 1 - permissivity:
            unbreakdownable_vks = {vk for vk in free_vks if vk not in bd}
            if free_vks - unbreakdownable_vks:  # don't remove the last one
                free_vks = free_vks - unbreakdownable_vks

        if len(free_vks) > 1 and permissivity > 1:
            best_vks = sorted((vk for vk in free_vks if vk in bd),
                key=lambda vk:
                    (abs(solution["sensitivities"]["constraints"][bd[vk][0][2]]),
                     str(bd[vk][0][0])), reverse=True)
            if best_vks:
                free_vks = set([best_vks[0]])
        fixed_vks = mon.vks - free_vks

        if len(free_vks) == 1:
            subkey, = free_vks
            power = mon.exp[subkey]
            if power != 1 and subkey not in bd:
                power = 1  # no need for a transform
        else:
            subkey = None
            power = 1
            if not free_vks:
                # prioritize showing some fixed_vks as if they were "free"
                if len(fixed_vks) == 1:
                    free_vks = fixed_vks
                    fixed_vks = set()
                else:
                    for vk in list(fixed_vks):
                        if vk.units and not vk.units.dimensionless:
                            free_vks.add(vk)
                            fixed_vks.remove(vk)

        if free_vks and (fixed_vks or mag(mon.c) != 1):
            if subkey:
                kindafree_vks = set(vk for vk in fixed_vks
                                    if vk not in solution["constants"])
                if kindafree_vks == fixed_vks:
                    kindafree_vks = set()  # don't remove ALL of them
                else:
                    free_vks.update(kindafree_vks)
            units = 1
            exp = HashVector()
            for vk in free_vks:
                exp[vk] = mon.exp[vk]
                if vk.units:
                    units *= vk.units**mon.exp[vk]
            subhmap = NomialMap({exp: 1})
            subhmap.units = None if units is 1 else units
            freemon = Monomial(subhmap)
            factor = mon/freemon  # autoconvert...
            if (factor.units is None and isinstance(factor, FixedScalar)
                    and abs(factor.value - 1) <= 1e-4):
                factor = 1  # minor fudge to clear numerical inaccuracies
            if factor != 1 :
                factor.ast = None
                if verbosity:
                    keyvalstr = "%s (%s)" % (factor.str_without(["lineage", "units"]),
                                             get_valstr(factor, solution))
                    print("  "*indent + "(with a factor of %s )" % keyvalstr)
                subsubtree = []
                transform = Transform(factor, 1, mon)
                subtree.append(Tree(transform, scaledmonval, subsubtree))
                subtree = subsubtree
            mon = freemon  # simplifies units
            if subkey and fixed_vks and kindafree_vks:
                units = 1
                exp = HashVector()
                for vk in kindafree_vks:
                    exp[vk] = mon.exp[vk]
                    if vk.units:
                        units *= vk.units**mon.exp[vk]
                subhmap = NomialMap({exp: 1})
                subhmap.units = None if units == 1 else units
                factor = Monomial(subhmap)
                if factor != 1:
                    factor.ast = None
                    if verbosity:
                        keyvalstr = "%s (%s)" % (factor.str_without(["lineage", "units"]),
                                                 get_valstr(factor, solution))
                        print("  "*indent + "(with a factor of " + keyvalstr + " )")
                    subsubtree = []
                    transform = Transform(factor, 1, mon)
                    subtree.append(Tree(transform, scaledmonval, subsubtree))
                    subtree = subsubtree
                    mon = mon/factor
                    mon.ast = None
            if power != 1:
                if verbosity:
                    print("  "*indent + "(with a power of %.2g )" % power)
                subsubtree = []
                transform = Transform(1, power, mon)
                subtree.append(Tree(transform, scaledmonval, subsubtree))
                subtree = subsubtree
                mon = mon**(1/power)
                mon.ast = None
        # TODO: make minscale an argument - currently an arbitrary 0.01
        if (subkey is not None and subkey not in visited_bdkeys
                and subkey in bd and scaledmonval > 0.01):
            if verbosity:
                verbosity = indent + 1  # slight hack
            try:
                subsubtree = crawl(subkey, bd, solution, scaledmonval,
                                   permissivity, verbosity, visited_bdkeys)
                subtree.append(subsubtree)
                continue
            except Exception as e:
                print(repr(e))

        if verbosity:
            keyvalstr = "%s (%s)" % (mon.str_without(["lineage", "units"]),
                                     get_valstr(mon, solution))
            print("  "*indent + keyvalstr)
        subtree.append(Tree(mon, scaledmonval, []))
    # clean up node combination - use a dictionary at the subtree level?
    ftidxs = defaultdict(list)
    for i, node in enumerate(orig_subtree):
        if isinstance(node.key, Transform):
            ftidxs[(node.key.factor, node.key.power)].append(i)
    to_delete = []
    to_insert = []
    for idxs in ftidxs.values():
        if len(idxs) > 1:
            valsum = 0
            newbranches = []
            new_origkeys = []
            for idx in idxs:
                key, val, subbranches = orig_subtree[idx]
                valsum += val
                new_origkeys.append((key.origkey, val))
                newbranches.extend(subbranches)
            to_delete.extend(idxs)
            newkey = Transform(key.factor, key.power, tuple(new_origkeys))
            to_insert.append(Tree(newkey, valsum, newbranches))
    for idx in sorted(to_delete, reverse=True): # high to low
        orig_subtree.pop(idx)
    if to_insert:
        orig_subtree.extend(to_insert)
        orig_subtree.sort(reverse=True, key=lambda branch: branch.value)
    return tree

SYMBOLS = string.ascii_uppercase + string.ascii_lowercase
for ambiguous_symbol in "lILT":
    SYMBOLS = SYMBOLS.replace(ambiguous_symbol, "")
SYMBOLS += "⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵"

def get_spanstr(legend, length, label, leftwards, solution):
    "Returns span visualization, collapsing labels to symbols"
    spacer, lend, rend = "│", "┯", "┷"
    if isinstance(label, Transform):
        spacer, lend, rend = "╎", "╤", "╧"
        if label.power != 1:
            spacer, lend, rend = "^", "^", "^"
        # remove origkey so they collide in the legends dictionary
        label = Transform(label.factor, label.power, None)
        # TODO: catch PI (or again could that come from AST parsing?)
        if label.power == 1 and len(str(label.factor)) == 1:
            legend[label] = str(label.factor)

    if label not in legend:
        shortname = SYMBOLS[len(legend)]
        legend[label] = shortname
    else:
        shortname = legend[label]

    if length <= 1:
        return shortname

    shortside = int(max(0, length - 2)/2)
    longside  = int(max(0, length - 3)/2)
    if leftwards:
        if length == 2:
            return lend + shortname
        return lend + spacer*shortside + shortname + spacer*longside + rend
    else:
        if length == 2:
            return shortname + rend
        # HACK: no corners on rightwards - only used for depth 0
        return "┃"*(longside+1) + shortname + "┃"*(shortside+1)

def layer(map, tree, extent, maxdepth, depth=0):
    "Turns the tree into a 2D-array"
    key, val, branches = tree
    if len(map) <= depth:
        map.append([])
    map[depth].append((key, extent))
    if depth > maxdepth or not val:
        return map
    scale = extent/val

    extents = [round(scale*node.value) for node in branches]
    for i, branch in enumerate(branches):
        k, v, bs = branch
        if isinstance(k, Transform):
            subscale = extents[i]/v
            if not any(round(subscale*subv) for _, subv, _ in bs):
                extents[i] = 0  # transform with no worthy heirs: misc it

    if not any(extents) or (extent == 1 and not isinstance(key, Transform)):
        branches = [Tree(None, val, [])]  # pad it out
        extents = [extent]
    elif not all(extents):  # create a catch-all
        branches = branches.copy()
        surplus = extent - sum(extents)
        miscvkeys, miscval = [], 0
        for subextent in reversed(extents):
            if not subextent or (branches[-1].value < miscval and surplus < 0):
                k, v, _ = branches.pop()
                if isinstance(k, Transform):
                    k = k.origkey  # TODO: this is the only use of origkey - remove it
                    if isinstance(k, tuple):
                        vkeys = [(-kv[1], str(kv[0]), kv[0]) for kv in k]
                if not isinstance(k, tuple):
                    vkeys = [(-v, str(k), k)]
                miscvkeys += vkeys
                surplus -= (round(scale*(miscval + v))
                            - round(scale*miscval) - subextent)
                miscval += v
        misckeys = tuple(k for _, _, k in sorted(miscvkeys))
        branches.append(Tree(misckeys, miscval, []))

    extents = [int(round(scale*node.value)) for node in branches]
    surplus = extent - sum(extents)
    if surplus:
        sign = int(np.sign(surplus))
        bump_priority = sorted((extent, sign*node.value, i)
                               for i, (node, extent)
                               in enumerate(zip(branches, extents)))
        while surplus:
            extents[bump_priority.pop()[-1]] += sign
            surplus -= sign
    if not isinstance(key, Transform):
        if len([ext for ext in extents if ext]) >= max(extent-1, 2):
            # if we'd branch a lot (all ones but one, or at all if extent <= 3)
            branches = [Tree(None, val, [])]
            extents = [extent]
    for branch, subextent in zip(branches, extents):
        if subextent:
            layer(map, branch, subextent, maxdepth, depth+1)
    return map

def plumb(tree, depth=0):
    "Finds maximum depth of a tree"
    maxdepth = depth
    for branch in tree.branches:
        maxdepth = max(maxdepth, plumb(branch, depth+1))
    return maxdepth

def graph(tree, solution, extent=None, maxdepth=None, collapse=True):
    "Prints breakdown"
    if maxdepth is None:
        maxdepth = plumb(tree) - 1
    if extent is None:  # autozoom in from 20
        prev_extent = None
        extent = 20
        while prev_extent != extent:
            mt = layer([], tree, extent, maxdepth)
            prev_extent = extent
            extent = min(extent, 4*max(len(at_depth) for at_depth in mt))
    else:
        mt = layer([], tree, extent, maxdepth)
    scale = 1/extent
    legend = {}
    chararray = np.full((len(mt), extent), "", "object")
    for depth, elements_at_depth in enumerate(mt):
        row = ""
        for i, (element, length) in enumerate(elements_at_depth):
            if element is None:
                row += " "*length
                continue
            leftwards = depth > 0 and length > 2
            row += get_spanstr(legend, length, element, leftwards, solution)
        if row.strip():
            chararray[depth, :] = list(row)

    # Format depth=0
    A_key, = [key for key, value in legend.items() if value == "A"]
    A_str = get_keystr(A_key, solution)
    A_valstr = get_valstr(A_key, solution, into="(%s)")
    fmt = "{0:>%s}" % (max(len(A_str), len(A_valstr)) + 3)
    for j, entry in enumerate(chararray[0,:]):
        if entry == "A":
            chararray[0,j] = fmt.format(A_str + "╺┫")
            chararray[0,j+1] = fmt.format(A_valstr + " ┃")
        else:
            chararray[0,j] = fmt.format(entry)

    # Format depths 1+
    labeled = set()
    new_legend = {}
    for pos in range(extent):
        for depth in reversed(range(1,len(mt))):
            value = chararray[depth, pos]
            if value == " ":  # spacer character
                chararray[depth, pos] = ""
                continue
            elif not value or value not in SYMBOLS:
                continue
            key, = [k for k, val in legend.items() if val == value]
            if getattr(key, "vks", None) and len(key.vks) == 1 and all(vk in new_legend for vk in key.vks):
                key, = key.vks
            if key not in new_legend and (isinstance(key, tuple) or (depth != len(mt) - 1 and chararray[depth+1, pos] != "")):
                new_legend[key] = SYMBOLS[len(new_legend)]
            if key in new_legend:
                chararray[depth, pos] = new_legend[key]
                if isinstance(key, tuple) and not isinstance(key, Transform):
                    chararray[depth, pos] =  "*" + chararray[depth, pos]
                if collapse:
                    continue
            tryup, trydn = True, True
            span = 0
            keystr = get_keystr(key, solution)
            if not collapse:
                fmt = "{0:<%s}" % (len(keystr) + 3)
            else:
                fmt = "{0:<1}"
            while tryup or trydn:
                span += 1
                if tryup:
                    if pos - span < 0:
                        tryup = False
                    else:
                        upchar = chararray[depth, pos-span]
                        if upchar == "│":
                            chararray[depth, pos-span] = fmt.format("┃")
                        elif upchar == "┯":
                            chararray[depth, pos-span] = fmt.format("┓")
                        else:
                            tryup = False
                if trydn:
                    if pos + span >= extent:
                        trydn = False
                    else:
                        dnchar = chararray[depth, pos+span]
                        if dnchar == "│":
                            chararray[depth, pos+span] = fmt.format("┃")
                        elif dnchar == "┷":
                            chararray[depth, pos+span] = fmt.format("┛")
                        else:
                            trydn = False
            #TODO: make submodels show up with this; bd should be an argument
            if key in bd or (hasattr(key, "vks") and key.vks and any(vk in bd for vk in key.vks)):
                linkstr = "┣┉"
            else:
                linkstr = "┣╸"
            if not (isinstance(key, FixedScalar) or keystr in labeled):
                labeled.add(keystr)
                valuestr = get_valstr(key, solution, into=" (%s)")
                if span > 1 and (pos + 2 >= extent or chararray[depth, pos+1] == "┃"):
                    chararray[depth, pos+1] += valuestr
                else:
                    keystr += valuestr
            chararray[depth, pos] = linkstr + keystr

    # Rotate and print
    vertstr = "\n".join("    " + "".join(row) for row in chararray.T.tolist())
    print()
    print(vertstr)
    print()
    legend = new_legend

    # Create and print legend
    if collapse:
        legend_lines = []
        for key, shortname in sorted(legend.items(), key=lambda kv: kv[1]):
            legend_lines.append(legend_entry(key, shortname, solution))
        maxlens = [max(len(el) for el in col) for col in zip(*legend_lines)]
        fmts = ["{0:<%s}" % L for L in maxlens]
        for line in legend_lines:
            line = "".join(fmt.format(cell)
                           for fmt, cell in zip(fmts, line) if cell).rstrip()
            print("    " + line)

def legend_entry(key, shortname, solution):
    "Returns list of legend elements"
    operator = note = ""
    keystr = valuestr = " "
    operator = "= " if shortname else "  + "
    if isinstance(key, Transform):
        if key.power == 1:
            operator = " ×"
            key = key.factor
            free, quasifixed = False, False
            if any(vk not in BASICALLY_FIXED_VARIABLES
                   for vk in get_free_vks(key, solution)):
                note = "  [free factor]"
        else:
            valuestr = "   ^%.3g" % key.power
            key = None
    if key is not None:
        if not isinstance(key, FixedScalar):
            keystr = get_keystr(key, solution)
        valuestr = get_valstr(key, solution, into="  "+operator+"%s")
    return ["%-4s" % shortname, keystr, valuestr, note]

def get_keystr(key, solution):
    if key is None:
        out = " "
    elif key is solution.costposy:
        out = "Cost"
    elif hasattr(key, "str_without"):
        out = key.str_without(["lineage", "units"])
    elif isinstance(key, tuple):
        return "(%i entries)" % len(key)
    else:
        out = str(key)
    # TODO: use fixedfactors to drop below 50
    if len(out) > 120:
        out = out[:120]+"…"
    return out

def get_valstr(key, solution, into="%s"):
    "Returns formatted string of the value of key in solution."
    try:
        value = solution(key)
    except (ValueError, TypeError):
        try:
            value = sum(solution(subkey) for subkey in key)
        except (ValueError, TypeError):
            return " "
    if isinstance(value, FixedScalar):
        value = value.value
    if hasattr(key, "unitstr"):
        unitstr = key.unitstr()
    else:
        if hasattr(value, "units"):
            value.ito_reduced_units()
        unitstr = get_unitstr(value)
    if unitstr[:2] == "1/":
        unitstr = "/" + unitstr[2:]
    value = mag(value)
    if 1e3 <= value < 1e6:
        valuestr = "{:,.0f}".format(value)
    else:
        valuestr = "%-.3g" % value
    return into % (valuestr + unitstr)


import pickle
from gpkit import ureg
ureg.define("pax = 1")
ureg.define("paxkm = km")
ureg.define("trip = 1")

print("STARTING...")

sol = pickle.load(open("bd.p", "rb"))

bd = get_breakdowns(sol)

mbd = get_model_breakdown(sol)
tree = crawl_modelbd(mbd)
graph(tree, sol, maxdepth=2, extent=20, collapse=False)

graph(tree.branches[0].branches[1],
      sol, maxdepth=2, extent=20, collapse=False)

graph(tree, sol, maxdepth=2, extent=20)

from gpkit.tests.helpers import StdoutCaptured

import difflib

# key, = [vk for vk in bd if "cruisevelocity[0]" in str(vk)]
# tree = crawl(key, bd, sol, permissivity=2, verbosity=1)
# graph(tree, sol)
#
# print("\n\nPERMISSIVITY = 0")
# tree = crawl(sol.costposy, bd, sol, permissivity=0)
# graph(tree, sol)
#
# print("\n\nPERMISSIVITY = 0.6")
# tree = crawl(sol.costposy, bd, sol, permissivity=0.6)
# graph(tree, sol)
#
# print("\n\nPERMISSIVITY = 1")
# tree = crawl(sol.costposy, bd, sol, permissivity=1)
# graph(tree, sol)
#
print("\n\nPERMISSIVITY = 2")
tree = crawl(sol.costposy, bd, sol, permissivity=2)
graph(tree, sol)

keys = sorted((key for key in bd.keys() if not key.idx or len(key.shape) == 1),
              key=lambda k: str(k))

permissivity = 2

# with StdoutCaptured("breakdowns.log"):
#     for key in keys:
#         tree = crawl(key, bd, sol, permissivity=permissivity)
#         graph(tree, sol)

# with StdoutCaptured("breakdowns.log.new"):
#     for key in keys:
#         tree = crawl(key, bd, sol, permissivity=permissivity)
#         graph(tree, sol)
#
# with open("breakdowns.log", "r") as original:
#     with open("breakdowns.log.new", "r") as new:
#         diff = difflib.unified_diff(
#             original.readlines(),
#             new.readlines(),
#             fromfile="original",
#             tofile="new",
#         )
#         for line in diff:
#             print(line[:-1])

print("DONE")
