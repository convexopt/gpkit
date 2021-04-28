"""Implement the SequentialGeometricProgram class"""
import warnings as pywarnings
from time import time
from collections import defaultdict
import numpy as np
from ..exceptions import (InvalidGPConstraint, Infeasible, UnnecessarySGP,
                          InvalidPosynomial, InvalidSGPConstraint)
from ..keydict import KeyDict
from ..nomials import Variable
from .gp import GeometricProgram
from ..nomials import PosynomialInequality, Posynomial
from .. import NamedVariables


EPS = 1e-6  # 1 +/- this is used in a few relative differences

# pylint: disable=too-many-instance-attributes
class SequentialGeometricProgram:
    """Prepares a collection of signomials for a SP solve.

    Arguments
    ---------
    cost : Posynomial
        Objective to minimize when solving
    constraints : list of Constraint or SignomialConstraint objects
        Constraints to maintain when solving (implicitly Signomials <= 1)
    verbosity : int (optional)
        Currently has no effect: SequentialGeometricPrograms don't know
        anything new after being created, unlike GeometricPrograms.

    Attributes with side effects
    ----------------------------
    `gps` is set during a solve
    `result` is set at the end of a solve

    Examples
    --------
    >>> gp = gpkit.geometric_program.SequentialGeometricProgram(
                        # minimize
                        x,
                        [   # subject to
                            1/x - y/x,  # <= 1, implicitly
                            y/10  # <= 1
                        ])
    >>> gp.solve()
    """
    gps = solver_outs = _results = result = model = None
    with NamedVariables("RelaxPCCP"):
        slack = Variable("C")

    def __init__(self, cost, model, substitutions, *,
                 use_pccp=True, pccp_penalty=2e2, checkbounds=True):
        self.pccp_penalty = pccp_penalty
        if cost.any_nonpositive_cs:
            raise InvalidPosynomial("""an SGP's cost must be Posynomial

    The equivalent of a Signomial objective can be constructed by constraining
    a dummy variable `z` to be greater than the desired Signomial objective `s`
    (z >= s) and then minimizing that dummy variable.""")
        self.gpconstraints, self.sgpconstraints = [], []
        if not use_pccp:
            self.slack = 1
        else:
            self.gpconstraints.append(self.slack >= 1)
        cost *= self.slack**pccp_penalty
        self.approxconstraints = []
        self.sgpvks = set()
        x0 = KeyDict(substitutions)
        x0.vks = model.vks  # for string access and so forth
        for cs in model.flat():
            try:
                if not hasattr(cs, "as_hmapslt1"):
                    raise InvalidGPConstraint(cs)
                if not isinstance(cs, PosynomialInequality):
                    cs.as_hmapslt1(substitutions)  # gp-compatible?
                self.gpconstraints.append(cs)
            except InvalidGPConstraint:
                if not hasattr(cs, "as_gpconstr"):
                    raise InvalidSGPConstraint(cs)
                self.sgpconstraints.append(cs)
                for hmaplt1 in cs.as_gpconstr(x0).as_hmapslt1({}):
                    constraint = (Posynomial(hmaplt1) <= self.slack)
                    constraint.generated_by = cs
                    self.approxconstraints.append(constraint)
                    self.sgpvks.update(constraint.vks)
        if not self.sgpconstraints:
            raise UnnecessarySGP("""Model valid as a Geometric Program.

SequentialGeometricPrograms should only be created with Models containing
Signomial Constraints, since Models without Signomials have global
solutions and can be solved with 'Model.solve()'.""")
        self._gp = GeometricProgram(
            cost, self.approxconstraints + self.gpconstraints,
            substitutions, checkbounds=checkbounds)
        self._gp.x0 = x0
        self.a_idxs = defaultdict(list)
        cost_mons = self._gp.k[0]
        sp_mons = sum(self._gp.k[:1+len(self.approxconstraints)])
        for row_idx, m_idx in enumerate(self._gp.A.row):
            if cost_mons <= m_idx <= sp_mons:
                self.a_idxs[self._gp.p_idxs[m_idx]].append(row_idx)

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def localsolve(self, solver=None, *, verbosity=1, x0=None, reltol=1e-4,
                   iteration_limit=50, **solveargs):
        """Locally solves a SequentialGeometricProgram and returns the solution.

        Arguments
        ---------
        solver : str or function (optional)
            By default uses one of the solvers found during installation.
            If set to "mosek", "mosek_cli", or "cvxopt", uses that solver.
            If set to a function, passes that function cs, A, p_idxs, and k.
        verbosity : int (optional)
            If greater than 0, prints solve time and number of iterations.
            Each GP is created and solved with verbosity one less than this, so
            if greater than 1, prints solver name and time for each GP.
        x0 : dict (optional)
            Initial location to approximate signomials about.
        reltol : float
            Iteration ends when this is greater than the distance between two
            consecutive solve's objective values.
        iteration_limit : int
            Maximum GP iterations allowed.
        mutategp: boolean
            Prescribes whether to mutate the previously generated GP
            or to create a new GP with every solve.
        **solveargs :
            Passed to solver function.

        Returns
        -------
        result : dict
            A dictionary containing the translated solver result.
        """
        self.gps, self.solver_outs, self._results = [], [], []
        starttime = time()
        if verbosity > 0:
            print("Starting a sequence of GP solves")
            print(" for %i free variables" % len(self.sgpvks))
            print("  in %i locally-GP constraints" % len(self.sgpconstraints))
            print("  and for %i free variables" % len(self._gp.varlocs))
            print("       in %i posynomial inequalities." % len(self._gp.k))
        prevcost, cost, rel_improvement = None, None, None
        while rel_improvement is None or rel_improvement > reltol:
            prevcost = cost
            if len(self.gps) > iteration_limit:
                raise Infeasible(
                    "Unsolved after %s iterations. Check `m.program.results`;"
                    " if they're converging, try `.localsolve(...,"
                    " iteration_limit=NEWLIMIT)`." % len(self.gps))
            gp = self.gp(x0, cleanx0=(len(self.gps) >= 1))  # clean the first x0
            self.gps.append(gp)  # NOTE: SIDE EFFECTS
            if verbosity > 1:
                print("\nGP Solve %i" % len(self.gps))
            if verbosity > 2:
                print("===============")
            solver_out = gp.solve(solver, verbosity=verbosity-1,
                                  gen_result=False, **solveargs)
            self.solver_outs.append(solver_out)
            cost = float(solver_out["objective"])
            x0 = dict(zip(gp.varlocs, np.exp(solver_out["primal"])))
            if verbosity > 2:
                result = gp.generate_result(solver_out, verbosity=verbosity-3)
                self._results.append(result)
                print(result.table(self.sgpvks))
            elif verbosity > 1:
                print("Solved cost was %.4g." % cost)
            if prevcost is None:
                continue
            rel_improvement = (prevcost - cost)/(prevcost + cost)
            if cost*(1 - EPS) > prevcost + EPS and verbosity > -1:
                pywarnings.warn(
                    "SGP not convergent: Cost rose by %.2g%% on GP solve %i."
                    " Details can be found in `m.program.results` or by"
                    " solving at a higher verbosity. Note that convergence"
                    " is not guaranteed for models with SignomialEqualities."
                    % (100*(cost - prevcost)/prevcost, len(self.gps)))
                rel_improvement = cost = None
        # solved successfully!
        self.result = gp.generate_result(solver_out, verbosity=verbosity-3)
        self.result["soltime"] = time() - starttime
        if verbosity > 1:
            print()
        if verbosity > 0:
            print("Solving took %.3g seconds and %i GP solves."
                  % (self.result["soltime"], len(self.gps)))
        if hasattr(self.slack, "key"):
            excess_slack = self.result["variables"][self.slack.key] - 1  # pylint: disable=no-member
            if excess_slack <= EPS:
                del self.result["freevariables"][self.slack.key]  # pylint: disable=no-member
                del self.result["variables"][self.slack.key]  # pylint: disable=no-member
                del self.result["sensitivities"]["variables"][self.slack.key]  # pylint: disable=no-member
                slcon = self.gpconstraints[0]
                slconsenss = self.result["sensitivities"]["constraints"][slcon]
                del self.result["sensitivities"]["constraints"][slcon]
                # TODO: create constraint in RelaxPCCP namespace
                self.result["sensitivities"]["models"][""] -= slconsenss
                if not self.result["sensitivities"]["models"][""]:
                    del self.result["sensitivities"]["models"][""]
            elif verbosity > -1:
                pywarnings.warn(
                    "Final solution let signomial constraints slacken by"
                    " %.2g%%. Calling .localsolve with a higher"
                    " `pccp_penalty` (it was %.3g this time) will reduce"
                    " final slack if the model is solvable with less. If"
                    " you think it might not be, check by generating an SP with"
                    "`use_pccp=False`, then solving with x0=(this model's final"
                    " solution); e.g. `m.sp(use_pccp=False).localsolve("
                    "x0=m.solution[\"variables\"])`.\n"
                    % (100*excess_slack, self.pccp_penalty))
        return self.result

    @property
    def results(self):
        "Creates and caches results from the raw solver_outs"
        if not self._results:
            self._results = [gp.generate_result(s_o, dual_check=False)
                             for gp, s_o in zip(self.gps, self.solver_outs)]
        return self._results

    def gp(self, x0=None, *, cleanx0=False):
        "Update self._gp for x0 and return it."
        if not x0:
            return self._gp  # return last generated
        if not cleanx0:
            cleanedx0 = KeyDict()
            cleanedx0.vks = self._gp.x0.vks
            cleanedx0.update(x0)
            x0 = cleanedx0
        self._gp.x0.update({vk: x0[vk] for vk in self.sgpvks if vk in x0})
        p_idx = 0
        for sgpc in self.sgpconstraints:
            for hmaplt1 in sgpc.as_gpconstr(self._gp.x0).as_hmapslt1({}):
                approxc = self.approxconstraints[p_idx]
                approxc.unsubbed = [Posynomial(hmaplt1)/self.slack]
                p_idx += 1  # p_idx=0 is the cost; sp constraints are after it
                hmap, = approxc.as_hmapslt1(self._gp.substitutions)
                self._gp.hmaps[p_idx] = hmap
                m_idx = self._gp.m_idxs[p_idx].start
                a_idxs = list(self.a_idxs[p_idx])  # A's entries we can modify
                for i, (exp, c) in enumerate(hmap.items()):
                    self._gp.exps[m_idx + i] = exp
                    self._gp.cs[m_idx + i] = c
                    for var, x in exp.items():
                        try:  # modify a particular A entry
                            row_idx = a_idxs.pop()
                            self._gp.A.row[row_idx] = m_idx + i
                            self._gp.A.col[row_idx] = self._gp.varidxs[var]
                            self._gp.A.data[row_idx] = x
                        except IndexError:  # numbers of exps increased
                            self.a_idxs[p_idx].append(len(self._gp.A.row))
                            self._gp.A.row.append(m_idx + i)
                            self._gp.A.col.append(self._gp.varidxs[var])
                            self._gp.A.data.append(x)
                for row_idx in a_idxs:  # number of exps decreased
                    self._gp.A.row[row_idx] = 0  # zero out this entry
                    self._gp.A.col[row_idx] = 0
                    self._gp.A.data[row_idx] = 0
        return self._gp
