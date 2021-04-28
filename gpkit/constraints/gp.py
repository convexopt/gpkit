"""Implement the GeometricProgram class"""
import sys
import warnings
from time import time
from collections import defaultdict
import numpy as np
from ..repr_conventions import lineagestr
from ..small_classes import CootMatrix, SolverLog, Numbers, FixedScalar
from ..keydict import KeyDict
from ..solution_array import SolutionArray
from .set import ConstraintSet
from ..exceptions import (InvalidPosynomial, Infeasible, UnknownInfeasible,
                          PrimalInfeasible, DualInfeasible, UnboundedGP,
                          InvalidLicense)


DEFAULT_SOLVER_KWARGS = {"cvxopt": {"kktsolver": "ldl"}}
SOLUTION_TOL = {"cvxopt": 1e-3, "mosek_cli": 1e-4, "mosek_conif": 1e-3}


class MonoEqualityIndexes:
    "Class to hold MonoEqualityIndexes"

    def __init__(self):
        self.all = set()
        self.first_half = set()


def _get_solver(solver, kwargs):
    """Get the solverfn and solvername associated with solver"""
    if solver is None:
        from .. import settings
        try:
            solver = settings["default_solver"]
        except KeyError:
            raise ValueError("No default solver was set during build, so"
                             " solvers must be manually specified.")
    if solver == "cvxopt":
        from ..solvers.cvxopt import optimize
    elif solver == "mosek_cli":
        from ..solvers.mosek_cli import optimize_generator
        optimize = optimize_generator(**kwargs)
    elif solver == "mosek_conif":
        from ..solvers.mosek_conif import optimize
    elif hasattr(solver, "__call__"):
        solver, optimize = solver.__name__, solver
    else:
        raise ValueError("Unknown solver '%s'." % solver)
    return solver, optimize


class GeometricProgram:
    # pylint: disable=too-many-instance-attributes
    """Standard mathematical representation of a GP.

    Attributes with side effects
    ----------------------------
    `solver_out` and `solve_log` are set during a solve
    `result` is set at the end of a solve if solution status is optimal

    Examples
    --------
    >>> gp = gpkit.constraints.gp.GeometricProgram(
                        # minimize
                        x,
                        [   # subject to
                            x >= 1,
                        ], {})
    >>> gp.solve()
    """
    _result = solve_log = solver_out = model = v_ss = nu_by_posy = None
    choicevaridxs = integersolve = None

    def __init__(self, cost, constraints, substitutions, *, checkbounds=True):
        self.cost, self.substitutions = cost, substitutions
        for key, sub in self.substitutions.items():
            if isinstance(sub, FixedScalar):
                sub = sub.value
                if hasattr(sub, "units"):
                    sub = sub.to(key.units or "dimensionless").magnitude
                self.substitutions[key] = sub
            if not isinstance(sub, (Numbers, np.ndarray)):
                raise TypeError("substitution {%s: %s} has invalid value type"
                                " %s." % (key, sub, type(sub)))
        cost_hmap = cost.hmap.sub(self.substitutions, cost.vks)
        if any(c <= 0 for c in cost_hmap.values()):
            raise InvalidPosynomial("a GP's cost must be Posynomial")
        hmapgen = ConstraintSet.as_hmapslt1(constraints, self.substitutions)
        self.hmaps = [cost_hmap] + list(hmapgen)
        self.gen()  # Generate various maps into the posy- and monomials
        if checkbounds:
            self.check_bounds(err_on_missing_bounds=True)

    def check_bounds(self, *, err_on_missing_bounds=False):
        "Checks if any variables are unbounded, through equality constraints."
        missingbounds = {}
        for var, locs in self.varlocs.items():
            upperbound, lowerbound = False, False
            for i in locs:
                if i not in self.meq_idxs.all:
                    if self.exps[i][var] > 0:  # pylint:disable=simplifiable-if-statement
                        upperbound = True
                    else:
                        lowerbound = True
                if upperbound and lowerbound:
                    break
            if not upperbound:
                missingbounds[(var, "upper")] = "."
            if not lowerbound:
                missingbounds[(var, "lower")] = "."
        if not missingbounds:
            return {}  # all bounds found in inequalities
        meq_bounds = gen_meq_bounds(missingbounds, self.exps, self.meq_idxs)
        fulfill_meq_bounds(missingbounds, meq_bounds)
        if missingbounds and err_on_missing_bounds:
            raise UnboundedGP(
                "\n\n".join("%s has no %s bound%s" % (v, b, x)
                            for (v, b), x in missingbounds.items()))
        return missingbounds

    def gen(self):
        """Generates nomial and solve data (A, p_idxs) from posynomials.

        k [posys]: number of monomials (rows of A) present in each constraint
        m_idxs [mons]: monomial indices of each posynomial
        p_idxs [mons]: posynomial index of each monomial
        cs, exps [mons]: coefficient and exponents of each monomial
        varlocs: {vk: monomial indices of each variables' location}
        meq_idxs: {all indices of equality mons} and {the first index of each}
        varidxs: {vk: which column corresponds to it in A}
        A [mons, vks]: sparse array of each monomials' variables' exponents

        """
        self.k = [len(hmap) for hmap in self.hmaps]
        self.m_idxs, self.p_idxs, self.cs, self.exps = [], [], [], []
        self.varkeys = self.varlocs = defaultdict(list)
        self.meq_idxs = MonoEqualityIndexes()
        m_idx = 0
        row, col, data = [], [], []
        for p_idx, (N_mons, hmap) in enumerate(zip(self.k, self.hmaps)):
            self.p_idxs.extend([p_idx]*N_mons)
            self.m_idxs.append(slice(m_idx, m_idx+N_mons))
            if getattr(self.hmaps[p_idx], "from_meq", False):
                self.meq_idxs.all.add(m_idx)
                if len(self.meq_idxs.all) > 2*len(self.meq_idxs.first_half):
                    self.meq_idxs.first_half.add(m_idx)
            self.exps.extend(hmap)
            self.cs.extend(hmap.values())
            for exp in hmap:
                if not exp:  # space out A matrix with constants for mosek
                    row.append(m_idx)
                    col.append(0)
                    data.append(0)
                for var in exp:
                    self.varlocs[var].append(m_idx)
                m_idx += 1
        self.p_idxs = np.array(self.p_idxs, "int32")  # to use array equalities
        self.varidxs = {vk: i for i, vk in enumerate(self.varlocs)}
        self.choicevaridxs = {vk: i for i, vk in enumerate(self.varlocs)
                              if vk.choices}
        for j, (var, locs) in enumerate(self.varlocs.items()):
            row.extend(locs)
            col.extend([j]*len(locs))
            data.extend(self.exps[i][var] for i in locs)
        self.A = CootMatrix(row, col, data)

    # pylint: disable=too-many-statements, too-many-locals
    def solve(self, solver=None, *, verbosity=1, gen_result=True, **kwargs):
        """Solves a GeometricProgram and returns the solution.

        Arguments
        ---------
        solver : str or function (optional)
            By default uses a solver found during installation.
            If "mosek_conif", "mosek_cli", or "cvxopt", uses that solver.
            If a function, passes that function cs, A, p_idxs, and k.
        verbosity : int (default 1)
            If greater than 0, prints solver name and solve time.
        gen_result : bool (default True)
            If True, makes a human-readable SolutionArray from solver output.
        **kwargs :
            Passed to solver constructor and solver function.


        Returns
        -------
        SolutionArray (or dict if gen_result is False)
        """
        solvername, solverfn = _get_solver(solver, kwargs)
        if verbosity > 0:
            print("Using solver '%s'" % solvername)
            print(" for %i free variables" % len(self.varlocs))
            print("  in %i posynomial inequalities." % len(self.k))

        solverargs = DEFAULT_SOLVER_KWARGS.get(solvername, {})
        solverargs.update(kwargs)
        if self.choicevaridxs and solvername == "mosek_conif":
            solverargs["choicevaridxs"] = self.choicevaridxs
            self.integersolve = True
        starttime = time()
        solver_out, infeasibility, original_stdout = {}, None, sys.stdout
        try:
            sys.stdout = SolverLog(original_stdout, verbosity=verbosity-2)
            solver_out = solverfn(c=self.cs, A=self.A, meq_idxs=self.meq_idxs,
                                  k=self.k, p_idxs=self.p_idxs, **solverargs)
        except Infeasible as e:
            infeasibility = e
        except InvalidLicense as e:
            raise InvalidLicense("license for solver \"%s\" is invalid."
                                 % solvername) from e
        except Exception as e:
            raise UnknownInfeasible("Something unexpected went wrong.") from e
        finally:
            self.solve_log = "\n".join(sys.stdout)
            sys.stdout = original_stdout
            self.solver_out = solver_out

        solver_out["solver"] = solvername
        solver_out["soltime"] = time() - starttime
        if verbosity > 0:
            print("Solving took %.3g seconds." % solver_out["soltime"])

        if infeasibility:
            if isinstance(infeasibility, PrimalInfeasible):
                msg = ("The model had no feasible points; "
                       "you may wish to relax some constraints or constants.")
            elif isinstance(infeasibility, DualInfeasible):
                msg = ("The model ran to an infinitely low cost;"
                       " bounding the right variables would prevent this.")
            elif isinstance(infeasibility, UnknownInfeasible):
                msg = ("Solver failed for an unknown reason. Relaxing"
                       " constraints/constants, bounding variables, or"
                       " using a different solver might fix it.")
            if (verbosity > 0 and solver_out["soltime"] < 1
                    and hasattr(self, "model")):  # fast, top-level model
                print(msg + "\nSince the model solved in less than a second,"
                      " let's run `.debug()` to analyze what happened.\n`")
                return self.model.debug(solver=solver)
            # else, raise a clarifying error
            msg += (" Running `.debug()` or increasing verbosity may pinpoint"
                    " the trouble.")
            raise infeasibility.__class__(msg) from infeasibility

        if not gen_result:
            return solver_out
        # else, generate a human-readable SolutionArray
        self._result = self.generate_result(solver_out, verbosity=verbosity-2)
        return self.result

    @property
    def result(self):
        "Creates and caches a result from the raw solver_out"
        if not self._result:
            self._result = self.generate_result(self.solver_out)
        return self._result

    def generate_result(self, solver_out, *, verbosity=0, dual_check=True):
        "Generates a full SolutionArray and checks it."
        if verbosity > 0:
            soltime = solver_out["soltime"]
            tic = time()
        # result packing #
        result = self._compile_result(solver_out)  # NOTE: SIDE EFFECTS
        if verbosity > 0:
            print("Result packing took %.2g%% of solve time." %
                  ((time() - tic) / soltime * 100))
            tic = time()
        # solution checking #
        try:
            tol = SOLUTION_TOL.get(solver_out["solver"], 1e-5)
            self.check_solution(result["cost"], solver_out['primal'],
                                solver_out["nu"], solver_out["la"], tol)
        except Infeasible as chkerror:
            chkwarn = str(chkerror)
            if not ("Dual" in chkwarn and not dual_check):
                print("Solution check warning: %s" % chkwarn)
        if verbosity > 0:
            print("Solution checking took %.2g%% of solve time." %
                  ((time() - tic) / soltime * 100))
        return result

    def _generate_nula(self, solver_out):
        if "nu" in solver_out:
            # solver gave us monomial sensitivities, generate posynomial ones
            solver_out["nu"] = nu = np.ravel(solver_out["nu"])
            nu_by_posy = [nu[mi] for mi in self.m_idxs]
            solver_out["la"] = la = np.array([sum(nup) for nup in nu_by_posy])
        elif "la" in solver_out:
            la = np.ravel(solver_out["la"])
            if len(la) == len(self.hmaps) - 1:
                # assume solver dropped the cost's sensitivity (always 1.0)
                la = np.hstack(([1.0], la))
            # solver gave us posynomial sensitivities, generate monomial ones
            solver_out["la"] = la
            z = np.log(self.cs) + self.A.dot(solver_out["primal"])
            m_iss = [self.p_idxs == i for i in range(len(la))]
            nu_by_posy = [la[p_i]*np.exp(z[m_is])/sum(np.exp(z[m_is]))
                          for p_i, m_is in enumerate(m_iss)]
            solver_out["nu"] = np.hstack(nu_by_posy)
        else:
            raise RuntimeWarning("The dual solution was not returned.")
        return la, nu_by_posy

    def _compile_result(self, solver_out):
        result = {"cost": float(solver_out["objective"])}
        primal = solver_out["primal"]
        if len(self.varlocs) != len(primal):
            raise RuntimeWarning("The primal solution was not returned.")
        result["freevariables"] = KeyDict(zip(self.varlocs, np.exp(primal)))
        result["constants"] = KeyDict(self.substitutions)
        result["variables"] = KeyDict(result["freevariables"])
        result["variables"].update(result["constants"])
        result["soltime"] = solver_out["soltime"]
        if self.integersolve:
            result["choicevariables"] = KeyDict( \
                {k: v for k, v in result["freevariables"].items()
                 if k in self.choicevaridxs})
            result["warnings"] = {"No Dual Solution": [(\
                "This model has the discretized choice variables"
                " %s and hence no dual solution. You can fix those variables"
                " to their optimal value and get sensitivities to the resulting"
                " continuous problem by updating your model's substitions with"
                " `sol[\"choicevariables\"]`."
                % sorted(self.choicevaridxs.keys()), self.choicevaridxs)]}
            return SolutionArray(result)
        if self.choicevaridxs:
            result["warnings"] = {"Freed Choice Variables": [(\
                "This model has the discretized choice variables"
                " %s, but since the '%s' solver doesn't support discretization"
                " they were treated as continuous variables."
                % (sorted(self.choicevaridxs.keys()), solver_out["solver"]),
                self.choicevaridxs)]}  # TODO: choicevaridxs seems unnecessary

        result["sensitivities"] = {"constraints": {}}
        la, self.nu_by_posy = self._generate_nula(solver_out)
        cost_senss = sum(nu_i*exp for (nu_i, exp) in zip(self.nu_by_posy[0],
                                                         self.cost.hmap))
        gpv_ss = cost_senss.copy()
        m_senss = defaultdict(float)
        for las, nus, c in zip(la[1:], self.nu_by_posy[1:], self.hmaps[1:]):
            while getattr(c, "parent", None) is not None:
                c = c.parent
            v_ss, c_senss = c.sens_from_dual(las, nus, result)
            for vk, x in v_ss.items():
                gpv_ss[vk] = x + gpv_ss.get(vk, 0)
            while getattr(c, "generated_by", None):
                c = c.generated_by
            result["sensitivities"]["constraints"][c] = c_senss
            m_senss[lineagestr(c)] += abs(c_senss)
        result["sensitivities"]["models"] = dict(m_senss)
        # carry linked sensitivities over to their constants
        for v in list(v for v in gpv_ss if v.gradients):
            dlogcost_dlogv = gpv_ss.pop(v)
            val = np.array(result["constants"][v])
            for c, dv_dc in v.gradients.items():
                with warnings.catch_warnings():  # skip pesky divide-by-zeros
                    warnings.simplefilter("ignore")
                    dlogv_dlogc = dv_dc * result["constants"][c]/val
                    gpv_ss[c] = gpv_ss.get(c, 0) + dlogcost_dlogv*dlogv_dlogc
                if v in cost_senss:
                    if c in self.cost.vks:  # TODO: seems unnecessary
                        dlogcost_dlogv = cost_senss.pop(v)
                        before = cost_senss.get(c, 0)
                        cost_senss[c] = before + dlogcost_dlogv*dlogv_dlogc
        # add fixed variables sensitivities to models
        for vk, senss in gpv_ss.items():
            m_senss[lineagestr(vk)] += abs(senss)
        result["sensitivities"]["cost"] = cost_senss
        result["sensitivities"]["variables"] = KeyDict(gpv_ss)
        result["sensitivities"]["constants"] = \
            result["sensitivities"]["variables"]  # NOTE: backwards compat.
        return SolutionArray(result)

    def check_solution(self, cost, primal, nu, la, tol, abstol=1e-20):
        """Run checks to mathematically confirm solution solves this GP

        Arguments
        ---------
        cost:   float
            cost returned by solver
        primal: list
            primal solution returned by solver
        nu:     numpy.ndarray
            monomial lagrange multiplier
        la:     numpy.ndarray
            posynomial lagrange multiplier

        Raises
        ------
        Infeasible if any problems are found
        """
        A = self.A.tocsr()
        def almost_equal(num1, num2):
            "local almost equal test"
            return (num1 == num2 or abs((num1 - num2) / (num1 + num2)) < tol
                    or abs(num1 - num2) < abstol)
        # check primal sol #
        primal_exp_vals = self.cs * np.exp(A.dot(primal))   # c*e^Ax
        if not almost_equal(primal_exp_vals[self.m_idxs[0]].sum(), cost):
            raise Infeasible("Primal solution computed cost did not match"
                             " solver-returned cost: %s vs %s." %
                             (primal_exp_vals[self.m_idxs[0]].sum(), cost))
        for mi in self.m_idxs[1:]:
            if primal_exp_vals[mi].sum() > 1 + tol:
                raise Infeasible("Primal solution violates constraint: %s is "
                                 "greater than 1" % primal_exp_vals[mi].sum())
        # check dual sol #
        if self.integersolve:
            return
        # note: follows dual formulation in section 3.1 of
        # http://web.mit.edu/~whoburg/www/papers/hoburg_phd_thesis.pdf
        if not almost_equal(self.nu_by_posy[0].sum(), 1):
            raise Infeasible("Dual variables associated with objective sum"
                             " to %s, not 1" % self.nu_by_posy[0].sum())
        if any(nu < 0):
            minnu = min(nu)
            if minnu < -tol/1000:
                raise Infeasible("Dual solution has negative entries as"
                                 " large as %s." % minnu)
        if any(np.abs(A.T.dot(nu)) > tol):
            raise Infeasible("Dual: sum of nu^T * A did not vanish.")
        b = np.log(self.cs)
        dual_cost = sum(
            self.nu_by_posy[i].dot(
                b[mi] - np.log(self.nu_by_posy[i]/la[i]))
            for i, mi in enumerate(self.m_idxs) if la[i])
        if not almost_equal(np.exp(dual_cost), cost):
            raise Infeasible("Dual cost %s does not match primal cost %s"
                             % (np.exp(dual_cost), cost))


def gen_meq_bounds(missingbounds, exps, meq_idxs):  # pylint: disable=too-many-locals,too-many-branches
    "Generate conditional monomial equality bounds"
    meq_bounds = defaultdict(set)
    for i in meq_idxs.first_half:
        p_upper, p_lower, n_upper, n_lower = set(), set(), set(), set()
        for key, x in exps[i].items():
            if (key, "upper") in missingbounds:
                if x > 0:
                    p_upper.add((key, "upper"))
                else:
                    n_upper.add((key, "upper"))
            if (key, "lower") in missingbounds:
                if x > 0:
                    p_lower.add((key, "lower"))
                else:
                    n_lower.add((key, "lower"))
        # (consider x*y/z == 1)
        # for a var (e.g. x) to be upper bounded by this monomial equality,
        #   - vars of the same sign (y) must be lower bounded
        #   - AND vars of the opposite sign (z) must be upper bounded
        p_ub = n_lb = frozenset(n_upper).union(p_lower)
        n_ub = p_lb = frozenset(p_upper).union(n_lower)
        for keys, ub in ((p_upper, p_ub), (n_upper, n_ub)):
            for key, _ in keys:
                needed = ub.difference([(key, "lower")])
                if needed:
                    meq_bounds[(key, "upper")].add(needed)
                else:
                    del missingbounds[(key, "upper")]
        for keys, lb in ((p_lower, p_lb), (n_lower, n_lb)):
            for key, _ in keys:
                needed = lb.difference([(key, "upper")])
                if needed:
                    meq_bounds[(key, "lower")].add(needed)
                else:
                    del missingbounds[(key, "lower")]
    return meq_bounds


def fulfill_meq_bounds(missingbounds, meq_bounds):
    "Bounds variables with monomial equalities"
    still_alive = True
    while still_alive:
        still_alive = False  # if no changes are made, the loop exits
        for bound in set(meq_bounds):
            if bound not in missingbounds:
                del meq_bounds[bound]
                continue
            for condition in meq_bounds[bound]:
                if not any(bound in missingbounds for bound in condition):
                    del meq_bounds[bound]
                    del missingbounds[bound]
                    still_alive = True
                    break
    for (var, bound) in meq_bounds:
        boundstr = (", but would gain it from any of these sets: ")
        for condition in list(meq_bounds[(var, bound)]):
            meq_bounds[(var, bound)].remove(condition)
            newcond = condition.intersection(missingbounds)
            if newcond and not any(c.issubset(newcond)
                                   for c in meq_bounds[(var, bound)]):
                meq_bounds[(var, bound)].add(newcond)
        boundstr += " or ".join(str(list(condition))
                                for condition in meq_bounds[(var, bound)])
        missingbounds[(var, bound)] = boundstr
