"Scripts for generating, solving and sweeping programs"
from time import time
import warnings as pywarnings
import numpy as np
from ad import adnumber
from ..nomials import parse_subs
from ..solution_array import SolutionArray
from ..keydict import KeyDict
from ..small_scripts import maybe_flatten
from ..small_classes import FixedScalar
from ..exceptions import Infeasible
from ..globals import SignomialsEnabled


def evaluate_linked(constants, linked):
    "Evaluates the values and gradients of linked variables."
    kdc = KeyDict({k: adnumber(maybe_flatten(v), k)
                   for k, v in constants.items()})
    kdc_plain = None
    array_calulated = {}
    for key in constants:  # remove gradients from constants
        if key.gradients:
            del key.descr["gradients"]
    for v, f in linked.items():
        try:
            if v.veckey and v.veckey.vecfn:
                if v.veckey not in array_calulated:
                    with SignomialsEnabled():  # to allow use of gpkit.units
                        vecout = v.veckey.vecfn(kdc)
                    if not hasattr(vecout, "shape"):
                        vecout = np.array(vecout)
                    array_calulated[v.veckey] = vecout
                out = array_calulated[v.veckey][v.idx]
            else:
                with SignomialsEnabled():  # to allow use of gpkit.units
                    out = f(kdc)
            if isinstance(out, FixedScalar):  # to allow use of gpkit.units
                out = out.value
            if hasattr(out, "units"):
                out = out.to(v.units or "dimensionless").magnitude
            elif out != 0 and v.units:
                pywarnings.warn(
                    "Linked function for %s did not return a united value."
                    "Modifying it to do so (e.g. by using `()` instead of `[]`"
                    " to access variables) will reduce errors." % v)
            if not hasattr(out, "x"):
                constants[v] = out
                continue  # a new fixed variable, not a calculated one
            constants[v] = out.x
            gradients = {adn.tag:
                         grad for adn, grad in out.d().items() if adn.tag}
            if gradients:
                v.descr["gradients"] = gradients
        except Exception as exception:  # pylint: disable=broad-except
            from .. import settings
            if settings.get("ad_errors_raise", None):
                raise
            print("Warning: skipped auto-differentiation of linked variable"
                  " %s because %s was raised. Set `gpkit.settings"
                  "[\"ad_errors_raise\"] = True` to raise such Exceptions"
                  " directly.\n" % (v, repr(exception)))
            if kdc_plain is None:
                kdc_plain = KeyDict(constants)
            constants[v] = f(kdc_plain)
            v.descr.pop("gradients", None)


def progify(program, return_attr=None):
    """Generates function that returns a program() and optionally an attribute.

    Arguments
    ---------
    program: NomialData
        Class to return, e.g. GeometricProgram or SequentialGeometricProgram
    return_attr: string
        attribute to return in addition to the program
    """
    def programfn(self, constants=None, **initargs):
        "Return program version of self"
        if not constants:
            constants, _, linked = parse_subs(self.varkeys, self.substitutions)
            if linked:
                evaluate_linked(constants, linked)
        prog = program(self.cost, self, constants, **initargs)
        prog.model = self  # NOTE SIDE EFFECTS
        if return_attr:
            return prog, getattr(prog, return_attr)
        return prog
    return programfn


def solvify(genfunction):
    "Returns function for making/solving/sweeping a program."
    def solvefn(self, solver=None, *, verbosity=1, skipsweepfailures=False,
                **solveargs):
        """Forms a mathematical program and attempts to solve it.

         Arguments
         ---------
         solver : string or function (default None)
             If None, uses the default solver found in installation.
         verbosity : int (default 1)
             If greater than 0 prints runtime messages.
             Is decremented by one and then passed to programs.
         skipsweepfailures : bool (default False)
             If True, when a solve errors during a sweep, skip it.
         **solveargs : Passed to solve() call

         Returns
         -------
         sol : SolutionArray
             See the SolutionArray documentation for details.

         Raises
         ------
         ValueError if the program is invalid.
         RuntimeWarning if an error occurs in solving or parsing the solution.
         """
        constants, sweep, linked = parse_subs(self.varkeys, self.substitutions)
        solution = SolutionArray()
        solution.modelstr = str(self)

        # NOTE SIDE EFFECTS: self.program and self.solution set below
        if sweep:
            run_sweep(genfunction, self, solution, skipsweepfailures,
                      constants, sweep, linked, solver, verbosity, **solveargs)
        else:
            self.program, progsolve = genfunction(self)
            result = progsolve(solver, verbosity=verbosity, **solveargs)
            if solveargs.get("process_result", True):
                self.process_result(result)
            solution.append(result)
        solution.to_arrays()
        self.solution = solution
        solution.costposy = self.cost
        solution.vks = self.vks
        return solution
    return solvefn


# pylint: disable=too-many-locals,too-many-arguments,too-many-branches
def run_sweep(genfunction, self, solution, skipsweepfailures,
              constants, sweep, linked, solver, verbosity, **solveargs):
    "Runs through a sweep."
    # sort sweeps by the eqstr of their varkey
    sweepvars, sweepvals = zip(*sorted(list(sweep.items()),
                                       key=lambda vkval: vkval[0].eqstr))
    if len(sweep) == 1:
        sweep_grids = np.array(list(sweepvals))
    else:
        sweep_grids = np.meshgrid(*list(sweepvals))

    N_passes = sweep_grids[0].size
    sweep_vects = {var: grid.reshape(N_passes)
                   for (var, grid) in zip(sweepvars, sweep_grids)}

    if verbosity > 0:
        print("Sweeping with %i solves:" % N_passes)
        tic = time()

    self.program = []
    last_error = None
    for i in range(N_passes):
        constants.update({var: sweep_vect[i]
                          for (var, sweep_vect) in sweep_vects.items()})
        if linked:
            evaluate_linked(constants, linked)
        program, solvefn = genfunction(self, constants)
        self.program.append(program)  # NOTE: SIDE EFFECTS
        try:
            if verbosity > 1:
                print("\nSolve %i:" % i)
            result = solvefn(solver, verbosity=verbosity-1, **solveargs)
            if solveargs.get("process_result", True):
                self.process_result(result)
            solution.append(result)
        except Infeasible as e:
            last_error = e
            if not skipsweepfailures:
                raise RuntimeWarning(
                    "Solve %i was infeasible; progress saved to m.program."
                    " To continue sweeping after failures, solve with"
                    " skipsweepfailures=True." % i) from e
            if verbosity > 0:
                print("Solve %i was %s." % (i, e.__class__.__name__))
    if not solution:
        raise RuntimeWarning("All solves were infeasible.") from last_error

    solution["sweepvariables"] = KeyDict()
    ksweep = KeyDict(sweep)
    for var, val in list(solution["constants"].items()):
        if var in ksweep:
            solution["sweepvariables"][var] = val
            del solution["constants"][var]
        elif linked:  # if any variables are linked, we check all of them
            if hasattr(val[0], "shape"):
                differences = ((l != val[0]).any() for l in val[1:])
            else:
                differences = (l != val[0] for l in val[1:])
            if not any(differences):
                solution["constants"][var] = [val[0]]
        else:
            solution["constants"][var] = [val[0]]

    if verbosity > 0:
        soltime = time() - tic
        print("Sweeping took %.3g seconds." % (soltime,))
