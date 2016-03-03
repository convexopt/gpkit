"""Tests for GP, SP, and Breakdown classes"""
import math
import unittest
from gpkit import (Model, Monomial, settings, VectorVariable, Variable,
                   SignomialsEnabled, ArrayVariable)
from gpkit.small_classes import CootMatrix
from gpkit.feasibility import feasibility_model
from gpkit import breakdown

NDIGS = {"cvxopt": 5, "mosek": 7, "mosek_cli": 5}
# name: decimal places of accuracy


class TestGP(unittest.TestCase):
    """
    Test GeometricPrograms.
    This TestCase gets run once for each installed solver.
    """
    name = "TestGP_"
    # solver and ndig get set in loop at bottom this file, a bit hacky
    solver = None
    ndig = None

    def test_trivial_gp(self):
        """
        Create and solve a trivial GP:
            minimize    x + 2y
            subject to  xy >= 1

        The global optimum is (x, y) = (sqrt(2), 1/sqrt(2)).
        """
        x = Monomial('x')
        y = Monomial('y')
        prob = Model(cost=(x + 2*y),
                     constraints=[x*y >= 1])
        sol = prob.solve(solver=self.solver, verbosity=0)
        self.assertEqual(type(prob.latex()), str)
        self.assertEqual(type(prob._repr_latex_()), str)
        self.assertAlmostEqual(sol("x"), math.sqrt(2.), self.ndig)
        self.assertAlmostEqual(sol("y"), 1/math.sqrt(2.), self.ndig)
        self.assertAlmostEqual(sol("x") + 2*sol("y"),
                               2*math.sqrt(2),
                               self.ndig)
        self.assertAlmostEqual(sol["cost"], 2*math.sqrt(2), self.ndig)

    def test_simple_united_gp(self):
        R = Variable('R', units="nautical_miles")
        a0 = Variable('a0', 340.29, 'm/s')
        theta = Variable(r'\theta', 0.7598)
        t = Variable('t', 10, 'hr')
        T_loiter = Variable('T_{loiter}', 1, 'hr')
        T_reserve = Variable('T_{reserve}', 45, 'min')
        M = VectorVariable(2, 'M')

        if R.units:
            prob = Model(1/R,
                         [t >= sum(R/a0/M/theta**0.5) + T_loiter + T_reserve,
                          M <= 0.76])
            sol = prob.solve(verbosity=0)
            self.assertAlmostEqual(sol["cost"], 0.0005532, self.ndig)

    def test_trivial_vector_gp(self):
        """
        Create and solve a trivial GP with VectorVariables
        """
        x = VectorVariable(2, 'x')
        y = VectorVariable(2, 'y')
        prob = Model(cost=(sum(x) + 2*sum(y)),
                     constraints=[x*y >= 1])
        sol = prob.solve(solver=self.solver, verbosity=0)
        self.assertEqual(sol('x').shape, (2,))
        self.assertEqual(sol('y').shape, (2,))
        for x, y in zip(sol('x'), sol('y')):
            self.assertAlmostEqual(x, math.sqrt(2.), self.ndig)
            self.assertAlmostEqual(y, 1/math.sqrt(2.), self.ndig)
        self.assertAlmostEqual(sol["cost"]/(4*math.sqrt(2)), 1., self.ndig)

    def test_zero_lower_unbounded(self):
        x = Variable('x', value=4)
        y = Variable('y', value=0)
        z = Variable('z')
        t1 = Variable('t1')
        t2 = Variable('t2')

        prob = Model(z, [z >= x + t1,
                         t1 >= t2,
                         t2 >= y])
        prob.zero_lower_unbounded_variables()
        sol = prob.solve(verbosity=0)
        self.assertAlmostEqual(sol["cost"]/x.value, 1, self.ndig)
        self.assertAlmostEqual(sol("t2"), 0, self.ndig)

    def test_mdd_example(self):
        Cl = Variable("Cl", 0.5, "-", "Lift Coefficient")
        Mdd = Variable("Mdd", "-", "Drag Divergence Mach Number")
        m1 = Model(1/Mdd, [1 >= 5*Mdd + 0.5, Mdd >= 0.00001])
        m2 = Model(1/Mdd, [1 >= 5*Mdd + 0.5])
        m3 = Model(1/Mdd, [1 >= 5*Mdd + Cl, Mdd >= 0.00001])
        sol1 = m1.solve(solver=self.solver, verbosity=0)
        sol2 = m2.solve(solver=self.solver, verbosity=0)
        sol3 = m3.solve(solver=self.solver, verbosity=0)
        # pylint: disable=no-member
        gp1, gp2, gp3 = [m.program for m in [m1, m2, m3]]
        self.assertEqual(gp1.A, CootMatrix(row=[0, 1, 2],
                                           col=[0, 0, 0],
                                           data=[-1, 1, -1]))
        self.assertEqual(gp2.A, CootMatrix(row=[0, 1],
                                           col=[0, 0],
                                           data=[-1, 1]))
        self.assertEqual(gp3.A, CootMatrix(row=[0, 1, 2],
                                           col=[0, 0, 0],
                                           data=[-1, 1, -1]))
        self.assertAlmostEqual(sol1(Mdd), sol2(Mdd))
        self.assertAlmostEqual(sol1(Mdd), sol3(Mdd))
        self.assertAlmostEqual(sol2(Mdd), sol3(Mdd))

    def test_additive_constants(self):
        x = Variable('x')
        m = Model(1/x, [1 >= 5*x + 0.5, 1 >= 10*x])
        m.solve(verbosity=0)
        # pylint: disable=no-member
        gp = m.program  # created by solve()
        self.assertEqual(gp.cs[1], gp.cs[2])
        self.assertEqual(gp.A.data[1], gp.A.data[2])

    def test_zeroing(self):
        L = Variable("L")
        k = Variable("k", 0)
        with SignomialsEnabled():
            constr = [L-5*k <= 10]
        sol = Model(1/L, constr).solve(verbosity=0, solver=self.solver)
        self.assertAlmostEqual(sol(L), 10, self.ndig)
        self.assertAlmostEqual(sol["cost"], 0.1, self.ndig)

    def test_singular(self):
        """
        Create and solve GP with a singular A matrix
        """
        if self.solver == 'cvxopt':
            # cvxopt can't solve this problem
            # (see https://github.com/cvxopt/cvxopt/issues/36)
            return
        x = Variable('x')
        y = Variable('y')
        m = Model(y*x, [y*x >= 12])
        sol = m.solve(solver=self.solver, verbosity=0)
        self.assertAlmostEqual(sol["cost"], 12, self.ndig)

    def test_constants_in_objective_1(self):
        '''Issue 296'''
        x1 = Variable('x1')
        x2 = Variable('x2')
        m = Model(1.+ x1 + x2, [x1 >= 1., x2 >= 1.])
        sol = m.solve(solver=self.solver, verbosity=0)
        self.assertAlmostEqual(sol["cost"], 3, self.ndig)

    def test_constants_in_objective_2(self):
        '''Issue 296'''
        x1 = Variable('x1')
        x2 = Variable('x2')
        m = Model(x1**2 + 100 + 3*x2, [x1 >= 10., x2 >= 15.])
        sol = m.solve(solver=self.solver, verbosity=0)
        self.assertAlmostEqual(sol["cost"]/245., 1, self.ndig)

    def test_feasibility_gp_(self):
        x = Variable('x')
        m = Model(x, [x**2 >= 1, x <= 0.5])
        self.assertRaises(RuntimeWarning, m.solve, verbosity=0)
        fm = feasibility_model(m.gp(), "max")
        # pylint: disable=no-member
        sol1 = fm.solve(verbosity=0)
        fm = feasibility_model(m.gp(), "product")
        sol2 = fm.solve(verbosity=0)
        self.assertTrue(sol1["cost"] >= 1)
        self.assertTrue(sol2["cost"] >= 1)

    def test_terminating_constant_(self):
        x = Variable('x')
        y = Variable('y', value=0.5)
        prob = Model(1/x, [x + y <= 4])
        sol = prob.solve(verbosity=0)
        self.assertAlmostEqual(sol["cost"], 1/3.5, self.ndig)

    def test_exps_is_tuple(self):
        """issue 407"""
        x = Variable('x')
        m = Model(x, [x >= 1])
        m.solve(verbosity=0)
        self.assertEqual(type(m.program.cost.exps), tuple)

    def test_posy_simplification(self):
        "issue 525"
        D = Variable('D')
        mi = Variable('m_i')
        V = Variable('V', 1)
        m1 = Model(D + V, [V >= mi + 0.4])
        m2 = Model(D + 1, [1 >= mi + 0.4])
        gp1, gp2 = m1.gp(verbosity=0), m2.gp(verbosity=0)
        # pylint: disable=no-member
        self.assertEqual(gp1.A, gp2.A)
        self.assertTrue((gp1.cs == gp2.cs).all())


class TestSP(unittest.TestCase):
    """test case for SP class -- gets run for each installed solver"""
    name = "TestSP_"
    solver = None
    ndig = None

    def test_trivial_sp(self):
        x = Variable('x')
        y = Variable('y')
        with SignomialsEnabled():
            m = Model(x, [x >= 1-y, y <= 0.1])
        sol = m.localsolve(verbosity=0, solver=self.solver)
        self.assertAlmostEqual(sol["variables"]["x"], 0.9, self.ndig)
        with SignomialsEnabled():
            m = Model(x, [x+y >= 1, y <= 0.1])
        sol = m.localsolve(verbosity=0, solver=self.solver)
        self.assertAlmostEqual(sol["variables"]["x"], 0.9, self.ndig)

    def test_relaxation(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            constraints = [y + x >= 2, y <= x]
        objective = x
        m = Model(objective, constraints)
        m.localsolve(verbosity=0)

        # issue #257

        A = VectorVariable(2, "A")
        B = ArrayVariable([2, 2], "B")
        C = VectorVariable(2, "C")
        with SignomialsEnabled():
            constraints = [A <= B.dot(C),
                           B <= 1,
                           C <= 1]
        obj = 1/A[0] + 1/A[1]
        m = Model(obj, constraints)
        m.localsolve(verbosity=0)

    def test_issue180(self):
        L = Variable("L")
        Lmax = Variable("L_{max}", 10)
        W = Variable("W")
        Wmax = Variable("W_{max}", 10)
        A = Variable("A", 10)
        Obj = Variable("Obj")
        a_val = 0.01
        a = Variable("a", a_val)
        with SignomialsEnabled():
            eqns = [L <= Lmax,
                    W <= Wmax,
                    L*W >= A,
                    Obj >= a*(2*L + 2*W) + (1-a)*(12 * W**-1 * L**-3)]
        m = Model(Obj, eqns)
        spsol = m.solve(verbosity=0, solver=self.solver)
        # now solve as GP
        eqns[-1] = (Obj >= a_val*(2*L + 2*W) + (1-a_val)*(12 * W**-1 * L**-3))
        m = Model(Obj, eqns)
        gpsol = m.solve(verbosity=0, solver=self.solver)
        self.assertAlmostEqual(spsol['cost'], gpsol['cost'])

    def test_trivial_sp2(self):
        x = Variable("x")
        y = Variable("y")

        # converging from above
        with SignomialsEnabled():
            constraints = [y + x >= 2, y >= x]
        objective = y
        x0 = 1
        y0 = 2
        m = Model(objective, constraints)
        sol1 = m.localsolve(x0={x: x0, y: y0}, verbosity=0, solver=self.solver)

        # converging from right
        with SignomialsEnabled():
            constraints = [y + x >= 2, y <= x]
        objective = x
        x0 = 2
        y0 = 1
        m = Model(objective, constraints)
        sol2 = m.localsolve(x0={x: x0, y: y0}, verbosity=0, solver=self.solver)

        self.assertAlmostEqual(sol1["variables"]["x"],
                               sol2["variables"]["x"], self.ndig)
        self.assertAlmostEqual(sol1["variables"]["y"],
                               sol2["variables"]["x"], self.ndig)

    def test_sp_initial_guess_sub(self):
        x = Variable("x")
        y = Variable("y")
        x0 = 1
        y0 = 2
        with SignomialsEnabled():
            constraints = [y + x >= 2, y <= x]
        objective = x
        m = Model(objective, constraints)
        try:
            sol = m.localsolve(x0={x: x0, y: y0}, verbosity=0,
                               solver=self.solver)
        except TypeError:
            self.fail("Call to local solve with only variables failed")
        self.assertAlmostEqual(sol(x), 1, self.ndig)
        self.assertAlmostEqual(sol["cost"], 1, self.ndig)

        try:
            sol = m.localsolve(x0={"x": x0, "y": y0}, verbosity=0,
                               solver=self.solver)
        except TypeError:
            self.fail("Call to local solve with only variable strings failed")
        self.assertAlmostEqual(sol("x"), 1, self.ndig)
        self.assertAlmostEqual(sol["cost"], 1, self.ndig)

        try:
            sol = m.localsolve(x0={"x": x0, y: y0}, verbosity=0,
                               solver=self.solver)
        except TypeError:
            self.fail("Call to local solve with a mix of variable strings "
                      "and variables failed")
        self.assertAlmostEqual(sol["cost"], 1, self.ndig)

    def test_small_signomial(self):
        x = Variable('x')
        z = Variable('z')
        local_ndig = 4
        nonzero_adder = 0.1  # TODO: support reaching zero, issue #348
        with SignomialsEnabled():
            J = 0.01*(x - 1)**2 + nonzero_adder
            m = Model(z, [z >= J])
        sol = m.localsolve(verbosity=0)
        self.assertAlmostEqual(sol['cost'], nonzero_adder, local_ndig)
        self.assertAlmostEqual(sol('x'), 0.987, 3)

    def test_sigs_not_allowed_in_cost(self):
        with SignomialsEnabled():
            x = Variable('x')
            y = Variable('y')
            J = 0.01*((x - 1)**2 + (y - 1)**2) + (x*y - 1)**2
            m = Model(J)
            with self.assertRaises(TypeError):
                _ = m.localsolve(verbosity=0)

    def test_partial_sub_signomial(self):
        """Test SP partial x0 initialization"""
        x = Variable('x')
        y = Variable('y')
        with SignomialsEnabled():
            m = Model(x, [x + y >= 1, y <= 0.5])
        m.localsolve(x0={x: 0.5}, verbosity=0)
        first_gp_constr_posy = m.program.gps[0].constraints[0].as_posyslt1()[0]
        self.assertEqual(first_gp_constr_posy.exp[x.key], -1./3)

    

class TestBreakdown(unittest.TestCase):

    """test case for Breakdown class -- gets run for each installed solver"""
    name = "TestBreakdown_"
    solver = None
    ndig = None
    def test_breakdown(self):
        input={'w':{'w1':{'w11':[3,"N","test"],'w12':{'w121':[2,"N"],'w122':[6,"N"]}},'w2':{'w21':[1,"N"],'w22':[2,"N"]},'w3':[1,"N"]}}
        
        bd=breakdown.Breakdown(input,"N")
        sol=bd.solve_method()

        self.assertAlmostEqual(sol('w')-15,0,5)
        self.assertAlmostEqual(sol('w1')-11,0,5)
        self.assertAlmostEqual(sol('w2')-3,0,5)
        self.assertAlmostEqual(sol('w3')-1,0,5)
        self.assertAlmostEqual(sol('w11')-3,0,5)
        self.assertAlmostEqual(sol('w12')-8,0,5)
        self.assertAlmostEqual(sol('w121')-2,0,5)
        self.assertAlmostEqual(sol('w122')-6,0,5)
        self.assertAlmostEqual(sol('w21')-1,0,5)
        self.assertAlmostEqual(sol('w22')-2,0,5)
        

TEST_CASES = [TestGP, TestSP, TestBreakdown]

TESTS = []
for testcase in TEST_CASES:
    for solver in settings["installed_solvers"]:
        if solver:
            test = type(testcase.__name__+"_"+solver,
                        (testcase,), {})
            setattr(test, "solver", solver)
            setattr(test, "ndig", NDIGS[solver])
            TESTS.append(test)

if __name__ == "__main__":
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    print TESTS
    run_tests(TESTS)
