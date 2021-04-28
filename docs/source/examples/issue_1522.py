"Tests broadcast_sub function for returned-dictionary substitutions"
from gpkit import Variable, Model, ConstraintSet, Vectorize
from gpkit.small_scripts import broadcast_substitution

class Pie(Model):
    "Pie model"
    def setup(self):
        self.x = x = Variable("x")
        z = Variable("z")
        constraints = [
            x >= z,
        ]
        substitutions = {'z': 1}
        return constraints, substitutions

class Cake(Model):
    "Cake model, containing a vector of Pies"
    def setup(self):
        self.y = y = Variable("y")
        with Vectorize(2):
            s = Pie()
        constraints = [y >= s.x]
        constraints += [s]
        subs = {'x': broadcast_substitution(s.x, [2, 3])}
        return constraints, subs

class Yum1(Model):
    "Total dessert system model containing 5 Cakes"
    def setup(self):
        with Vectorize(5):
            cake = Cake()
        y = cake.y
        self.cost = sum(y)
        constraints = ConstraintSet([cake])
        return constraints

m = Yum1()
sol = m.solve(verbosity=0)
print(sol.table())

class Yum2(Model):
    "Total dessert system model containing 1 Cake"
    def setup(self):
        with Vectorize(1):
            cake = Cake()
        y = cake.y
        self.cost = sum(y)
        constraints = ConstraintSet([cake])
        return constraints

m = Yum2()
sol = m.solve(verbosity=0)
print(sol.table())
