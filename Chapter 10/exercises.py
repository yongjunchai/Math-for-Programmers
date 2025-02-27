from fontTools.misc.cython import returns

from expressions import *
from sympy import *
from sympy.core.core import *


def contains(expr: Expression, var : Variable) -> bool:
    if isinstance(expr, Sum):
        return any([ contains(exp, var) for exp in expr.exps])
    elif isinstance(expr, Product):
        return contains(expr.exp1, var) or contains(expr.exp2, var)
    elif isinstance(expr, Difference):
        return contains(expr.exp1, var) or contains(expr.exp2, var)
    elif isinstance(expr, Quotient):
        return contains(expr.numerator, var) or contains(expr.denominator, var)
    elif isinstance(expr, Negative):
        return contains(expr.exp, var)
    elif isinstance(expr, Number):
        return False
    elif isinstance(expr, Power):
        return contains(expr.base, var) or contains(expr.exponent, var)
    elif isinstance(expr, Apply):
        return contains(expr.argument, var)
    elif isinstance(expr, Variable):
        return expr.symbol == var.symbol
    else:
        raise TypeError("Not a valid expression.")

def exercise_10_9():
    varA = Variable("a")
    varB = Variable("b")
    varC = Variable("c")
    varD = Variable("d")
    p = Product(varA, varB)
    d = Difference(p, varA)
    q = Quotient(d, p)
    pp = Power(d, q)
    s = Sum(p, d, q, pp)
    print(f"{contains(s, varD)}")

def distinct_functions(expr: Expression) -> set :
    if isinstance(expr, Sum):
        return set.union(* [distinct_functions(exp) for exp in expr.exps])
    elif isinstance(expr, Product):
        return distinct_functions(expr.exp1).union(distinct_functions(expr.exp2))
    elif isinstance(expr, Difference):
        return distinct_functions(expr.exp1).union(distinct_functions(expr.exp2))
    elif isinstance(expr, Quotient):
        return distinct_functions(expr.numerator).union(distinct_functions(expr.denominator))
    elif isinstance(expr, Negative):
        return distinct_functions(expr.exp)
    elif isinstance(expr, Number):
        return set()
    elif isinstance(expr, Power):
        return distinct_functions(expr.base).union(distinct_functions(expr.exponent))
    elif isinstance(expr, Apply):
        return {expr.function.name}.union(distinct_functions(expr.argument))
    elif isinstance(expr, Variable):
        return set()
    else:
        return set()


def exercise_10_10():
    expr = Sum(Difference(Number(5), Number(6)), Quotient(Apply(Function("sin"), 10), Power(Number(2), Number(3))),
               Quotient(Apply(Function("log"), Number(3)), Power(Number(2), Number(3)))
               )
    print(distinct_functions(expr))


def contains_sum(expr) -> bool:
    if isinstance(expr, Sum):
        return True
    elif isinstance(expr, Product):
        return contains_sum(expr.exp1) or contains_sum(expr.exp2)
    elif isinstance(expr, Difference):
        return contains_sum(expr.exp1) or contains_sum(expr.exp2)
    elif isinstance(expr, Quotient):
        return contains_sum(expr.numerator) or contains_sum(expr.denominator)
    elif isinstance(expr, Negative):
        return contains_sum(expr.exp)
    elif isinstance(expr, Number):
        return False
    elif isinstance(expr, Power):
        return contains_sum(expr.base) or contains_sum(expr.exponent)
    elif isinstance(expr, Apply):
        return contains_sum(expr.argument)
    elif isinstance(expr, Variable):
        return False
    else:
        raise TypeError("Not a valid expression.")

def exercise_10_11():
    expr = Quotient(Apply(Function("log"), Number(3)), Power(Number(2), Sum(Number(3), Number(6))))
    print(contains_sum(expr))

def exercise_10_23():
    x = Symbol("x")
    ex = Mul(x, cos(x))
    print(ex.integrate(x))

def exercise_10_24():
    x = Symbol("x")
    ex = x ** 2
    print(ex.integrate(x))


exercise_10_24()
