import math
from abc import ABCMeta, abstractmethod, abstractproperty
from random import uniform, random, randint
from math import isclose, sin
from vectors import add, scale
from datetime import datetime, timedelta
from matrices import multiply_matrix_vector, matrix_multiply

def random_scalar():
    return uniform(-100, 100)

class Vector(metaclass=ABCMeta):
    @abstractmethod
    def scale(self, scalar):
        pass

    @abstractmethod
    def add(self, other):
        pass

    def __mul__(self, scalar):
        return self.scale(scalar)

    def __rmul__(self, scalar):
        return self.scale(scalar)

    def __add__(self, other):
        return self.add(other)

    def subtract(self, other):
        return self.add(-1 * other)

    def __sub__(self, other):
        return self.subtract(other)

    def __truediv__(self, scalar):
        return self.scale(1.0/scalar)

    @classmethod
    @abstractproperty
    def zero(cls):
        pass

    def __neg__(self):
        return self.scale(-1)

class CoordinateVector(Vector):
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    def __init__(self, *coordinates):
        assert len(coordinates) == self.dimension
        self.coordinates = tuple(x for x in  coordinates)

    def scale(self, scalar):
        return self.__class__(*scale(scalar, self.coordinates))

    def add(self, other):
        assert self.dimension == other.dimension
        return self.__class__(*add(self.coordinates, other.coordinates))

    def __repr__(self):
        return f"{self.__class__.__qualname__}{self.coordinates}"

    def __eq__(self, other):
        assert self.__class__ == other.__class__
        assert len(self.coordinates) == len(other.coordinates)
        for vv, ww in zip(self.coordinates, other.coordinates):
            if not isclose(vv, ww):
                return False
        return True

class Vec3(Vector):
    @classmethod
    def zero(cls):
        return Vec3(0, 0, 0)

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def scale(self, scalar):
        return Vec3(scalar * self.x, scalar * self.y, scalar * self.z)

    def add(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __repr__(self):
        return f"Vec3({self.x}, {self.y}, {self.z})"

def approx_equal_vec(v: CoordinateVector, w: CoordinateVector):
    assert len(v.coordinates) == len(w.coordinates)
    assert len(v.coordinates) >=1
    same = True
    for vv, ww in zip(v.coordinates, w.coordinates):
        if not isclose(vv, ww):
            same= False
            break
    return same

def linear_combination_rules_check(eq, a, b, u, v, w):
    assert eq(u + v, v + u)
    assert eq(u + (v + w), (u + v) + w)
    assert eq(a * (b * v), (a * b) * v)
    assert eq(1 * v, v)
    assert eq((a+b) * v, a * v + b * v)
    assert eq(a * v + a * w, a * (v + w))

def exercise_6_1():
    a, b = 1, 1
    u, v, w = Vec3(1, 2, 3), Vec3(1, 2, 3), Vec3(1, 2, 3)
    print(a * u + b * v + w)

class Vec4(CoordinateVector):
    @classmethod
    def zero(cls):
        return Vec4(0, 0, 0, 0)

    @property
    def dimension(self):
        return 4

class Vec6(CoordinateVector):
    @classmethod
    def zero(cls):
        return Vec6(0, 0, 0, 0, 0, 0)

    @property
    def dimension(self):
        return 6

def exercise_6_2():
    v = Vec6(1, 2, 3, 4, 5, 6)
    w = Vec6(1, 2, 3, 4, 5, 6)
    print(v + w)
    print(10 * v)

def exercise_6_3():
    v = Vec6.zero()
    w = Vec6(1, 2, 3, 4, 5, 6)
    print(v)
    print(-w)

def random_vec6():
    return Vec6(random_scalar(), random_scalar(), random_scalar(), random_scalar(), random_scalar(), random_scalar())

def run_test_against_linear_rules():
    for _ in range(0, 100):
        a, b = random_scalar(), random_scalar()
        u, v, w = random_vec6(), random_vec6(), random_vec6()
        print(a, b, u, v, w)
        linear_combination_rules_check(approx_equal_vec, a, b, u, v, w)


def exercise_6_4():
    run_test_against_linear_rules()


def zero_test(zero, eq, a, b, u, v, w):
    assert eq(zero + v, v)
    assert eq(0 * v, zero)
    assert eq(-v + v, zero)


def exercise_6_5():
    for _ in range(0, 100):
        a, b = random_scalar(), random_scalar()
        u, v, w = random_vec6(), random_vec6(), random_vec6()
        zero_test(Vec6.zero(), approx_equal_vec, a, b, u, v, w)


def exercise_6_6():
    assert Vec6(1,2, 3, 4) == Vec6(1, 2, 3, 4, 5, 6)


def exercise_6_7():
    v = Vec6(1, 2, 3, 4, 5, 6)
    print(v, v / 2)

def exercise_6_8():
    for _ in range(0, 100):
        a, b = random_scalar(), random_scalar()
        u, v, w = random_scalar(), random_scalar(), random_scalar()
        print(a, b, u, v, w)
        linear_combination_rules_check(isclose, a, b, u, v, w)

class CarForSale(Vector):
    retrieved_date = datetime(2018,11,30,12) #1
    def __init__(self, model_year, mileage, price, posted_datetime,
                 model="(virtual)", source="(virtual)", #2
                 location="(virtual)", description="(virtual)"):
        self.model_year = model_year
        self.mileage = mileage
        self.price = price
        self.posted_datetime = posted_datetime
        self.model = model
        self.source = source
        self.location = location
        self.description = description
    def add(self, other):
        def add_dates(d1, d2): #3
            age1 = CarForSale.retrieved_date - d1
            age2 = CarForSale.retrieved_date - d2
            sum_age = age1 + age2
            return CarForSale.retrieved_date - sum_age
        return CarForSale( #4
            self.model_year + other.model_year,
            self.mileage + other.mileage,
            self.price + other.price,
            add_dates(self.posted_datetime, other.posted_datetime)
        )
    def scale(self,scalar):
        def scale_date(d): #5
            age = CarForSale.retrieved_date - d
            return CarForSale.retrieved_date - (scalar * age)
        return CarForSale(
            scalar * self.model_year,
            scalar * self.mileage,
            scalar * self.price,
            scale_date(self.posted_datetime)
        )
    @classmethod
    def zero(cls):
        return CarForSale(0, 0, 0, CarForSale.retrieved_date)

    def __repr__(self):
        return f"CarForSale({self.model_year}, {self.mileage}, {self.price}, {self.posted_datetime})"

def random_time():
    return CarForSale.retrieved_date - timedelta(days=uniform(0, 10))

def approx_equal_time(t1, t2):
    test = datetime.now()
    return isclose((test - t1).total_seconds(), (test - t2).total_seconds())

def random_car():
    return CarForSale(randint(1990, 2019), randint(0, 250000), 27000. * random(), random_time())

def approx_equal_car(c1: CarForSale, c2: CarForSale):
    return (isclose(c1.model_year, c2.model_year) and isclose(c1.mileage, c2.mileage) and isclose(c1.price, c2.price)
            and approx_equal_time(c1.posted_datetime, c2.posted_datetime))

def linear_combination_rules_with_zero_check(zero, eq, a, b, u, v, w):
    assert eq(u + v, v + u)
    assert eq(u + (v + w), (u + v) + w)
    assert eq(a * (b * v), (a * b) * v)
    assert eq(1 * v, v)
    assert eq((a+b) * v, a * v + b * v)
    assert eq(a * v + a * w, a * (v + w))
    assert eq(zero + v, v)
    assert eq(0 * v, zero)
    assert eq(-v + v, zero)

def exercise_6_9():
    for i in range(0, 1000):
        a, b = random_scalar(), random_scalar()
        u, v, w = random_car(), random_car(), random_car()
        print(a, b, u, v, w)
        linear_combination_rules_with_zero_check(CarForSale.zero(), approx_equal_car, a, b, u, v, w)

class Function(Vector):
    def __init__(self, f):
        self.function = f

    def scale(self, scalar):
        return Function(lambda x: scalar * self.function(x))

    def add(self, other):
        return Function(lambda x: self.function(x) + other.function(x))

    @classmethod
    def zero(cls):
        return Function(lambda x: 0)

    def __call__(self, arg):
        return self.function(arg)

import numpy as np
import matplotlib.pyplot as plt

def plot(fs, xmin, xmax):
    xs = np.linspace(xmin,xmax,1000000)
    _, ax = plt.subplots()
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    for f in fs:
        ys = [f(x) for x in xs]
        plt.plot(xs,ys)
    plt.show()

def exercise_6_10():
    f = Function(lambda x: 0.5 * x + 3)
    g = Function(sin)
    plot([f, g, f+g, 3 * g], -10, 10)

def approx_equal_function(f, g):
    results = []
    for _ in range(0, 100):
        x = uniform(-10, 10)
        results.append(isclose(f(x), g(x)))
    return all(results)

def exercise_6_11():
    assert approx_equal_function(lambda x: (x*x)/x, lambda x:x)

class Polynomial(Vector):
    def __init__(self, *coefficients):
        self.coefficients = coefficients

    def __call__(self,x):
        return sum(coefficient * x ** power for (power,coefficient) in enumerate(self.coefficients))

    def add(self,p):
        return Polynomial([a + b for a,b in zip(self.coefficients, p.coefficients)])

    def scale(self,scalar):
        return Polynomial([scalar * a for a in self.coefficients])

    def _repr_latex_(self):
        monomials = [repr(coefficient) if power == 0
                               else "x ^ {%d}" % power if coefficient == 1
                               else "%s x ^ {%d}" % (coefficient,power)
                               for (power,coefficient) in enumerate(self.coefficients)
                               if coefficient != 0]
        return "$ %s $" % (" + ".join(monomials))

    @classmethod
    def zero(cls):
        return Polynomial(0)

def random_function():
    degree = randint(0, 5)
    p = Polynomial(*[ uniform(-10, 10) for _ in range(0, degree)])
    return Function(lambda x : p(x))

def exercise_6_12():
    for i in range(0, 100):
        a, b = random_scalar(), random_scalar()
        u, v, w = random_function(), random_function(), random_function()
        linear_combination_rules_with_zero_check(Function.zero(), approx_equal_function, a, b, u, v, w)

class Function2(Vector):
    def __init__(self, f):
        self.function = f

    def scale(self, scalar):
        return Function2(lambda x, y: 3 * self.function(x, y))

    def add(self, other):
        return Function2(lambda x, y: self.function(x, y) + other.function(x, y))

    @classmethod
    def zero(cls):
        return Function(lambda x, y: 0)

    def __call__(self, *args):
        return self.function(*args)


def exercise_6_13():
    f = Function2(lambda x,y : x + y)
    g = Function2(lambda x, y: math.pow(x, 2) + math.pow(y, 2))
    x, y = 3, 10
    print(f(x, y))
    print(g(x, y))
    print((f + g)(3, 10))

class Matrix(Vector):
    def __init__(self, entries):
        self.entries = entries

    @property
    @abstractmethod
    def rows(self):
        pass

    @property
    @abstractmethod
    def columns(self):
        pass

    def scale(self, scalar):
        return self.__class__(tuple(
            tuple(scalar * entry  for entry in row )
            for row in self.entries
        ))

    def add(self, other):
        return self.__class__(
            tuple(
                tuple(
                    self.entries[i][j] + other.entries[i][j]
                    for j in range(0, self.columns))
                for i in range(0, self.rows)
            )
        )

    def zero(self):
        return self.__class__(
            tuple(
                tuple(0 for j in range(0, self.columns))
                for i in range(0, self.rows)
            )
        )

    def __eq__(self, other):
        assert self.__class__ == other.__class__
        assert self.rows == other.rows and self.columns == other.columns
        for i in range(0, self.rows):
            for j in range(0, self.columns):
                if not isclose(self.entries[i][j], other.entries[i][j]):
                    return False
        return True

    def __repr__(self):
        return f"{self.__class__.__qualname__}{self.entries}"


class Matrix2By2(Matrix):

    @property
    def rows(self):
        return 2

    @property
    def columns(self):
        return 2


def exercise_6_15():
    a = Matrix2By2(((1, 2), (3, 4)))
    b = Matrix2By2(((5, 6), (7, 8)))
    print(2 * a)
    print(2 * b)
    print(2 * a + 2 * b)

class Matrix5By3(Matrix):

    @property
    def rows(self):
        return 5

    @property
    def columns(self):
        return 3

def approx_equal_matrix(rows, columns):
    def cmp(m1: Matrix, m2: Matrix):
        for i in range(0, rows):
            for j in range(0, columns):
                if not isclose(m1.entries[i][j], m2.entries[i][j]):
                    return False
        return True
    return cmp

def random_matrix(rows, columns):
    return tuple(
        tuple(uniform(-10, 10) for j in range(0, columns))
        for i in range(0, rows)
    )

def exercise_6_16():
    rows = 5
    columns = 3
    approx_equal_matrix_5_by_3 = approx_equal_matrix(rows, columns)
    for _ in range(0, 100):
        a, b = random_scalar(), random_scalar()
        u, v, w = Matrix5By3(random_matrix(rows, columns)), Matrix5By3(random_matrix(rows, columns)), Matrix5By3(random_matrix(rows, columns))
        linear_combination_rules_with_zero_check(u.zero(), approx_equal_matrix_5_by_3, a, b, u, v, w)

class Vec5(CoordinateVector):
    @classmethod
    def zero(cls):
        return Vec5(0, 0, 0, 0, 0)

    @property
    def dimension(self):
        return 5

class LinearMap3dTo5d(Matrix):

    @property
    def rows(self):
        return 5

    @property
    def columns(self):
        return 3

    def __call__(self, vector3):
        return multiply_matrix_vector(self.entries, vector3)


def exercise_6_17():
    rows = 5
    columns = 3
    for _ in range(0, 100):
        a, b = random_scalar(), random_scalar()
        u, v, w = LinearMap3dTo5d(random_matrix(rows, columns)), LinearMap3dTo5d(random_matrix(rows, columns)), LinearMap3dTo5d(random_matrix(rows, columns))
        for _ in range(0, 100):
            vec3 = (random_scalar(), random_scalar(), random_scalar())
            def cmp(f, g):
                v1 = f(vec3)
                v2 = g(vec3)
                l = len(v1)
                for i in range(0, l):
                    if not isclose(v1[i], v2[i]):
                        return False
                return True
            linear_combination_rules_with_zero_check(u.zero(), cmp, a, b, u, v, w)

from PIL import Image
class ImageVector(Vector):
    size = (300,300) #1
    def __init__(self,input):
        try:
            img = Image.open(input).resize(ImageVector.size) #2
            self.pixels = img.getdata()
        except:
            self.pixels = input #3
    def image(self):
        img = Image.new('RGB', ImageVector.size) #4
        img.putdata([(int(r), int(g), int(b))
                     for (r,g,b) in self.pixels])
        return img
    def add(self,img2): #5
        return ImageVector([(r1+r2,g1+g2,b1+b2)
                            for ((r1,g1,b1),(r2,g2,b2))
                            in zip(self.pixels,img2.pixels)])
    def scale(self,scalar): #6
        return ImageVector([(scalar*r,scalar*g,scalar*b)
                      for (r,g,b) in self.pixels])
    @classmethod
    def zero(cls): #7
        total_pixels = cls.size[0] * cls.size[1]
        return ImageVector([(0,0,0) for _ in range(0,total_pixels)])
    def _repr_png_(self): #8
        return self.image()._repr_png_()

def exercise_6_19():
    img = ImageVector("melba_toy.JPG")
    zero = ImageVector.zero()
    img.image().show("melba")
    zero.image().show("zero")
    combined = img + zero
    combined.image().show("combined")

def exercise_6_20():
    inside = ImageVector("inside.JPG")
    outside = ImageVector("outside.JPG")
    linear_combo = [ inside * (0.1 * s) + outside * (1 - (0.1 * s))   for s in range(0, 11)]
    for img in linear_combo:
        img.image().show()

# run_test_against_linear_rules()




