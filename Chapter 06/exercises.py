from abc import ABCMeta, abstractmethod, abstractproperty
from random import uniform
from math import isclose
from vectors import add, scale
def random_scalar():
    return uniform(-10, 10)

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
        return Vec6(0, 0, 0, 0)

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

exercise_6_7()
# run_test_against_linear_rules()




















