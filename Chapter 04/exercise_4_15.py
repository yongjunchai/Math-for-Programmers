from math import cos, sin, atan2, pi, sqrt
from hypothesis import given
from hypothesis import strategies as st

def length(v):
    return sqrt(sum([coord ** 2 for coord in v]))

def add(*vectors):
    return tuple(map(sum, zip(*vectors)))

def scale(scalar, v):
    return tuple(scalar * coord for coord in v)

def to_cartesian(polar_vector):
    length, angle = polar_vector[0], polar_vector[1]
    return (length * cos(angle), length * sin(angle))

def rotate(angle, vectors):
    polars = [to_polar(v) for v in vectors]
    return [to_cartesian((l, a + angle)) for l, a in polars]

def to_polar(vector):
    x, y = vector[0], vector[1]
    angle = atan2(y, x)
    return (length(vector), angle)

def rotate_x_by(angle):
    def rotate_x(vectors):
        return rotate(angle, vectors)
    return rotate_x


min_value = 0
max_value = 100000000
error = 0.000001
@given(st.tuples(st.integers(min_value=min_value, max_value=max_value), st.integers(min_value=min_value, max_value=max_value)), st.tuples(st.integers(min_value=min_value, max_value=max_value), st.integers(min_value=min_value, max_value=max_value)))
def test_rotate(tup1, tup2):
    transformer = rotate_x_by(pi/2)
    # transformer = lambda v: [(v[0][0]**2, v[0][1]**2)]
    r1 = transformer([add(tup1, tup2)])[0]
    r2 = add(transformer([tup1])[0], transformer([tup2])[0])
    print(f"addition input = {tup1}, {tup2}")
    print(f"addition result = {r1}, {r2}")
    assert abs(r1[0] - r2[0]) <= error
    assert abs(r1[1] - r2[1]) <= error
    scalar = 30
    r1 = transformer([scale(scalar, tup1)])[0]
    r2 = scale(scalar, transformer([tup1])[0])
    print(f"scalar multiplication input = {tup1}, {tup2}")
    print(f"scalar multiplication result = {r1}, {r2}")
    assert abs(r1[0] - r2[0]) <= error
    assert abs(r1[1] - r2[1]) <= error
