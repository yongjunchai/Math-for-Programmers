from random import randint
from vectors import *
from transforms import rotate_z_by
import math
from teapot import load_triangles
from draw_model import draw_model

def create_base_vector(n, val_index):
    return tuple(1 if i == val_index else 0 for i in range(0, n))

def infer_matrix(n, transformation):
    transformed_base_vectors = tuple(transformation(create_base_vector(n, i)) for i in range(0, n))
    return tuple(zip(*transformed_base_vectors))


def matrix_multiply(a, b):
    return tuple(tuple(dot(row, col) for col in zip(*b)) for row in a)

def matrix_multiply_vector(a, v):
    return tuple(dot(row, v) for row in a)

def exercise_5_1():
    vector = [1, 2, 3]
    transformation = rotate_z_by(math.pi/2)
    print(vector)
    print(transformation(vector))
    m = infer_matrix(3, transformation)
    print(m)
    print(matrix_multiply_vector(m, vector))


def exercise_5_2():
    m = ((1.3, 0.7),
         (6.5, 3.2))
    v = (-2.5, 0.3)
    print(matrix_multiply_vector(m, v))

def gen_random_matrix(rows, cols, min_val=1, max_val=3):
    return tuple( tuple( randint(min_val, max_val) for _ in range(0, cols))  for _ in range(0, rows))

def exercise_5_3():
    m1 = gen_random_matrix(3, 3, 0, 10)
    m2 = gen_random_matrix(3, 3, 0, 10)
    print(m1)
    print(m2)
    print(matrix_multiply(m1, m2))

def transform_identity(n):
    return tuple( tuple(1 if i == j else 0 for j in range(0, n)) for i in range(0, n))

def exercise_5_5():
    n = 5
    m1 = gen_random_matrix(n, n)
    identity_matrix = transform_identity(n)
    print(m1)
    print(identity_matrix)
    transformed = matrix_multiply(m1, identity_matrix)
    print(transformed)

def exercise_5_6():
    transformation_matrix = (
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2)
    )
    teapot = load_triangles()
    output = tuple( tuple( matrix_multiply_vector(transformation_matrix, vec) for vec in tri)    for tri in teapot)
    draw_model(output)

exercise_5_6()
