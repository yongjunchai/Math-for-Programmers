from random import randint

from transforms import *
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


def multiply_matrix_vector(matrix, vector):
    return tuple(sum(row_entry * vector_entry for row_entry, vector_entry in  zip(row, vector ))  for row in matrix)

def exercise_5_7():
    m1 = gen_random_matrix(3, 3, min_val=0, max_val=10)
    v1= [1, 2, 3]
    r1 = matrix_multiply_vector(m1, v1)
    r2 = multiply_matrix_vector(m1, v1)
    print(m1)
    print(v1)
    print(r1)
    print(r2)

def exercise_5_10():
    A = ((1, 1, 0),
         (1, 0, 1),
         (1, -1, 1)
        )
    B = ((0, 2, 1),
         (0, 1, 0),
         (1, 0, -1))

    def transform_a(vector):
        return matrix_multiply_vector(A, vector)

    def transform_b(vector):
        return matrix_multiply_vector(B, vector)

    compose_a_b = compose(transform_a, transform_b)
    m1 = matrix_multiply(A, B)
    m2 = infer_matrix(3, compose_a_b)
    print(m1)
    print(m2)

def exercise_5_11():
    m270 = ((0, 1), (-1, 0))
    m90 = ((0, -1), (1, 0))
    r1 = matrix_multiply(m270, m90)
    r2 = matrix_multiply(m90, m270)
    print(r1)
    print(r2)
    v_random = (randint(1, 100), randint(1, 100))
    print(v_random)
    print(matrix_multiply_vector(r1, v_random))

def matrix_power(power, matrix):
    r = matrix
    for _ in range(1, power):
        r = matrix_multiply(r, matrix)
    return r

def sample_matrix_multiply():
    c = ((-1, -1, 0), (-2, 1, 2), (1, 0, -1))
    d = ((1,), (2,), (3,))
    r = matrix_multiply(c,d)
    print(r)

def figure_5_10():
    first_matrix = ((1, -2, 0), (-1, -2, 2))
    second_matrix = ((2, 0, -1, 2), (0, -2, 2, -2), (-1, -1, 2, 1))
    r = matrix_multiply(first_matrix, second_matrix)
    print(r)

def exercise_5_15():
    m_3_2 = ((1, 2), (1, 2), (1, 2))
    m_4_5 = ((1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5))
    r = matrix_multiply(m_3_2, m_4_5)
    print(r)

def transpose(matrix):
    return tuple(zip(*matrix))

def exercise_5_18():
    m = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12))
    print(transpose(m))
    m= ((1, 2, 3, 4), )
    print(transpose(m))
    m = ((1,), (2,), (3,), (4,))
    print(transpose(m))

def exercise_5_22():
    def project_x_z(vec):
        x, y, z = vec
        return x, z
    tm = infer_matrix(3, project_x_z)
    print(tm)

def exercise_5_23():
    m = (
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1)
    )
    vector = (1, 2, 3, 4, 5)
    print(matrix_multiply_vector(m, vector))


def exercise_5_24():
    #   (l, e, m, o, n, s) => (s, o, l, e, m, n)
    m = (
        (0, 0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0, 0),
        (1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 1, 0)
    )
    vec = (1, 2, 3, 4, 5, 6)
    print(matrix_multiply_vector(m, vec))

from vector_drawing import *

def get_dino_vectors():
    return [(6,4), (3,1), (1,2), (-1,5), (-2,5), (-3,4), (-4,4),
        (-5,3), (-5,2), (-2,2), (-5,1), (-4,0), (-2,1), (-1,0), (0,-3),
        (-1,-4), (1,-4), (2,-3), (1,-2), (3,-1), (5,1)
    ]

def draw_dino():
    dino_vectors = get_dino_vectors()
    draw(
        Points(*dino_vectors),
        Polygon(*dino_vectors)
    )

def translate_2d(translation):
    a, b = translation
    matrix = (
        (1, 0, a),
        (0, 1, b),
        (0, 0, 1)
    )
    def new_function(target):
        x, y = target
        vector = (x, y, 1)
        x_out, y_out, z_out = matrix_multiply_vector(matrix, vector)
        return x_out, y_out
    return new_function

def exercise_5_26():
    dino_vectors = get_dino_vectors()
    translation = (5, 10)
    translation_fun = translate_2d(translation)
    translated_dino_vectors = [translation_fun(vec) for vec in dino_vectors]
    draw(Points(*dino_vectors, color=green), Polygon(*dino_vectors, color=green),
         Points(*translated_dino_vectors, color=red), Polygon(*translated_dino_vectors, color=red)
         )

def exercise_5_27():
    translation = (-2, -2)
    dino_vectors = get_dino_vectors()
    translation_fun = translate_2d(translation)
    translated_dino_vectors = [translation_fun(vec) for vec in dino_vectors]
    draw(Points(*dino_vectors, color=green), Polygon(*dino_vectors, color=green),
         Points(*translated_dino_vectors, color=red), Polygon(*translated_dino_vectors, color=red)
         )

import math
def exercise_5_29():
    rotate_radiant = math.pi / 4
    scale_factor = 0.5
    matrix_rotate = (
        (math.cos(rotate_radiant), -math.cos(rotate_radiant)),
        (math.sin(rotate_radiant), sin(rotate_radiant))
    )
    print("matrix_rotate")
    print(matrix_rotate)
    matrix_scale = (
        (scale_factor, 0),
        (0, scale_factor)
    )
    print("matrix_scale")
    print(matrix_scale)
    rotate_scale = matrix_multiply(matrix_rotate, matrix_scale)
    scal_rotate = matrix_multiply(matrix_scale, matrix_rotate)
    ((a, b), (c, d)) = scal_rotate
    final_matrix = (
        (a, b, 2),
        (c, d, 2),
        (0, 0, 1)
    )
    print("final_matrix")
    print(final_matrix)
    dino_vectors = get_dino_vectors()
    def append_z_1(vector):
        x, y = vector
        return x, y, 1

    def matrix_multiply_vec(vec):
        x_out, y_out, z_out = matrix_multiply_vector(final_matrix, vec)
        return x_out, y_out

    dino_vectors_z_1 = [append_z_1(vec) for vec in dino_vectors]
    dino_vectors_final = [ matrix_multiply_vec(vec) for vec in dino_vectors_z_1]
    draw(Points(*dino_vectors, color=green), Polygon(*dino_vectors, color=green),
         Points(*dino_vectors_final, color=red), Polygon(*dino_vectors_final, color=red)
         )

def translate_4d(translation):
    a, b, c, d = translation
    matrix = (
        (1, 0, 0, 0, a),
        (0, 1, 0, 0, b),
        (0, 0, 1, 0, c),
        (0, 0, 0, 1, d),
        (0, 0, 0, 0, 1)
    )
    def new_function(target):
        x, y, z, t = target
        vector = (x, y, z, t, 1)
        x_out, y_out, z_out, t_out, d_out = matrix_multiply_vector(matrix, vector)
        return x_out, y_out, z_out, t_out
    return new_function


def exercise_5_31():
    translation = (1, 2, 3, 4)
    translation_fun = translate_4d(translation)
    to_translate = (10, 20, 30, 40)
    translated = translation_fun(to_translate)
    print(translation)
    print(to_translate)
    print(translated)


# draw_dino()
exercise_5_31()





























