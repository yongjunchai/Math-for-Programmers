from audioop import reverse

from PIL.Image import preinit

from vectors import *
from teapot import *
from draw_model import *
from draw2d import *

def translate_by(translation):
    def trans(vec):
        return add(translation, vec)
    return trans

def polygon_map(transformation, polygons):
    return [ [transformation(vector)  for vector in polygon] for polygon in polygons]


def scale_by(scalar):
    def new_function(vector):
        return scale(scalar, vector)
    return new_function

def compose(*args):
    def new_function(input):
        state = input
        for my_fun in reversed(args):
            state = my_fun(state)
        return state
    return new_function

def curry2(f):
    def g(x):
        def new_function(y):
            return f(x, y)
        return new_function
    return g

def exercise_4_1():
    def translate_by(translation):
        def trans(vec):
            return add(translation, vec)
        return trans


def exercise_4_2():
    translate_by_z_neg_1 = translate_by((0, 0, -20))
    draw_model(polygon_map(translate_by_z_neg_1, load_triangles()))

def exercise_4_3():
    # draw_model(polygon_map(scale_by(0.5), load_triangles()))
    draw_model(polygon_map(scale_by(-1), load_triangles()))

def exercise_4_4():
    translate_left_20 = translate_by((-1, 0, 0))
    scale_2 = scale_by(2)
    # draw_model(polygon_map(compose(scale_2, translate_left_20), load_triangles()))
    draw_model(polygon_map(compose(translate_left_20, scale_2), load_triangles()))

def exercise_4_6():
    def prepend(string):
        def new_function(input):
            return string + input
        return new_function
    myf = compose(prepend("p"), prepend("y"), prepend("t"))
    print(myf("hon"))

def exercise_4_7():
    scale_by = curry2(scale)
    output = scale_by(2)((1, 2, 3))
    print(output)

def exercise_4_8():
    r_x_f = rotate_x_by(pi/2)
    r_z_f = rotate_z_by(pi / 2)
    r_ny_f = rotate_y_by((-1) * pi / 2)
    r_y_f = rotate_y_by(pi / 2)

    # the answer
    # draw_model(polygon_map(compose(r_z_f, r_x_f), load_triangles()))
    # draw_model(polygon_map(compose(r_ny_f, r_z_f), load_triangles()))
    # draw_model(polygon_map(compose(r_x_f, r_ny_f), load_triangles()))



    # the follow-up answer
    # draw_model(polygon_map(compose(r_x_f, r_z_f), load_triangles()))
    # draw_model(polygon_map(compose(r_y_f, r_x_f), load_triangles()))
    # draw_model(polygon_map(compose(r_z_f, r_y_f), load_triangles()))

def stretch_x(scalar, vector):
    x, y, z = vector
    return (scalar * x, y, z)

def stretch_x_by(scalar):
    def new_function(input):
        x, y, z = input
        return (scalar * x, y, z)
    return new_function

def exercise_4_9():
    draw_model(polygon_map(stretch_x_by(1.5), load_triangles()))

def exercise_4_13():
    u = (5, 3)
    v = (-2, 1)

    mid = scale(0.5, add(u, v))
    print(f"u= {u}, v = {v}, mid = {mid}")
    draw2d(Arrow2D(u, color=red), Arrow2D(v, color=green), Segment2D(u, v, color=blue),
           Points2D(mid, color=black))

def exercise_4_14():
    points = [(x, y) for x in range(0, 6) for y in range(0, 6)]
    transformed = [(v[0]**2, v[1]**2) for v in points]
    draw2d(Points2D(*points), Points2D(*transformed, color=green))

exercise_4_14()

