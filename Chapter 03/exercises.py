from draw3d import *
from draw2d import *
from vectors import *
from math import sin, cos, pi, ceil, floor,acos

def exercise_3_1():
    v = (-1, -2, 2)
    draw3d(Points3D(v),
           Box3D(*v))

def exercise_3_2():
    pml = [-1, 1]
    vertices = [(x, y, z) for x in pml for y in pml for z in pml]
    edges = [((-1, y, z), (1, y, z)) for y in pml for z in pml] + \
            [((x, -1, z), (x, 1, z)) for x in pml for z in pml] + \
            [((x, y, -1), (x, y, 1)) for x in pml for y in pml]

    draw3d(Points3D(* vertices, color=blue), * [ Segment3D(* edge, color=blue) for edge in edges])

def exercise_3_3():
    v = (4, 0, 3)
    w = (-1, 0, 1)
    u = add(v, w)
    draw3d(
    Arrow3D(v, color=red),
            Arrow3D(w, color=blue),
            Arrow3D(u, v, color=green),
            Arrow3D(u, w, color=purple),
            Arrow3D(u, color=black)
           )
    # tip-toe (x -> w) -> (w -> u)  : w + v
    # tip-toe (x -> v) -> (v -> u)  : v + w

def exercise_3_5():
    vs = [(sin(pi * t /6), cos(pi*t/6), 1.0/3)  for t in  range(0, 24)]
    print(f"sum = {add(*vs)}")
    len_vs = len(vs)
    arrows = list()
    for i in range(0, len_vs):
        arrows.append(Arrow3D(vs[i%len_vs], vs[(i + 1)%len_vs]))
    draw3d(*arrows)

def exercise_3_5_2():
    vs = [(sin(pi * t /6), cos(pi*t/6), 1.0/3)  for t in  range(0, 24)]
    len_vs = len(vs)
    arrows = list()
    running_sum = (0, 0, 0)
    for i in range(0, len_vs):
        next_sum = add(running_sum, vs[i])
        arrows.append(Arrow3D(next_sum, running_sum))
        running_sum = next_sum
    print(f"sum = {running_sum}")
    draw3d(*arrows)



def exercise_3_6():
    def scale(scalar, vector):
        return tuple([ scalar * i  for i in vector])

def exercise_3_7():
    u = (8, -5, -1)
    v = (2, 3, 2)
    w = add(u, scale(0.5, subtract(v, u)))
    draw3d(Points3D(u, v, w))

def vector_generator(n):
    # combination with repetition
    for x in range(1, n):
        for y in range(1, x + 1):
            for z in range(1, y + 1):
                yield (1.0* x, 1.0* y, 1.0 * z)

def vector_generator_2(n):
    # combination with repetition
    for x in range(1, n):
        for y in range(x, n):
            for z in range(y, n):
                yield (1.0* x, 1.0* y, 1.0 * z)

def exercise_3_9():
    max_to_check = 100
    generator = vector_generator(max_to_check)
    hits = 0
    checks = 0
    for v in generator:
        v_length = length(v)
        checks += 1
        if ceil(v_length) == floor(v_length):
            print(f"length({v}) = {v_length}")
            hits += 1
    print(f"total hits= {hits}")
    print(f"total checks= {checks}")


def exercise_3_10():
    v = (-1, -1, 2)
    diff = 0.0000000000000000000000000001
    begin = 0.1
    end = 1
    while begin < end:
        mid = (begin + end) / 2
        nv = scale(mid, v)
        nl = length(nv)
        print(f"scalar={mid} length({nv} = {nl})")
        if abs(1 - nl) <= diff:
            break
        if nl > 1:
            end = mid
        else:
            begin = mid
    # scalar=0.408248290463863 length((-0.408248290463863, -0.408248290463863, 0.816496580927726) = 1.0)

def exercise_3_10_2():
    v = (-1, -1, 2)
    v_len = length(v)
    n_v = scale(1/v_len, v)
    print(f"scalar = {1/v_len}, length({n_v}) = {length(n_v)}")
    # scalar =0.4082482904638631, length((-0.4082482904638631, -0.4082482904638631, 0.8164965809277261)) = 1.0

def exercise_3_15():
    u = (3, 0)
    v_length = 7
    vs = [(v_length * cos(i), v_length * sin(i)) for i in range(1, 4)]
    for v in vs:
        print(f"length({u}) = {length(u)}, length({v}) = {length(v)}")
        print(f"dot({u}, {v}) = {dot(u, v)}")
    draw2d(Points2D(*vs))

def degree_to_radian(degree):
    return (degree * pi) / 180

def radian_to_degree(radian):
    return (radian * 180) / pi

def exercise_3_16():
    u_length = 3.61
    v_length = 1.44
    radiant = degree_to_radian(101.3)
    print(f"dot(u, v) = {u_length * v_length * cos(radiant)}")

def exercise_3_17():
    u = (3, 4)
    v = (4, 3)

    polar_u = to_polar(u)
    polar_v = to_polar(v)

    print(f"diff: {polar_u[1] - polar_v[1]}")

def exercise_3_18():
    u = (1, 1, 1)
    v = (-1, -1, 1)

    dp = dot(u, v)
    radiant = acos(dp / (length(u) * length(v)))
    print(f"angle between {u} {v} is: { radian_to_degree(radiant) } ")

def cross(u, v):
    ux, uy, uz = u
    vx, vy, vz = v
    return (uy*vz - uz*vy, uz*vx - ux*vz, ux * vy - uy*vx)


def exercise_3_22():
    u = (1, -2, 1)
    v = (-6, 12, -6)
    print(cross(u, v))
    draw3d(Arrow3D(u, color=red), Arrow3D(v, color=green))

def exercise_3_24():
    u = (1, 0, 1)
    v = (-1, 0, 0)
    w = cross(u, v)
    print(f"w= {w}")
    draw3d(Arrow3D(u, color=red), Arrow3D(v, color=blue), Arrow3D(w, color=green))

def exercise_3_27():
    top = (0, 0, 1)
    bottom = (0, 0, -1)
    x_y_plane = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]

    edges = [Segment3D(top, v) for v in x_y_plane] + \
            [Segment3D(bottom, v) for v in x_y_plane] + \
            [Segment3D(x_y_plane[i], x_y_plane[(i + 1) % 4]) for i in range(0, 4)]
    draw3d(*edges)


exercise_3_27()
