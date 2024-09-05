from vector_drawing import *
from vectors import *
from math import pi, cos, sin, tan, atan, atan2
from random import uniform

def exercise_2_9():
    v = (1, 6)
    w = (2, 8)

    draw(
        Arrow(v, color=blue),
        Arrow(w, color=red),
        Arrow(add(v, w), color=black),
        Arrow(subtract(v, w), color=green),
        Arrow(subtract(w, v), color=purple)
    )

def exercise_2_11():
    dino_vectors = [(6, 4), (3, 1), (1, 2), (-1, 5), (-2, 5), (-3, 4), (-4, 4),
                    (-5, 3), (-5, 2), (-2, 2), (-5, 1), (-4, 0), (-2, 1), (-1, 0), (0, -3),
                    (-1, -4), (1, -4), (2, -3), (1, -2), (3, -1), (5, 1)
                    ]
    offset=  (15, 15)
    translations = [multiply((x, y), offset)  for x in range(-5, 5) for y in range(-5, 5)]
    dinos = [Polygon(*translate(t, dino_vectors)) for t in translations]
    draw(* dinos)



def exercise_2_15():
    dino_vectors = [(6, 4), (3, 1), (1, 2), (-1, 5), (-2, 5), (-3, 4), (-4, 4),
                    (-5, 3), (-5, 2), (-2, 2), (-5, 1), (-4, 0), (-2, 1), (-1, 0), (0, -3),
                    (-1, -4), (1, -4), (2, -3), (1, -2), (3, -1), (5, 1)
                    ]
    print(max(dino_vectors, key=length))


def exercise_2_16():
    w = (sqrt(2), sqrt(3))
    v = scale(pi, w)
    draw(Arrow(w, color=blue), Arrow(v, color=red))


def uniform_r():
    return uniform(-1, 1)

def uniform_s():
    return uniform(-1, 1)

def exercise_2_19():
    u = (-1, 1)
    v = (1, 1)
    points = [add(scale(uniform_r(), u), scale(uniform_s(), v)) for _ in range(1, 1000)]
    draw(Points(*points))

def distance(v1, v2):
    return sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)


def perimeter(*vectors):
    ven_length = len(vectors)
    return sum([distance(vectors[i%ven_length], vectors[(i+1)%ven_length])  for i in range(0, ven_length)])


def exercise_2_25():
    dino_vectors = [(6, 4), (3, 1), (1, 2), (-1, 5), (-2, 5), (-3, 4), (-4, 4),
                    (-5, 3), (-5, 2), (-2, 2), (-5, 1), (-4, 0), (-2, 1), (-1, 0), (0, -3),
                    (-1, -4), (1, -4), (2, -3), (1, -2), (3, -1), (5, 1)
                    ]
    print(perimeter(*dino_vectors))

def degree_to_radian(degree):
    return (degree * pi) / 180

def radian_to_degree(radian):
    return (radian * 180) / pi

def exercise_2_26():
    u = (1, -1)
    for n in range(13, 1, -1):
        for m in range(12, 0, -1):
            if n <= m:
                continue
            if distance(u, (n, m)) == 13:
                print(f"({n}, {m})")

def exercise_2_27():
    u = (-1.34, 2.68)
    print(length(u))

def to_cartesian(polar):
    length = polar[0]
    radian = polar[1]
    return (length * cos(radian), length * sin(radian))

def exercise_2_29():
    print(to_cartesian((15, degree_to_radian(37))))

def exercise_2_30():
    print(to_cartesian((8.5, degree_to_radian(125))))

def exercise_2_34():
    print(sqrt(1-0.643**2))

def exercise_2_35():
    print(degree_to_radian(116.57))
    print(tan(degree_to_radian(116.57)))

def exercise_2_36():
    radian = (10* pi) / 6
    print(radian)
    print(radian_to_degree(radian))

def exercise_2_37():
    polar_coords = [(cos((5*x*pi)/500), 2*pi*x/1000.0) for x in range(0, 1000)]
    vertices = [ to_cartesian(p)  for p in polar_coords]
    draw(Polygon(*vertices), nice_aspect_ratio=False)

def exercise_2_38():
    target_sin = 3/sqrt(13)
    begin = pi/2
    end = pi
    while begin < end:
        mid = (end + begin) / 2
        print(f"radian: {mid}")
        print(f"degree: {radian_to_degree(mid)}")
        if abs(sin(mid) - target_sin) <= 0.0000000001:
            break
        if sin(mid) > target_sin:
            begin = mid
        else:
            end = mid

def exercise_2_39():
    print(radian_to_degree(atan(-3/2)))


def exercise_2_41():
    print(f"mouse = {radian_to_degree(atan(1/3))}")
    print(f"toe = {radian_to_degree(atan(1))}")
    print(f"tip= {90 -radian_to_degree(atan(1/3)) - radian_to_degree(atan(1))}")

exercise_2_41()

