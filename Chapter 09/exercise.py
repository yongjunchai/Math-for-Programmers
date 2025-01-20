import vectors
from draw2d import *
from draw3d import *
import math

def eulers_method(s0, v0, a, total_time, step_count):
    ss = [s0]
    s = s0
    v = v0
    dt = total_time / step_count
    for _ in range(0, step_count):
        s = vectors.add(s, vectors.scale(dt, v))
        v = vectors.add(v, vectors.scale(dt, a))
        ss.append(s)
    return ss

def eulers_method_overapprox(s0, v0, a, total_time, step_count):
    ss = [s0]
    s = s0
    v = v0
    dt = total_time / step_count
    for _ in range(0, step_count):
        v = vectors.add(v, vectors.scale(dt, a))
        s = vectors.add(s, vectors.scale(dt, v))
        ss.append(s)
    return ss


def pairs(lst):
    return list(zip(lst[:-1], lst[1:]))

def exercise_9_2():
    s0 = (0, 0)
    v0 = (1, 0)
    a = (0, 0.2)
    total_time = 10
    step_count = 10
    ss = eulers_method(s0, v0, a, total_time, step_count)
    print(ss)
    draw2d(Points2D(*ss), *[Segment2D(t,h,color='C0') for (h,t) in pairs(ss)])


def exercise_9_3():
    s0 = (0, 0)
    v0 = (1, 0)
    a = (0, 0.2)
    total_time = 10
    step_count = 10
    ss1 = eulers_method(s0, v0, a, total_time, step_count)
    ss2 = eulers_method_overapprox(s0, v0, a, total_time, step_count)
    draw2d(Points2D(*ss1), *[Segment2D(t,h,color='C0') for (h,t) in pairs(ss1)])
    draw2d(Points2D(*ss2), *[Segment2D(t,h,color='C1') for (h,t) in pairs(ss2)])
    plt.show()

def exercise_9_4():
    angle = 20 * math.pi / 180
    velocity = 30
    s0 = (0, 1.5)
    v0 = (velocity * math.cos(angle), velocity * math.sin(angle))
    a = (0, -9.81)
    ss = eulers_method(s0, v0, a, 3, 100)
    print(ss)
    draw2d(Points2D(*ss), *[Segment2D(t,h,color='C1') for (h,t) in pairs(ss)])
    plt.show()

def find_angle():
    def find_x(ss):
        for i in range(0, len(ss)):
            if ss[i][1] < 0:
                # find the one which is closer to x-axe
                if abs(ss[i][1]) < abs(ss[i-1][1]):
                    return ss[i][0]
                else:
                    return ss[i-1][0]
        return 0

    def get_x_length(angle: int):
        angle = angle * math.pi / 180
        velocity = 30
        s0 = (0, 0)
        v0 = (velocity * math.cos(angle), velocity * math.sin(angle))
        a = (0, -9.81)
        ss = eulers_method(s0, v0, a, 8, 1000)
        return find_x(ss)

    prev_x = get_x_length(0)
    angel_1 = 0
    for i in range(1, 90):
        x = get_x_length(i)
        if x < prev_x:
            break
        prev_x = x
        angel_1 = i
    x_1 = prev_x
    angel_2= 90
    prev_x = get_x_length(90)
    for i in range(1, 90):
        x = get_x_length(90 - i)
        if x < prev_x:
            break
        prev_x = x
        angel_2 = 90 - i

    def get_x(val):
        return val[0]

    return max((x_1, angel_1), (prev_x, angel_2), key=get_x),

def exercise_9_5():
    t = find_angle()
    max_x, angle = t[0][0], t[0][1]
    print(f"angel: {angle}, x_len: {max_x}")
    angle = angle * math.pi / 180
    velocity = 30
    s0 = (0, 0)
    v0 = (velocity * math.cos(angle), velocity * math.sin(angle))
    a = (0, -9.81)
    ss = eulers_method(s0, v0, a, 8, 100)
    print(ss)
    draw2d(Points2D(*ss), *[Segment2D(t,h,color='C1') for (h,t) in pairs(ss)])
    plt.show()

def exercise_9_6():
    s0 = (0, 0, 0)
    v0 = (1, 2, 0)
    a = (0, -1, 1)
    ss = eulers_method(s0, v0, a, 10, 1000)
    print(ss[len(ss) - 1])
    draw3d(Points3D(*ss))


exercise_9_6()

