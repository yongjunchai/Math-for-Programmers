import math
import time

import matplotlib.pyplot as plt
import numpy as np

def plot_function(f,tmin,tmax,tlabel=None,xlabel=None,axes=False, **kwargs):
    ts = np.linspace(tmin,tmax,1000)
    if tlabel:
        plt.xlabel(tlabel,fontsize=18)
    if xlabel:
        plt.ylabel(xlabel,fontsize=18)
    plt.plot(ts, [f(t) for t in ts], **kwargs)
    if axes:
        total_t = tmax-tmin
        plt.plot([tmin-total_t/10,tmax+total_t/10],[0,0],c='k',linewidth=1)
        plt.xlim(tmin-total_t/10,tmax+total_t/10)
        xmin, xmax = plt.ylim()
        plt.plot([0,0],[xmin,xmax],c='k',linewidth=1)
        plt.ylim(xmin,xmax)

def plot_volume(f,tmin,tmax,axes=False,**kwargs):
    plot_function(f,tmin,tmax,tlabel="time (hr)", xlabel="volume (bbl)", axes=axes, **kwargs)

def plot_flow_rate(f,tmin,tmax,axes=False,**kwargs):
    plot_function(f,tmin,tmax,tlabel="time (hr)", xlabel="flow rate (bbl/hr)", axes=axes, **kwargs)

# def average(f, t1, t2):
#     return (f(t2) - f(t1)) / (t2 - t1)
def average(f, t1, t2):
    step1 =  f(t2) - f(t1)
    step2 =  (t2 - t1)
    step3 = step1 / step2
    return step3

def volume(t):
    return (t-4)**3 / 64 + 3.3

def flow_rate(t):
    return 3*(t-4)**2 / 64

def decreasing_volume(t):
    if t < 5:
        return 10 - (t**2)/5
    else:
        return 0.2*(10-t)**2


def exercise_8_1():
    startMileage = 77641
    endMileage = 77905
    print((endMileage - startMileage) / 4.5)

def secant_line(f, x1, x2):
    def line(x):
        return ((f(x2) - f(x1)) / (x2-x1)) * (x - x1) + f(x1)
    return line

def plot_secant(f, x1, x2, color='k'):
    line = secant_line(f, x1, x2)
    plot_function(line, x1, x2, color=color)
    plot_function(f, x1, x2, color=color)
    plt.scatter([x1, x2], [f(x1), f(x2)], color=color)

def exercise_8_3():
    plot_secant(volume, 3, 8, color="red")
    plot_volume(volume, 0, 10, color="green")
    plt.show()

def interval_flow_rate(v, t1, t2, dt):
    return [(t, average(v, t, t + dt)) for t in np.arange(t1, t2, dt)]

def plot_interval_flow_rate(f, t1, t2, dt):
    rates = interval_flow_rate(f, t1, t2, dt)
    print(f"rates = {rates}")
    xs = [x for x, _ in rates]
    ys = [y for _, y in rates]
    plt.scatter(xs, ys)

def exercise_8_4():
    plot_interval_flow_rate(decreasing_volume, 0, 10, 0.5)
    plt.show()

def linear_volume_function(t):
    return 25 * t + 5.0

def exercise_8_5():
    plot_interval_flow_rate(linear_volume_function, 0, 10, 0.5)
    plt.show()


def exercise_8_6():
    print(volume(1))
    line = secant_line(volume, 0.999, 1.001)
    print(line(1))

def exercise_8_7():
    print(average(volume, 7.9, 8.1))
    print(average(volume, 7.99, 8.01))
    print(average(volume, 7.999, 8.001))
    print(average(volume, 7.9999, 8.0001))
    print(average(volume, 7.99999, 8.00001))
    print(average(volume, 7.999999, 8.000001))

def sign(x):
    return x /abs(x)

def exercise_8_8():
    # plot_function(sign, -10, 10)
    # plt.show()
    print(average(sign, -0.1, 0.1))
    print(average(sign, -0.01, 0.01))
    print(average(sign, -0.001, 0.001))
    print(average(sign, -0.0001, 0.0001))
    print(average(sign, -0.00001, 0.00001))
    print(average(sign, -0.000001, 0.000001))
    print(average(sign, -0.0000001, 0.0000001))

def small_volume_change(q, t, dt):
    return q(t) * dt

def volume_change(q, t1, t2, dt):
    return sum(small_volume_change(q, t, dt) for t in np.arange(t1, t2, dt))

def exercise_8_9():
    print(volume_change(flow_rate, 0, 6, 0.00001))
    print(volume(6) - volume(0))
    print(volume_change(flow_rate, 6, 10, 0.00001))
    print(volume(10) - volume(6))

def get_volume_function(q,v0,digits=6):
    def volume_function(T):
        tolerance = 10 ** (-digits)
        dt = 1
        approx = v0 + volume_change(q,0,T,dt)
        for i in range(0,digits*2):
            dt = dt / 10
            print(f"dt = {dt}")
            next_approx = v0 + volume_change(q,0,T,dt)
            if abs(next_approx - approx) < tolerance:
                return round(next_approx,digits)
            else:
                approx = next_approx
        raise Exception("Did not converge!")
    return volume_function

def measure_time():
    v = get_volume_function(flow_rate,2.3,digits=6)
    start_time = time.perf_counter()
    print(v(1))
    end_time = time.perf_counter()
    print(f"time used: {end_time - start_time}")

measure_time()

# exercise_8_9()

# plot_function(math.sin, - 20 * math.pi, 20 * math.pi, axes=True)
# plt.show()
