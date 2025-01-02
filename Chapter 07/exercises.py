# Import a library of functions called 'pygame'
import datetime
import logging
import numpy as np
import pygame
from IPython.utils.capture import capture_output

import vectors
from math import pi, sqrt, cos, sin, atan2
from random import randint, uniform
from linear_solver import do_segments_intersect
import sys


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# DEFINE OBJECTS OF THE GAME


# def do_segments_intersect(s1, s2):
#     pass

MAX_SIZE = 10

class PolygonModel:
    def __init__(self,points):
        self.points = points
        self.x = 0
        self.y = 0
        self.rotation_angel = 0
        # the radiant degree to move in one second
        self.angular_velocity = 0

    def move(self, milliseconds):
        self.rotation_angel += self.angular_velocity * milliseconds/1000.0

    def transformed(self):
        rotated = [vectors.rotate2d(self.rotation_angel, p)  for p in self.points]
        return [(self.x + v[0], self.y + v[1]) for v in rotated]

    def segments(self):
        points = self.transformed()
        total_points = len(points)
        return [(points[i], points[(i + 1) % total_points]) for i in range(0, total_points)]

    def does_intersect(self, other_segment):
        for segment in self.segments():
           if do_segments_intersect(segment, other_segment):
               return True
        return False

    def does_collide(self, other):
        for segment in other.segments():
            if self.does_intersect(segment):
                return True
        return False


class Ship(PolygonModel):
    def __init__(self):
        super().__init__([(0.5, 0), (-0.25, 0.25), (-0.25, -0.25)])

    def laser_segment(self):
        dist = 10 * sqrt(2)
        start = self.transformed()[0]
        end = (dist * cos(self.rotation_angel), dist * sin(self.rotation_angel))
        return start, end

class Asteroid(PolygonModel):
    def __init__(self):
        sides = randint(3, 20)
        super().__init__([vectors.to_cartesian((uniform(0.1, 1), i * 2 * pi / sides)) for i in range(0, sides)])
        self.angular_velocity = uniform(-5*pi, 5*pi)
        self.x = randint(-(MAX_SIZE - 1), MAX_SIZE - 1)
        self.y = randint(-(MAX_SIZE - 1), MAX_SIZE - 1)
        self.collided = False

width, height = 1024, 768
RED =   (255, 0, 0)
GREEN = (0, 255, 0)
BLUE =  (0, 0, 255)
WHITE = (255, 255, 255)

def to_pixel(x, y):
    return width / 2 + (width / 2) * (x / MAX_SIZE), height / 2 - (height / 2) * (y / MAX_SIZE)

def draw_poly(screen, polygon_model: PolygonModel, color=GREEN):
    pixel_points = [to_pixel(*p) for p in polygon_model.transformed()]
    pygame.draw.aalines(screen, color, True, pixel_points, 10)

def draw_segment(screen, v1,v2,color=RED):
    pygame.draw.aaline(screen, color, to_pixel(*v1), to_pixel(*v2), 10)


# INITIALIZE GAME STATE

asteroid_count = 100

ship = Ship()
asteroids = [Asteroid() for _ in range(0, asteroid_count)]

grids = dict()
GRID_WIDTH = 2
MAX_GRID = 2 * MAX_SIZE / GRID_WIDTH - 1
GRIDS_PER_ROW = 2 * MAX_SIZE / GRID_WIDTH
MIN_X, MAX_X = -10, 10
MIN_Y, MAX_Y = -10, 10
def get_corners(asteroid: Asteroid):
    points = asteroid.transformed()
    mins = tuple(map(min, zip(*points)))
    maxs = tuple(map(max, zip(*points)))
    return [mins, maxs, (mins[0], maxs[1]), (maxs[0], mins[1])]

def get_index(x, y):
    x_diff = x - MIN_X
    y_diff = y - MIN_Y
    row = 0
    column = 0
    if y_diff % GRID_WIDTH == 0:
        row = y_diff // GRID_WIDTH
    else:
        row = y_diff // GRID_WIDTH + 1
    if x_diff % GRID_WIDTH == 0:
        column = x_diff // GRID_WIDTH
    else:
        column = x_diff // GRID_WIDTH + 1
    if row >= MAX_GRID:
        row = MAX_GRID - 1
    if column >= MAX_GRID:
        column = MAX_GRID - 1
    return int(row * GRIDS_PER_ROW + column)

def assign_grid(asteroid: Asteroid):
    corners = get_corners(asteroid)
    target_grids = set()
    for corner in corners:
        target_grids.add(get_index(*corner))
    for target_grid in target_grids:
        cur_asteroids = grids.get(target_grid)
        if cur_asteroids is None:
            cur_asteroids = list()
            grids[target_grid] = cur_asteroids
        cur_asteroids.append(asteroid)

def remove_collide_asteroids(audit_mode=True):
    logging.debug("start remove_collide_asteroids")
    for input_asteroids in grids.values():
        count = len(input_asteroids)
        logging.debug(f"before process gird of size: {count}")
        for i in range(0, count):
            for j in range(i +1, count):
                if input_asteroids[i].does_collide(input_asteroids[j]):
                    input_asteroids[i].collided = True
                    input_asteroids[j].collided = True
        logging.debug(f"done processing grid of size: {count}")
        logging.debug(f"before remove from lis of size {len(input_asteroids)}")
        for asteroid in input_asteroids:
            if asteroid.collided:
                if not audit_mode:
                    input_asteroids.remove(asteroid)
                    logging.warning(f"{datetime.datetime.now()} remove collided asteroid from grid: {asteroid.points}")
        logging.debug(f"after remove from lis of size {len(input_asteroids)}")
    logging.info("processed all grids")
    to_remove = [s.points for s in asteroids if s.collided is True]
    logging.debug(to_remove)
    for asteroid in asteroids:
        if asteroid.collided:
            if not audit_mode:
                asteroids.remove(asteroid)
                logging.info(f"{datetime.datetime.now()} remove collided asteroid from main list: {asteroid.points}")
    logging.debug("end remove_collide_asteroids")

def main():
    pygame.init()
    screen = pygame.display.set_mode([width,height])
    pygame.display.set_caption("Asteroids!")
    done = False
    clock = pygame.time.Clock()
    count = 0
    for asteroid in asteroids:
        assign_grid(asteroid)

    while not done:
        clock.tick(300)
        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                done=True # Flag that we are done so we exit this loop
        # UPDATE THE GAME STATE
        milliseconds = clock.get_time()
        keys = pygame.key.get_pressed()
        for asteroid in asteroids:
            asteroid.move(milliseconds)
        remove_collide_asteroids(count < 30)


        if keys[pygame.K_LEFT]:
            ship.rotation_angel += 0.2 * pi * milliseconds /1000
        if keys[pygame.K_RIGHT]:
            ship.rotation_angel -= 0.2 * pi * milliseconds / 1000
        count += 1
        logging.warning(f"{count}: {milliseconds}ms")
        # DRAW THE SCENE
        screen.fill(WHITE)
        laser = ship.laser_segment()
        draw_poly(screen, ship)
        if keys[pygame.K_SPACE]:
            draw_segment(screen, *laser)

        for asteroid in asteroids:
            if keys[pygame.K_SPACE] and asteroid.does_intersect(laser):
                    asteroids.remove(asteroid)
            else:
                draw_poly(screen, asteroid)

        pygame.display.flip()
    pygame.quit()


def standard_form(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    a = y2 - y1
    b = x1 - x2
    c = x1 * y2 - y1 * x2
    return a, b, c

def intersection(u1, u2, v1, v2):
    a1, b1, c1 = standard_form(u1, u2)
    a2, b2, c2 = standard_form(v1, v2)
    m = np.array(((a1, b1), (a2, b2)))
    c = np.array((c1, c2))
    return np.linalg.solve(m, c)

def segment_checks(s1, s2):
    u1, u2 = s1
    v1, v2 = s2
    l1, l2 = vectors.distance(*s1), vectors.distance(*s2)
    x, y = intersection(u1, u2, v1, v2)
    intersect = (x, y)
    return [
        vectors.distance(u1, intersect) <= l1,
        vectors.distance(u2, intersect) <= l1,
        vectors.distance(v1, intersect) <= l2,
        vectors.distance(v2, intersect) <= l2
    ]

def exercise_7_11():
    a, b, c = standard_form((3, 0), (3, 1))
    print(f"{a}, {b}, {c}")

def exercise_7_12():
    s1 = ((3, -3), (3, 3))
    s2 = ((4, 0), (10, 0))
    print(segment_checks(s1, s2))


def plane_equation(p1, p2, p3):
    parallel1 = vectors.subtract(p1, p2)
    parallel2 = vectors.subtract(p2, p3)
    a, b, c = vectors.cross(parallel1, parallel2)
    d = vectors.dot((a, b, c), p3)
    return a, b, c, d

def exercise_7_18():
    print(plane_equation((1, 1, 1), (3, 0, 0), (0, 0, 3)))

def exercise_7_25():
    matrix = np.array(((0, 0, 0, 0, 1), (0, 1, 0, 0, 0), (0, 0, 0, 1, 0), (1, 0, 0, 0, 0), (1, 1, 1, 0, 0)))
    vector = np.array((3, 1, -1, 0, -2))
    print(np.linalg.solve(matrix, vector))


def exercise_7_26():
    matrix = np.array(((1, 1, -1), (0, 2, -1), (1, 0, 1)))
    vector = np.array((-1, 3, 2))
    r_matrix = np.linalg.inv(matrix)
    print(f"result by the revert identity matrix: {np.matmul(r_matrix, vector)}")
    print(f"result by the solver of numpy: {np.linalg.solve(matrix, vector)}")


# main()
exercise_7_26()















