# Import a library of functions called 'pygame'
import pygame
import vectors
from math import pi, sqrt, cos, sin, atan2
from random import randint, uniform
from linear_solver import do_segments_intersect
import sys

# DEFINE OBJECTS OF THE GAME

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
        sides = randint(3, 10)
        super().__init__([vectors.to_cartesian((uniform(0.5, 1), i * 2 * pi / sides)) for i in range(0, sides)])
        self.angular_velocity = uniform(-5*pi, 5*pi)
        self.x = randint(-9, 9)
        self.y = randint(-9, 9)

width, height = 1024, 768
RED =   (255, 0, 0)
GREEN = (0, 255, 0)
BLUE =  (0, 0, 255)
WHITE = (255, 255, 255)

def to_pixel(x, y):
    return width / 2 + (width / 2) * (x / 10), height / 2 - (height / 2) * (y / 10)

def draw_poly(screen, polygon_model: PolygonModel, color=GREEN):
    pixel_points = [to_pixel(*p) for p in polygon_model.transformed()]
    pygame.draw.aalines(screen, color, True, pixel_points, 10)


def draw_segment(screen, v1,v2,color=RED):
    pygame.draw.aaline(screen, color, to_pixel(*v1), to_pixel(*v2), 10)


# INITIALIZE GAME STATE

asteroid_count = 100

ship = Ship()
asteroids = [Asteroid() for _ in range(0, asteroid_count)]

def main():
    pygame.init()
    screen = pygame.display.set_mode([width,height])
    pygame.display.set_caption("Asteroids!")
    done = False
    clock = pygame.time.Clock()
    count = 0
    while not done:
        clock.tick(100)
        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                done=True # Flag that we are done so we exit this loop
        # UPDATE THE GAME STATE
        milliseconds = clock.get_time()
        keys = pygame.key.get_pressed()
        for asteroid in asteroids:
            asteroid.move(milliseconds)
        if keys[pygame.K_LEFT]:
            ship.rotation_angel += 0.2 * pi * milliseconds /1000
        if keys[pygame.K_RIGHT]:
            ship.rotation_angel -= 0.2 * pi * milliseconds / 1000
        count += 1
        print(f"{count}: {milliseconds}ms")
        # DRAW THE SCENE
        screen.fill(WHITE)
        for asteroid in asteroids:
            draw_poly(screen, asteroid)
        draw_poly(screen, ship)
        laser = ship.laser_segment()
        if keys[pygame.K_SPACE]:
            draw_segment(screen, *laser)

        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()
