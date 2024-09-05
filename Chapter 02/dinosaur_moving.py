from vector_drawing import *

def add(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1])


dino_vectors = [(6,4), (3,1), (1,2), (-1,5), (-2,5), (-3,4), (-4,4),
    (-5,3), (-5,2), (-2,2), (-5,1), (-4,0), (-2,1), (-1,0), (0,-3),
    (-1,-4), (1,-4), (2,-3), (1,-2), (3,-1), (5,1)
]



steps = 8
moves = [(-steps, 0), (-steps, 0),
         (0, -steps), (0, -steps),
         (steps, 0), (steps, 0),
         (steps, 0), (steps, 0),
         (0, steps), (0, steps),
         (- steps, 0), (-steps, 0),
         ]

count = 0
while True:
    #dino_vectors = [add(moves[count % len(moves)], v) for v in dino_vectors]
    draw(
        Points(*dino_vectors),
        Polygon(*dino_vectors, fill=True),
        fixed_limits=True,
        nice_aspect_ratio=False
    )
    count += 1

