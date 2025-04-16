import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

# Define the vertices of the cube
vertices = (
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, -1),
    (1, -1, 1),
    (-1, 1, 1),
    (-1, 1, -1),
    (-1, -1, -1),
    (-1, -1, 1)
)

# Define the edges of the cube
edges = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7)
)

# Define the colors for each vertex
colors = (
    (1, 0, 0),      # Red
    (1, 1, 0),      # Yellow
    (0, 1, 0),      # Green
    (1, 0.5, 0),    # Orange
    (1, 0, 1),      # Magenta
    (0, 0, 1),      # Blue
    (0.5, 0, 0.5),  # Purple
    (0, 1, 1)       # Cyan
)

def cube():
    glBegin(GL_QUADS)
    for face in ((0, 1, 2, 3), (3, 2, 6, 7), (7, 6, 5, 4), (4, 5, 1, 0), (0, 3, 7, 4), (1, 2, 6, 5)):
        for vertex_index in face:
            glColor3fv(colors[vertex_index])
            glVertex3fv(vertices[vertex_index])
    glEnd()

    glLineWidth(5.0)    # Set line thickness
    glBegin(GL_LINES)
    for edge in edges:
        v1_index, v2_index = edge
        color1 = colors[v1_index]
        color2 = colors[v2_index]
        # Calcula uma cor mais escura (média com um fator de escurecimento)
        darkening_factor = 0.5
        edge_color = (
            (color1[0] + color2[0]) / 2.0 * darkening_factor,
            (color1[1] + color2[1]) / 2.0 * darkening_factor,
            (color1[2] + color2[2]) / 2.0 * darkening_factor
        )
        glColor3fv(edge_color)
        glVertex3fv(vertices[v1_index])
        glVertex3fv(vertices[v2_index])
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0.0, -5)

    glRotatef(0, 0, 0, 0)

    rotation_speed = 1.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(rotation_speed, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cube()
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()