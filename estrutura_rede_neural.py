from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pygame
import math
import random

width, height = 1400, 800
mode = 0  # 0 = Perceptron, 1 = MLP
structures = [
    [3, 2],       # Perceptron simples
    [3, 4, 2]     # MLP com uma camada oculta
]

network_structure = structures[mode]
neuron_positions = []
active_connections = []

# --- Inicializar Pygame para uso de fontes ---
pygame.init()
pygame.font.init()

# --- Função para desenhar texto com OpenGL ---
def draw_text(x, y, text, font_size=16):
    font = pygame.font.SysFont("Arial", font_size)
    surface = font.render(text, True, (255, 255, 255), (0, 0, 0))
    text_data = pygame.image.tostring(surface, "RGBA", True)
    w, h = surface.get_size()
    glRasterPos2f(x, y)
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

# --- Posicionamento dos neurônios ---
def calculate_positions():
    global neuron_positions
    neuron_positions = []
    total_layers = len(network_structure)

    for layer_index, num_neurons in enumerate(network_structure):
        layer = []
        x = -0.8 + (1.6 * layer_index / (total_layers - 1))
        spacing = 1.6 / (num_neurons - 1) if num_neurons > 1 else 0
        for i in range(num_neurons):
            y = -0.8 + i * spacing if num_neurons > 1 else 0
            layer.append((x, y))
        neuron_positions.append(layer)

# --- Desenho dos neurônios (círculos) ---
def draw_circle(x, y, radius=0.05, segments=20):
    glPointSize(20)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        glVertex2f(x + math.cos(angle) * radius, y + math.sin(angle) * radius)
    glEnd()

# --- Conexões entre os neurônios ---
def draw_connections():
    for i in range(len(neuron_positions) - 1):
        for j, src in enumerate(neuron_positions[i]):
            for k, dst in enumerate(neuron_positions[i + 1]):
                if (i, j, k) in active_connections:
                    glColor3f(1.0, 0.2, 0.2)  # Ativa
                else:
                    glColor3f(0.6, 0.6, 0.6)  # Inativa
                glLineWidth(8)
                glBegin(GL_LINES)
                glVertex2f(src[0], src[1])
                glVertex2f(dst[0], dst[1])
                glEnd()

# --- Neurônios (azuis) ---
def draw_neurons():
    for layer in neuron_positions:
        for (x, y) in layer:
            glColor3f(0.2, 0.7, 1.0)
            draw_circle(x, y)

# --- Simulação de treinamento (ativa conexões aleatórias) ---
def simulate_training_step():
    global active_connections
    active_connections = []
    for i in range(len(neuron_positions) - 1):
        src_layer = neuron_positions[i]
        dst_layer = neuron_positions[i + 1]
        for j in range(len(src_layer)):
            k = random.choice(range(len(dst_layer)))
            active_connections.append((i, j, k))

# --- Exibição principal ---
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    simulate_training_step()
    draw_connections()
    draw_neurons()

    # --- Texto informativo ---
    glColor3f(1, 1, 1)
    draw_text(-0.95, 0.9, f"Arquitetura: {'Perceptron' if mode == 0 else 'MLP'}", 32)

    glutSwapBuffers()

# --- Redimensionamento da janela ---
def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)

# --- Temporizador para atualizar a tela ---
def timer(value):
    glutPostRedisplay()
    glutTimerFunc(500, timer, 0)

# --- Teclado: alterna entre modos ---
def keyboard(key, x, y):
    global mode, network_structure
    if key == b' ':
        mode = 1 - mode
        network_structure = structures[mode]
        calculate_positions()
        glutPostRedisplay()

# --- Execução principal ---
def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutCreateWindow(b"Arquitetura da Rede Neural - OpenGL")

    calculate_positions()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutTimerFunc(500, timer, 0)
    glutMainLoop()

if __name__ == "__main__":
    main()
