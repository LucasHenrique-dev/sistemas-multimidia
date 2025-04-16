from OpenGL.GLU import *
from OpenGL.GL import *
import numpy as np
import pygame

# ========== Funções de classificação ==========
def linear(x): return 0.5 * x + 0.2
def quadratic(x): return 2 * x**2 - 0.5 * x - 0.6
def sine(x): return 0.2 * np.sin(3 * x) + 0.2

boundary_functions = [("Linear", linear), ("Quadrática", quadratic), ("Senoidal", sine)]

# ========== Redes Neurais ==========
class Perceptron:
    def __init__(self, lr=0.01):
        self.w = np.random.randn(3)
        self.lr = lr
        self.error = 1E10

    def predict(self, inputs):
        return np.sign(np.dot(self.w, inputs))

    def train(self, inputs, label):
        prediction = self.predict(inputs)
        error = label - prediction
        self.w += self.lr * error * np.array(inputs)
        self.error = error

class MLP:
    def __init__(self, input_size=2, hidden_size=256, lr=0.01):
        self.lr = lr
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w1 = np.random.randn(hidden_size, input_size + 1) * 0.1
        self.w2 = np.random.randn(1, hidden_size + 1) * 0.1
        self.error = []

    def _activation(self, x):
        return np.maximum(0, x)

    def _activation_deriv(self, x):
        return np.where(x > 0, 1, 0)

    def predict(self, inputs):
        x = np.array(inputs)
        x = np.append(x, 1)  # bias
        self.z1_raw = np.dot(self.w1, x)
        self.z1 = self._activation(self.z1_raw)
        self.z1 = np.append(self.z1, 1)  # bias
        self.out = np.dot(self.w2, self.z1)
        return 1 if self.out >= 0 else -1

    def train(self, inputs, label):
        x = np.array(inputs)
        x = np.append(x, 1)  # bias
        z1_raw = np.dot(self.w1, x)
        z1 = self._activation(z1_raw)
        z1_bias = np.append(z1, 1)  # bias na camada oculta

        out = np.dot(self.w2, z1_bias)
        error = label - out  # MELHOR ajuste do erro (contínuo)
        self.error = error

        # Atualização dos pesos da camada de saída
        self.w2 += self.lr * error * z1_bias

        # Backpropagation para a camada oculta
        dz1 = self._activation_deriv(z1_raw)
        d_hidden = error * self.w2[0][:-1] * dz1
        self.w1 += self.lr * d_hidden[:, np.newaxis] @ x[np.newaxis, :]

# ========== Geração de pontos ==========
def generate_points(n, func):
    pts = []
    for _ in range(n):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        label = 1 if y > func(x) else -1
        pts.append((x, y, label))
    return pts

# ========== Desenho OpenGL ==========
def draw_points(points, classify_func):
    glPointSize(20)
    glBegin(GL_POINTS)
    for x, y, label in points:
        true_label = 1 if y > classify_func(x) else -1
        if true_label == 1:
            glColor3f(0, 1, 0)
        else:
            glColor3f(1, 0, 0)
        glVertex2f(x, y)
    glEnd()

def draw_decision_boundary(model):
    glLineWidth(8)
    glBegin(GL_LINE_STRIP)
    glColor3f(0.0, 0.8, 1.0)
    for x in np.linspace(-1, 1, 200):
        try:
            y = None
            if isinstance(model, Perceptron):
                w = model.w
                if w[1] != 0:
                    y = -(w[0]*x + w[2]) / w[1]
            elif isinstance(model, MLP):
                y = find_boundary_y_mlp(model, x)
            if y is not None and -1 <= y <= 1:
                glVertex2f(x, y)
        except:
            pass
    glEnd()

def find_boundary_y_mlp(model, x, samples=100):
    y_vals = np.linspace(-1, 1, samples)
    preds = [model.predict([x, y]) for y in y_vals]
    for i in range(1, len(preds)):
        if preds[i] != preds[i - 1]:
            return y_vals[i]
    return None

def draw_true_function(func):
    glLineWidth(8)
    glBegin(GL_LINE_STRIP)
    glColor3f(1.0, 1.0, 0.0)
    for x in np.linspace(-1, 1, 200):
        y = func(x)
        if -1 <= y <= 1:
            glVertex2f(x, y)
    glEnd()

def draw_text(x, y, text, font_size=16):
    font = pygame.font.SysFont('Arial', font_size)
    surface = font.render(text, True, (255, 255, 255), (0, 0, 0))
    text_data = pygame.image.tostring(surface, "RGBA", True)
    width, height = surface.get_size()

    # Configurar OpenGL para desenhar 2D
    glRasterPos2f(x, y)
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

def draw_text_block(lines, x=-0.95, y=0.9, dy=0.1):
    for i, line in enumerate(lines):
        draw_text(x, y - dy * i, line, font_size=32)

# ========== Inicialização ==========
WIDTH, HEIGHT = 1400, 800
pygame.init()
pygame.font.init()
font = pygame.font.SysFont("Arial", 36, bold=True)
pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Aprendizado de Rede Neural com OpenGL")
gluOrtho2D(-1, 1, -1, 1)
clock = pygame.time.Clock()

current_func_idx = 0
current_func_name, current_func = boundary_functions[current_func_idx]
learning_rate = 0.01
n_points = 100
use_mlp = False
model = Perceptron(learning_rate)
points = generate_points(n_points, current_func)

# ========== Loop principal ==========
running = True
while running:
    glClear(GL_COLOR_BUFFER_BIT)

    # Treinamento
    for x, y, label in points:
        inputs = [x, y, 1] if isinstance(model, Perceptron) else [x, y]
        model.train(inputs, label)


    # Desenho
    draw_true_function(current_func)
    draw_decision_boundary(model)
    draw_points(points, current_func)

    glLoadIdentity()
    glColor3f(1, 1, 1)
    erro_modelo = model.error[0] if use_mlp else model.error

    draw_text_block([
        f"Função: {current_func_name}",
        f"Modelo: {'Perceptron' if not use_mlp else 'MLP'}",
        f"Erro: {erro_modelo:.4f} | Pontos: {n_points}",
        "Verde = +1 | Vermelho = -1",
        "Amarelo = Função real",
        "Azul = Fronteira aprendida"
    ])

    pygame.display.flip()
    clock.tick(60)

    # Entrada
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                n_points = min(n_points + 50, 10000)
                points = generate_points(n_points, current_func)
            elif event.key == pygame.K_DOWN:
                n_points = max(n_points - 50, 50)
                points = generate_points(n_points, current_func)
            elif event.key == pygame.K_LEFT:
                current_func_idx = (current_func_idx - 1) % len(boundary_functions)
                current_func_name, current_func = boundary_functions[current_func_idx]
                points = generate_points(n_points, current_func)
            elif event.key == pygame.K_RIGHT:
                current_func_idx = (current_func_idx + 1) % len(boundary_functions)
                current_func_name, current_func = boundary_functions[current_func_idx]
                points = generate_points(n_points, current_func)
            elif event.key == pygame.K_SPACE:
                model = MLP(lr=learning_rate) if use_mlp else Perceptron(learning_rate)
            elif event.key == pygame.K_z:
                use_mlp = not use_mlp
                model = MLP(lr=learning_rate) if use_mlp else Perceptron(learning_rate)
                points = generate_points(n_points, current_func)

pygame.quit()
