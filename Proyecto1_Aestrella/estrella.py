import pygame

pygame.init()
# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos")

#Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
RED = (217, 136, 128)
BLUE = (133, 193, 233)
CURRENT_BLUE = (27, 79, 114)

FILAS = 11


class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.nodo_dependiente = None
        self.numeracion_nodo = None
        self.texto = None

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

        texto = self.texto

        if texto is not None:
            fuente = pygame.font.SysFont("Arial", int(self.ancho / 5))
            palabras = texto.split("-")
            y = self.y
            for i in range(len(palabras)):
                superficie_texto = fuente.render(palabras[i], True, (0, 0, 0))
                ventana.blit(superficie_texto, (self.x, y))
                y += int(self.ancho / 5) + 5

            if self.es_visitado() or self.es_fin() or self.es_inicio() or self.color == VERDE:
                superficie_texto = fuente.render("OK", True, (0, 0, 0))
                ventana.blit(superficie_texto, ((self.x + self.ancho) - int(self.ancho / 5),
                                                (y - int(self.ancho / 5) - 5) if FILAS > 11 else y))

    def set_nodo_dependiente(self, nodo):
        self.nodo_dependiente = nodo

    def get_nodo_dependiente(self):
        return self.nodo_dependiente

    def hacer_camino(self):
        self.color = VERDE

    def get_numeracion(self):
        return self.numeracion_nodo

    def set_numeracion(self, numeracion):
        self.numeracion_nodo = numeracion

    def hacer_visita(self):
        self.color = RED

    def es_visitado(self):
        return self.color == RED

    def hacer_visita_neighbor(self):
        self.color = BLUE

    def get_text(self):
        return self.texto

    def set_text(self, texto):
        self.texto = texto

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def hacer_current(self):
        self.color = CURRENT_BLUE