import pygame

pygame.init()
# Configuraciones iniciales
ANCHO_VENTANA = 700
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

FILAS = 10


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


def init_numeration(grid):
    numeracion = 1
    for i in range(FILAS):
        for j in range(FILAS):
            grid[j][i].set_numeracion(numeracion)
            numeracion += 1
    return grid


def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    grid = init_numeration(grid)
    return grid


def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))


def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)

    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.flip()
    pygame.display.update()


def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col


def reconstruir_camino(nodo, grid, ventana):
    is_creating = True

    while is_creating:
        dibujar(ventana, grid, FILAS, ANCHO_VENTANA)

        if nodo.es_inicio():
            is_creating = False

        if is_creating:
            nodo.hacer_camino()
            print(nodo.get_numeracion(), nodo.get_pos())
            nodo = nodo.get_nodo_dependiente()


def heuristica(actually_nodo, objetivo):
    dist_fila = objetivo.get_pos()[0] - actually_nodo.get_pos()[0]
    dist_col = objetivo.get_pos()[1] - actually_nodo.get_pos()[1]
    return (abs(dist_fila) + abs(dist_col)) * 10


def a_estrella(inicio, final, grid, ventana):
    open_set = [inicio]
    g_score = {}
    f_score = {}

    g_score[inicio] = 0
    f_score[inicio] = heuristica(inicio, final)

    g_lineal = 10
    g_diagonal = 14

    current = inicio
    current.set_text(f"H = {heuristica(current, final)}-G = {g_score[current]}-F = {f_score[current]}")
    last_current = current
    while open_set:
        open_set.remove(current)

        if not current.es_inicio() and not current == final:
            if not last_current.es_inicio() and not last_current == final:
                last_current.hacer_visita()
                last_current.dibujar(ventana)
            current.hacer_current()

        current.dibujar(ventana)
        dibujar_grid(ventana, FILAS, ANCHO_VENTANA)
        pygame.display.flip()
        pygame.display.update()
        pygame.time.delay(100)

        fila, col = current.get_pos()
        print(current.get_numeracion(),
              f"[ h = {heuristica(current, final)}, g = {g_score[current]}, f = {f_score[current]} ] {current.get_pos()}")
        f_score.pop(current)

        if current == final:
            reconstruir_camino(current.get_nodo_dependiente(), grid, ventana)
            return

        movimientos = [
            (-1, -1, g_diagonal),  # Arriba izquierda
            (-1, 0, g_lineal),  # Arriba
            (-1, 1, g_diagonal),  # Arriba derecha
            (0, -1, g_lineal),  # Izquierda
            (0, 1, g_lineal),  # Derecha
            (1, -1, g_diagonal),  # Abajo izquierda
            (1, 0, g_lineal),  # Abajo
            (1, 1, g_diagonal)  # Abajo derecha
        ]

        for movimiento in movimientos:
            fila_neighbor = fila + movimiento[0]
            col_neighbor = col + movimiento[1]
            g = movimiento[2]

            if FILAS > fila_neighbor >= 0 and FILAS > col_neighbor >= 0:
                neighbor_nodo = grid[fila_neighbor][col_neighbor]
                if not neighbor_nodo.es_pared():
                    if not neighbor_nodo.es_visitado():
                        tentative_g_score = g_score[current] + g

                        if neighbor_nodo in g_score:
                            if tentative_g_score < g_score[neighbor_nodo]:
                                g_score[neighbor_nodo] = tentative_g_score
                                f_score[neighbor_nodo] = heuristica(neighbor_nodo, final) + g_score[neighbor_nodo]
                                neighbor_nodo.set_nodo_dependiente(current)
                                open_set.append(neighbor_nodo)
                                neighbor_nodo.set_text(
                                    f"H = {heuristica(neighbor_nodo, final)}-G = {g_score[neighbor_nodo]}-F = {f_score[neighbor_nodo]}")
                        else:
                            g_score[neighbor_nodo] = tentative_g_score
                            f_score[neighbor_nodo] = heuristica(neighbor_nodo, final) + g_score[neighbor_nodo]
                            neighbor_nodo.set_nodo_dependiente(current)
                            open_set.append(neighbor_nodo)
                            neighbor_nodo.set_text(
                                f"H = {heuristica(neighbor_nodo, final)}-G = {g_score[neighbor_nodo]}-F = {f_score[neighbor_nodo]}")

                        if not neighbor_nodo.es_inicio() and not neighbor_nodo == final and not neighbor_nodo.es_visitado():
                            neighbor_nodo.hacer_visita_neighbor()

                        neighbor_nodo.dibujar(ventana)
                        dibujar_grid(ventana, FILAS, ANCHO_VENTANA)
                        pygame.display.flip()
                        pygame.display.update()
                        # pygame.time.delay(300)

        last_current = current
        f_min = min(f_score.values())
        current = [clave for clave, value in f_score.items() if value == f_min][0]


def main(ventana, ancho):
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if inicio and fin:
                        a_estrella(inicio, fin, grid, ventana)

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()

                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()

                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

    pygame.quit()


main(VENTANA, ANCHO_VENTANA)