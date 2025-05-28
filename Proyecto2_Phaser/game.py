import pygame
import random
import random
import csv
from datetime import datetime
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


directory_to_save_datasets = 'C:/Users/brayi/Desktop/NOVENO SEMESTRE/IA/RESPOSITORIO/Inteligencia-Artificial-2025/Proyecto2_Phaser/datasets'
directory_to_save_desition_tree = 'C:/Users/brayi/Desktop/NOVENO SEMESTRE/IA/RESPOSITORIO/Inteligencia-Artificial-2025/Proyecto2_Phaser/desition_tree'
decision_tree_trained_horizontal_ball = None
decision_tree_trained_vertical_ball = None
modo_decision_tree = False

# Variables para el modelo de red neuronal
directory_to_save_neural_network = 'C:/Users/brayi/Desktop/NOVENO SEMESTRE/IA/RESPOSITORIO/Inteligencia-Artificial-2025/Proyecto2_Phaser/neural_network'
neural_network_trained_horizontal_ball = None
mode_neural_network = False
prediction_counter_horizontal_ball = 0
prediction_counter_vertical_ball = 0

# Variables para el modelo KNN
knn_model = None
directory_to_save_knn = 'C:/Users/brayi/Desktop/NOVENO SEMESTRE/IA/RESPOSITORIO/Inteligencia-Artificial-2025/Proyecto2_Phaser/knn_model'

last_csv_path_saved_for_horizontal_ball = ''
last_csv_path_saved_for_vertical_ball = ''

# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# Variables del jugador, bala, nave, fondo, etc.
jugador = None
bala = None
fondo = None
nave = None
menu = None

# Variables de salto
salto = False
salto_altura = 15  # Velocidad inicial de salto
gravedad = 1
en_suelo = True

# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False  # Indica si el modo de juego es automático

# Lista para guardar los datos de velocidad, distancia y salto (target)
datos_modelo = []

# Cargar las imágenes
jugador_frames = [
    pygame.image.load('assets/sprites/mono_frame_1.png'),
    pygame.image.load('assets/sprites/mono_frame_2.png'),
    pygame.image.load('assets/sprites/mono_frame_3.png'),
    pygame.image.load('assets/sprites/mono_frame_4.png')
]

bala_img = pygame.image.load('assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('assets/game/fondo2.png')
nave_img = pygame.image.load('assets/game/ufo.png')
menu_img = pygame.image.load('assets/game/menu.png')

# Escalar la imagen de fondo para que coincida con el tamaño de la pantalla
fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear el rectángulo del jugador y de la bala
jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
nave2 = pygame.Rect(10, 0, 32, 48)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)  # Tamaño del menú

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# Variables para la bala
velocidad_bala = -10  # Velocidad de la bala hacia la izquierda
bala_disparada = False

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

############

def cargar_modelo_neural_network():
    global neural_network_trained_horizontal_ball, neural_network_trained_vertical_ball
    try:
        model_path_horizontal_ball = os.path.join(directory_to_save_neural_network, 'neural_network_model_horizontal_ball.keras')
        neural_network_trained_horizontal_ball = load_model(model_path_horizontal_ball)
        model_path_vertical_ball = os.path.join(directory_to_save_neural_network, 'neural_network_model_vertical_ball.keras')
        neural_network_trained_vertical_ball = load_model(model_path_vertical_ball)
        print("Modelo de red neuronal cargado exitosamente.")
    except:
        print("No se pudo cargar el modelo de red neuronal")


def predecir_salto_neural_network(velocidad_bala, desplazamiento_bala):
    if neural_network_trained_horizontal_ball is None:
        print("El modelo de red neuronal no está cargado.")
        return False

    # Preparar los datos de entrada
    input_data = np.array([[velocidad_bala, desplazamiento_bala]])

    # Realizar la predicción
    prediction = neural_network_trained_horizontal_ball.predict(input_data, verbose=0)
    # prediction = neural_network_trained.predict(input_data)

    # La predicción será un número entre 0 y 1
    # Podemos establecer un umbral, por ejemplo, 0.5
    return prediction[0][0] > 0.5

def predecir_retroceso_neural_network(velocidad_bala, desplazamiento_bala):
    if neural_network_trained_vertical_ball is None:
        print("El modelo de red neuronal no está cargado.")
        return False

    # Preparar los datos de entrada
    input_data = np.array([[velocidad_bala, desplazamiento_bala]])

    # Realizar la predicción
    prediction = neural_network_trained_vertical_ball.predict(input_data, verbose=0)
    # prediction = neural_network_trained.predict(input_data)

    # La predicción será un número entre 0 y 1
    # Podemos establecer un umbral, por ejemplo, 0.5
    return prediction[0][0] > 0.5

def predecir_retroceso_neural_network(velocidad_bala, desplazamiento_bala):
    if neural_network_trained_vertical_ball is None:
        print("El modelo de red neuronal no está cargado.")
        return False

    # Preparar los datos de entrada
    input_data = np.array([[velocidad_bala, desplazamiento_bala]])

    # Realizar la predicción
    prediction = neural_network_trained_vertical_ball.predict(input_data, verbose=0)
    # prediction = neural_network_trained.predict(input_data)

    # La predicción será un número entre 0 y 1
    # Podemos establecer un umbral, por ejemplo, 0.5
    return prediction[0][0] > 0.5

def generate_neural_network():
    global last_csv_path_saved_for_horizontal_ball, directory_to_save_neural_network, last_csv_path_saved_for_vertical_ball

    # Cargar el dataset
    df_horizontal_ball = pd.read_csv(os.path.join(last_csv_path_saved_for_horizontal_ball))
    df_vertical_ball = pd.read_csv(os.path.join(last_csv_path_saved_for_vertical_ball))

    # Separar características (X) y etiquetas (y)
    X_horizontal = df_horizontal_ball[['Velocidad Bala', 'Desplazamiento Bala']].values
    y_horizontal = df_horizontal_ball['Estatus Salto'].values
    X_vertical = df_vertical_ball[['Velocidad Bala', 'Desplazamiento Bala Y']].values
    y_vertical = df_vertical_ball['Estatus Retroceso'].values

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train_horizontal, X_test_horizontal, y_train_horizontal, y_test_horizontal = train_test_split(X_horizontal, y_horizontal, test_size=0.2, random_state=42)
    X_train_vertical, X_test_vertical, y_train_vertical, y_test_vertical = train_test_split(X_vertical, y_vertical, test_size=0.2, random_state=42)

    # Crear el modelo de red neuronal
    model_horizontal_ball = Sequential([
        Dense(8, input_dim=2, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_vertical_ball = Sequential([
        Dense(8, input_dim=2, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compilar el modelo
    model_horizontal_ball.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_vertical_ball.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model_horizontal_ball.fit(X_train_horizontal, y_train_horizontal, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    model_vertical_ball.fit(X_train_vertical, y_train_vertical, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluar el modelo
    loss_horizontal, accuracy_horizontal = model_horizontal_ball.evaluate(X_test_horizontal, y_test_horizontal, verbose=0)
    loss_vertical, accuracy_vertical = model_vertical_ball.evaluate(X_test_vertical, y_test_vertical, verbose=0)
    print(f"\nPrecisión en el conjunto de prueba bala 1: {accuracy_horizontal:.2f}")
    print(f"Precisión en el conjunto de prueba bala 2: {accuracy_vertical:.2f}")

    # Guardar el modelo
    save_model(model_horizontal_ball, os.path.join(directory_to_save_neural_network, 'neural_network_model_horizontal_ball.keras'))
    save_model(model_vertical_ball, os.path.join(directory_to_save_neural_network, 'neural_network_model_vertical_ball.keras'))

    print("Modelo de red neuronal generado y guardado exitosamente.")


### ---------------- DESICITION TREE --------------- ###

def cargar_modelo_decision_tree():
    global decision_tree_trained_horizontal_ball, decision_tree_trained_vertical_ball
    print(directory_to_save_desition_tree)
    try:
        decision_tree_trained_horizontal_ball = joblib.load(directory_to_save_desition_tree + '/decision_tree_model_horizontal_ball.joblib')
        decision_tree_trained_vertical_ball = joblib.load(directory_to_save_desition_tree + '/decision_tree_model_vertical_ball.joblib')
        print("Desition tree cargado exitosamente.")
    except:
        print("No se pudo cargar el modelo de árbol de decisión")


def predecir_salto_desition_tree(velocidad_bala, desplazamiento_bala):
    global decision_tree_trained_horizontal_ball
    if decision_tree_trained_horizontal_ball is not None:
        prediccion = decision_tree_trained_horizontal_ball.predict([[velocidad_bala, desplazamiento_bala]])
        #print("Predicción de salto: " + str(prediccion[0]))
        if prediccion[0] == '1':
            return True
    return False

def predecir_retroceso_desition_tree(velocidad_bala, desplazamiento_bala):
    global decision_tree_trained_vertical_ball
    if decision_tree_trained_vertical_ball is not None:
        prediccion = decision_tree_trained_vertical_ball.predict([[velocidad_bala, desplazamiento_bala]])
        #print("Predicción de salto: " + str(prediccion[0]))
        if prediccion[0] == '1':
            return True
    return False

def generate_desition_treee():
    global last_csv_path_saved_for_horizontal_ball, directory_to_save_desition_tree, last_csv_path_saved_for_vertical_ball

    if last_csv_path_saved_for_horizontal_ball == '' or last_csv_path_saved_for_vertical_ball == '':
        print('Primero debe de guardar el data set')
        return

    # Asegurarse de que el directorio existe
    os.makedirs(directory_to_save_desition_tree, exist_ok=True)

    # Leer el CSV sin encabezados
    dataset_horizontal_ball = pd.read_csv(last_csv_path_saved_for_horizontal_ball, header=None)
    dataset_vertical_ball = pd.read_csv(last_csv_path_saved_for_vertical_ball, header=None)

    # Eliminar la primera fila que contiene encabezados incorrectos
    dataset_cleaned_horizontal_ball = dataset_horizontal_ball.iloc[1:].reset_index(drop=True)
    dataset_cleaned_horizontal_ball = dataset_cleaned_horizontal_ball.dropna()
    dataset_cleaned_vertical_ball = dataset_vertical_ball.iloc[1:].reset_index(drop=True)
    dataset_cleaned_vertical_ball = dataset_cleaned_vertical_ball.dropna()

    # Guardar el CSV limpio sin índice
    cleaned_csv_path_horizontal_ball = os.path.join(directory_to_save_desition_tree, 'dataset_cleaned_horizontal_ball.csv')
    dataset_cleaned_horizontal_ball.to_csv(cleaned_csv_path_horizontal_ball, index=False, header=False)
    print(f"CSV limpio guardado en: {cleaned_csv_path_horizontal_ball}")
    cleaned_csv_path_vertical_ball = os.path.join(directory_to_save_desition_tree, 'dataset_cleaned_vertical_ball.csv')
    dataset_cleaned_vertical_ball.to_csv(cleaned_csv_path_vertical_ball, index=False, header=False)
    #print(f"CSV limpio guardado en: {cleaned_csv_path_vertical_ball}")

    # Definir características (X) y etiquetas (y)
    X_horizontal = dataset_cleaned_horizontal_ball.iloc[:, :2]  # Las dos primeras columnas son las características
    y_horizontal = dataset_cleaned_horizontal_ball.iloc[:, 2]  # La tercera columna es la etiqueta
    X_vertical = dataset_cleaned_vertical_ball.iloc[:, :2]  # Las dos primeras columnas son las características
    y_vertical = dataset_cleaned_vertical_ball.iloc[:, 2]  # La tercera columna es la etiqueta

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train_horizontal, X_test_horizontal, y_train_horizontal, y_test_horizontal = train_test_split(X_horizontal, y_horizontal, test_size=0.2, random_state=42)
    X_train_vertical, X_test_vertical, y_train_vertical, y_test_vertical = train_test_split(X_vertical, y_vertical, test_size=0.2, random_state=42)

    # Crear el clasificador de Árbol de Decisión
    clf_horizontal = DecisionTreeClassifier()
    clf_vertical = DecisionTreeClassifier()

    # Entrenar el modelo
    clf_horizontal.fit(X_train_horizontal, y_train_horizontal)
    clf_vertical.fit(X_train_vertical, y_train_vertical)

    # Guardar el arbol de decisión en un archivo PDF
    pdf_path_horizontal_ball = os.path.join(directory_to_save_desition_tree, 'decision_tree_horizontal_ball.pdf')
    pdf_path_vertical_ball = os.path.join(directory_to_save_desition_tree, 'decision_tree_vertical_ball.pdf')
    with PdfPages(pdf_path_horizontal_ball) as pdf:
        plt.figure(figsize=(12, 8))
        plot_tree(clf_horizontal, feature_names=['V. Bala', 'D. Bala'], class_names=['C. 0 (Suelo)', 'C. 1 (Salto)'],
                  filled=True)
        plt.title("Árbol de Decisión - Horizontal Ball")
        pdf.savefig()  # Guarda el gráfico en el PDF
        plt.close()
    with PdfPages(pdf_path_vertical_ball) as pdf:
        plt.figure(figsize=(12, 8))
        plot_tree(clf_vertical, feature_names=['V. Bala', 'D. Bala'], class_names=['C. 0 (Suelo)', 'C. 1 (Salto)'],
                  filled=True)
        plt.title("Árbol de Decisión - Vertical Ball")
        pdf.savefig()

    # Guardar el modelo entrenado COMO JOBLIB en el directorio especificado
    model_path_horizontal_ball = os.path.join(directory_to_save_desition_tree, 'decision_tree_model_horizontal_ball.joblib')
    joblib.dump(clf_horizontal, model_path_horizontal_ball)
    model_path_vertical_ball = os.path.join(directory_to_save_desition_tree, 'decision_tree_model_vertical_ball.joblib')
    joblib.dump(clf_vertical, model_path_vertical_ball)
    print(f"Modelo de árbol de decisión guardado en: {model_path_horizontal_ball}")

###########


# Función para disparar la bala
def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-8, -3)  # Velocidad aleatoria negativa para la bala
        bala_disparada = True

# Función para reiniciar la posición de la bala
def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50  # Reiniciar la posición de la bala
    bala_disparada = False

# Función para disparar la segunda bala
def disparar_bala2():
    global bala2_disparada, bala2, velocidad_bala2
    if not bala2_disparada:
        bala2.x = w - 950
        bala2.y = 0
        #velocidad_bala2 = random.randint(7, 12)  # Velocidad aleatoria hacia abajo
        velocidad_bala2 = 3  # Velocidad constante hacia abajo
        bala2_disparada = True

# Función para reiniciar la posición de la segunda bala
def reset_bala2():
    global bala2, bala2_disparada
    bala2.x = w - 950
    bala2.y = 0
    bala2_disparada = False


# Función para manejar el salto
def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura  # Mover al jugador hacia arriba
        salto_altura -= gravedad  # Aplicar gravedad (reduce la velocidad del salto)

        # Si el jugador llega al suelo, detener el salto
        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15  # Restablecer la velocidad de salto
            en_suelo = True

# Función para manejar el retroceso
def manejar_retroceso():
    global jugador, retroceso, retroceso_distancia, regreso, en_pocision_inicial

    if retroceso:
        jugador.x += retroceso_distancia  # Mover al jugador hacia atras
        retroceso_distancia -= regreso  # Aplicar gravedad (reduce la velocidad del retroceso)

        # Si el jugador llega a la posición inicial, detener el retroceso
        if jugador.x <= 0:
            jugador.x = 50
            retroceso = False
            retroceso_distancia = 10
            en_pocision_inicial = True


# Función para actualizar el juego
def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2
    global modo_decision_tree, salto

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    # Si el primer fondo sale de la pantalla, lo movemos detrás del segundo
    if fondo_x1 <= -w:
        fondo_x1 = w

    # Si el segundo fondo sale de la pantalla, lo movemos detrás del primero
    if fondo_x2 <= -w:
        fondo_x2 = w

    # Dibujar los fondos
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    # Dibujar el jugador con la animación
    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))

    # Dibujar la nave
    pantalla.blit(nave_img, (nave.x, nave.y))
    pantalla.blit(nave_img, (nave2.x, nave2.y))

    # Mover y dibujar la bala
    if bala_disparada:
        bala.x += velocidad_bala

    # Si la bala sale de la pantalla, reiniciar su posición
    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    # Colisión entre la bala y el jugador
    if jugador.colliderect(bala):
        print("Colisión detectada!")
        reiniciar_juego()  # Terminar el juego y mostrar el menú
   
    # Mover y dibujar la segunda bala si está en modo 2 balas
    if modo_2_balas:
        if bala2_disparada:
            bala2.y += velocidad_bala2
        else:
            disparar_bala2()

        # Si la bala2 sale de la pantalla, reiniciar su posición
        if bala2.y > h:
            reset_bala2()

        pantalla.blit(bala_img, (bala2.x, bala2.y))

        # Colisión entre la bala2 y el jugador
        if jugador.colliderect(bala2):
            print("Colisión con bala 2 detectada!")
            reiniciar_juego()


# Función para guardar datos del modelo en modo manual
def guardar_datos():
    global jugador, bala, velocidad_bala, salto, bala2, velocidad_bala2
    global modo_manual, modo_2_balas
    global retroceso

    if modo_manual and not modo_2_balas:
        distancia = abs(jugador.x - bala.x)
        salto_hecho = 1 if salto else 0  # 1 si saltó, 0 si no saltó
        # Guardar velocidad de la bala, distancia al jugador y si saltó o no
        datos_modelo.append((velocidad_bala, distancia, salto_hecho))

    if modo_2_balas:
        distancia = abs(jugador.x - bala.x)
        salto_hecho = 1 if salto else 0  # 1 si saltó, 0 si no saltó
        retroceso_hecho = 1 if retroceso else 0  # 1 si retrocedió, 0 si no retrocedió
        # Guardar velocidad de la bala, distancia al jugador y si saltó o no
        datos_modelo.append((velocidad_bala, distancia, salto_hecho))
        distanciaY = jugador.y - bala2.y
        datos_modelo_vertical_ball.append((velocidad_bala2, distanciaY, retroceso_hecho))

# Función para pausar el juego y guardar los datos
def pausa_juego():
    global pausa, menu_activo
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos registrados hasta ahora:", datos_modelo)
        menu_activo = True
        mostrar_menu()
    else:
        print("Juego reanudado.")

# Función para guardar el dataset en un archivo CSV
def save_data_set():
    global last_csv_path_saved_for_horizontal_ball, last_csv_path_saved_for_vertical_ball
    global datos_modelo, datos_modelo_vertical_ball

    if modo_manual and not modo_2_balas:
        # Generar un nombre de archivo único con la fecha y hora actual
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_horizontal_ball = f"dataset_horizontal_ball_{timestamp}.csv"

        # Crear la ruta completa del archivo
        file_path_horizontal_ball = os.path.join(directory_to_save_datasets, filename_horizontal_ball)

        try:
            # Asegurarse de que el directorio existe
            os.makedirs(directory_to_save_datasets, exist_ok=True)

            with open(file_path_horizontal_ball, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Escribir el encabezado
                writer.writerow(["Velocidad Bala", "Desplazamiento Bala", "Estatus Salto"])

                # Escribir los datos

                for dato in datos_modelo:
                    writer.writerow(dato)

            last_csv_path_saved_for_horizontal_ball = file_path_horizontal_ball
            print(f"Dataset guardado exitosamente como '{last_csv_path_saved_for_horizontal_ball}'")
        except Exception as e:
            print(f"Error al guardar el dataset: {e}")

    if modo_2_balas:
        # Generar un nombre de archivo único con la fecha y hora actual
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_horizontal_ball = f"dataset_horizontal_ball_{timestamp}.csv"
        filename_vertical_ball = f"dataset_vertical_ball_{timestamp}.csv"

        # Crear la ruta completa del archivo
        file_path_horizontal_ball = os.path.join(directory_to_save_datasets, filename_horizontal_ball)
        file_path_vertical_ball = os.path.join(directory_to_save_datasets, filename_vertical_ball)

        try:
            # Asegurarse de que el directorio existe
            os.makedirs(directory_to_save_datasets, exist_ok=True)

            with open(file_path_horizontal_ball, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Escribir el encabezado
                writer.writerow(["Velocidad Bala", "Desplazamiento Bala", "Estatus Salto"])

                # Escribir los datos
                for dato in datos_modelo:
                    writer.writerow(dato)

            with open(file_path_vertical_ball, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Escribir el encabezado
                writer.writerow(["Velocidad Bala", "Desplazamiento Bala Y", "Estatus Retroceso"])

                # Escribir los datos
                for dato in datos_modelo_vertical_ball:
                    writer.writerow(dato)

            last_csv_path_saved_for_horizontal_ball = file_path_horizontal_ball
            last_csv_path_saved_for_vertical_ball = file_path_vertical_ball
            print(f"Dataset guardado exitosamente como '{last_csv_path_saved_for_horizontal_ball}'")
            print(f"Dataset guardado exitosamente como '{last_csv_path_saved_for_vertical_ball}'")
        except Exception as e:
            print(f"Error al guardar el dataset: {e}")

# Función para mostrar el menú de opciones
def print_menu_options():
    lineas = [
        "============ MENU =============",
        "",
        "Press M - Manual Mode",
        "Press N - Auto Mode Neural Network",
        "Press D - Auto Mode Decision Tree",
        "Press K - Auto Mode KNN",
        "Press S - Save DataSet",
        "Press T - Training Models",
        "",
        "Press Q - Exit",
    ]

    # Posición inicial
    x = w // 4
    y = h // 2 - (len(lineas) * 20)  # Ajusta el desplazamiento vertical según el número de líneas

    for linea in lineas:
        texto = fuente.render(linea, True, NEGRO)
        pantalla.blit(texto, (x, y))
        y += 40
    pygame.display.flip()

# Función para entrenar los modelos
def train_models():
    #generate_neural_network()
    generate_desition_treee()
    #generate_knn_model()

# Función para mostrar el menú y seleccionar el modo de juego
def mostrar_menu():
    global pausa, menu_activo, modo_auto, modo_manual, modo_2_balas
    global modo_decision_tree, modo_manual, modo_auto, mode_neural_network
    global datos_modelo, datos_modelo_vertical_ball

    pantalla.fill(BLANCO)
    print_menu_options()
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_m:
                    datos_modelo = []
                    modo_auto = False
                    modo_manual = True
                    modo_auto = False
                    modo_decision_tree = False
                    modo_2_balas = True
                    menu_activo = False
                    pausa = False
                elif evento.key == pygame.K_2:
                    datos_modelo = []
                    modo_auto = False
                    modo_manual = False
                    modo_2_balas = True
                    menu_activo = False
                elif evento.key == pygame.K_s:
                    save_data_set()
                    menu_activo = True
                elif evento.key == pygame.K_t:
                    train_models()
                    menu_activo = True
                elif evento.key == pygame.K_n:
                    modo_auto = True
                    modo_decision_tree = False
                    mode_neural_network = True
                    modo_manual = False
                    modo_2_balas = True
                    menu_activo = False
                    pausa = False
                    cargar_modelo_neural_network()
                elif evento.key == pygame.K_d:
                    modo_auto = True
                    modo_decision_tree = True
                    mode_neural_network = False
                    modo_manual = False
                    modo_2_balas = True
                    menu_activo = False
                    pausa = False
                    cargar_modelo_decision_tree()
                elif evento.key == pygame.K_k:
                    modo_auto = True
                    modo_decision_tree = False
                    mode_neural_network = False
                    modo_manual = False
                    modo_2_balas = True
                    menu_activo = False
                    pausa = False
                    cargar_modelo_knn()
                elif evento.key == pygame.K_q:
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

    pygame.display.flip()


# Función para reiniciar el juego tras la colisión
def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo, bala2_disparada, salto_altura
    global datos_modelo, datos_modelo_vertical_ball
    global retroceso, retroceso_distancia, regreso, en_pocision_inicial

    menu_activo = True  # Activar de nuevo el menú
    jugador.x, jugador.y = 50, h - 100  # Reiniciar posición del jugador
    bala.x = w - 50  # Reiniciar posición de la bala
    nave.x, nave.y = w - 100, h - 100  # Reiniciar posición de la nave
    bala_disparada = False
    salto = False
    retroceso = False
    salto_altura = 15  # Restablecer la velocidad de salto
    retroceso_distancia = 10
    en_suelo = True
    en_pocision_inicial = True
    # Reiniciar la segunda bala
    bala2.x = random.randint(0, w - 16)
    bala2.y = 0
    bala2_disparada = False
    # Mostrar los datos recopilados hasta el momento
    print("Datos recopilados para el modelo: ", datos_modelo)

    # datos_modelo = []

    mostrar_menu()  # Mostrar el menú de nuevo para seleccionar modo


def run_any_mode(correr):
    global salto, en_suelo, bala_disparada, bala2_disparada
    global modo_decision_tree, modo_manual, modo_auto, modo_2_balas
    global bala, velocidad_bala, jugador, prediction_counter_horizontal_ball, prediction_counter_vertical_ball, velocidad_bala2, bala2
    global retroceso, regreso, en_pocision_inicial

    pygame.display.flip()
    reloj = pygame.time.Clock()
    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa or evento.key == pygame.K_UP:  # Detectar la tecla espacio para saltar
                    # print('saltando.....')
                    salto = True
                    en_suelo = False
                    salto_altura = 15  # Restablecer la velocidad de salto al iniciar un nuevo salto
                if evento.key == pygame.K_RIGHT and en_pocision_inicial and not pausa:  # Detectar la flecha izquierda para retroceder
                    retroceso = True
                    en_pocision_inicial = False
                    retroceso_distancia = 10  # Restablecer la velocidad de retroceso al iniciar un nuevo retroceso
                if evento.key == pygame.K_p:  # Presiona 'p' para pausar el juego
                    pausa_juego()
                if evento.key == pygame.K_q:  # Presiona 'q' para terminar el juego
                    print("Juego terminado.")
                    pygame.quit()
                    exit()

        if not pausa:
            # Modo manual: el jugador controla el salto
            if not modo_auto:
                if salto:
                    manejar_salto()
                if retroceso:
                    manejar_retroceso()
                # Guardar los datos si estamos en modo manual
                guardar_datos()

            # Modo automático: neural network
            elif mode_neural_network:
                # This module oprand has used to minimize the calls to the neural network for predictions
                prediction_counter_horizontal_ball += 1
                prediction_counter_vertical_ball += 1
                if prediction_counter_horizontal_ball % 1 == 0 and prediction_counter_vertical_ball % 1 == 0:
                    if mode_neural_network and neural_network_trained_horizontal_ball is not None:
                        desplazamiento_bala = bala.x - jugador.x
                        if predecir_salto_neural_network(velocidad_bala, desplazamiento_bala) and en_suelo:
                            salto = True
                            en_suelo = False
                        if predecir_retroceso_neural_network(velocidad_bala2, bala2.y - jugador.y) and en_pocision_inicial:
                            retroceso = True
                            en_pocision_inicial = False
                    if salto:
                        manejar_salto()
                    if retroceso:
                        manejar_retroceso()

            # Modo automático: árbol de decisión
            elif modo_decision_tree:
                if modo_decision_tree and decision_tree_trained_horizontal_ball is not None and decision_tree_trained_vertical_ball is not None:
                    desplazamiento_bala = bala.x - jugador.x
                    desplazamiento_bala_y = bala2.y - jugador.y
                    if predecir_salto_desition_tree(velocidad_bala, desplazamiento_bala) and en_suelo:
                        print('saltando... prediction true...')
                        salto = True
                        en_suelo = False
                    if predecir_retroceso_desition_tree(velocidad_bala2, desplazamiento_bala_y) and en_pocision_inicial:
                        print('retrocediendo... prediction true...')
                        retroceso = True
                        en_pocision_inicial = False
                if salto:
                    manejar_salto()
                if retroceso:
                    manejar_retroceso()

            # Modo automático: KNN
            elif modo_auto and knn_model_horizontal_ball is not None and knn_model_vertical_ball is not None:
                desplazamiento_bala = bala.x - jugador.x
                desplazamiento_bala_y = bala2.y - jugador.y
                if predecir_salto_knn(velocidad_bala, desplazamiento_bala) and en_suelo:
                    print('saltando... prediction true...')
                    salto = True
                    en_suelo = False
                if predecir_retroceso_knn(velocidad_bala2, desplazamiento_bala_y) and en_pocision_inicial:
                    print('retrocediendo... prediction true...')
                    retroceso = True
                    en_pocision_inicial = False
                if salto:
                    manejar_salto()
                if retroceso:
                    manejar_retroceso()

            # Movimiento manual del jugador
            # keys = pygame.key.get_pressed()
            # if keys[pygame.K_LEFT]:
                # jugador.x -= 5
            # if keys[pygame.K_RIGHT]:
                # jugador.x += 5

            # Mantener al jugador dentro de los límites de la pantalla
            if jugador.x < 0:
                jugador.x = 0
            if jugador.x > w - jugador.width:
                jugador.x = w - jugador.width
            if jugador.y < 0:
                jugador.y = 0
            if jugador.y > h - jugador.height:
                jugador.y = h - jugador.height

            # Actualizar el juego
            if not bala_disparada:
                disparar_bala()

            update()

        # Actualizar la pantalla
        pygame.display.flip()
        reloj.tick(60)  # Limitar el juego a 60 FPS


def main():
    global salto, en_suelo, bala_disparada
    global modo_decision_tree, modo_manual, modo_auto
    global bala, velocidad_bala, jugador, prediction_counter
    global retroceso, en_pocision_inicial

    mostrar_menu()  # Mostrar el menú al inicio
    correr = True
    run_any_mode(correr)

    pygame.quit()

if __name__ == "__main__":
    main()