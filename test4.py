import cv2
import numpy as np

# Cargar el video
cap = cv2.VideoCapture('lineas.mp4')

# Verificar que se pueda abrir el video
if not cap.isOpened():
    print("Error al abrir el video")

# Recorte del frame
def recorte(frame):
    x1, y1 = 280, 400  # Coordenadas de la esquina superior izquierda
    x2, y2 = 1280, 720  # Coordenadas de la esquina inferior derecha
    return frame[y1:y2, x1:x2]

# Conversión de cada fotograma de BGR a HSV
def convertir_a_HSV(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Calcular el histograma manualmente y ajustar el contraste
def ajustar_contraste(frame):
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cálculo manual del histograma
    histograma = np.zeros(256, dtype=int)
    for pixel in frame.flatten():
        histograma[pixel] += 1

    # Normalizar el histograma
    num_pixels = frame.size
    hist_norm = histograma / num_pixels

    # Calcular el histograma acumulativo
    hist_acumulativo = np.cumsum(hist_norm)

    # Crear una LUT (Look-Up Table) para mapear los valores de intensidad
    lut = np.uint8(255 * hist_acumulativo)

    # Aplicar la LUT a la imagen para ajustar el contraste
    frame_contraste = lut[frame]

    return frame_contraste


# Crear una máscara para los colores blanco y amarillo
def crear_mascara_color(hsv_frame):
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)

    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    return cv2.bitwise_or(mask_white, mask_yellow)

# Aplicar la corrección gamma manualmente
def correccion_gamma(frame):
    gamma = 1.2
    frame_gamma = np.array(255 * (frame / 255) ** gamma, dtype="uint8")
    return frame_gamma

# Ciclo principal de procesamiento de video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Recortar la región de interés
    frame_recortado = recorte(frame)

    # Convertir a HSV
    hsv_frame = convertir_a_HSV(frame_recortado)

    # Crear la máscara de las líneas blancas y amarillas
    mask = crear_mascara_color(hsv_frame)

    # Aplicar el ajuste de contraste
    frame_contraste = ajustar_contraste(frame_recortado)

    # Aplicar corrección gamma
    frame_final = correccion_gamma(frame_contraste)

    # Mostrar el resultado con las líneas resaltadas
    resultado = cv2.bitwise_or(frame_final, frame_final, maks=mask)
    cv2.imshow('Líneas resaltadas', resultado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
