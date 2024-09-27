import cv2
import numpy as np

# Cargar el video
cap = cv2.VideoCapture('lineas.mp4')

# Verificar que se pueda abrir el video
if not cap.isOpened():
    print("Error al abrir el video")

# Recorte del frame
def recorte(frame):
    x1, y1 = 280, 400
    x2, y2 = 1280, 720
    return frame[y1:y2, x1:x2]

# Calcular el histograma del canal V y ajustar el rango de colores blanco y amarillo
def ajustar_rangos_color_con_histograma(hsv_frame):
    # Extraemos el canal V (brillo)
    h_channel, s_channel, v_channel = cv2.split(hsv_frame)

    # Cálculo manual del histograma del canal V
    hist_v = np.zeros(256, dtype=int)
    for pixel in v_channel.flatten():
        hist_v[pixel] += 1

    # Normalizamos el histograma
    num_pixels = v_channel.size
    hist_norm = hist_v / num_pixels

    # Determinamos los valores dinámicos de blanco y amarillo a partir del histograma
    umbral_blanco = np.argmax(hist_norm[200:]) + 200  # Detectar máximos en brillo alto (blanco)
    umbral_amarillo_min = np.argmax(hist_norm[140:180]) + 140  # Detectar un pico para amarillo
    umbral_amarillo_max = np.argmax(hist_norm[180:]) + 180

    # Definir rangos dinámicos
    lower_white_dynamic = np.array([0, 0, int(umbral_blanco)])
    upper_white_dynamic = np.array([180, 50, 255])

    lower_yellow_dynamic = np.array([18, 94, int(umbral_amarillo_min)])
    upper_yellow_dynamic = np.array([48, 255, int(umbral_amarillo_max)])
    
    # Valores estáticos de referencia
    lower_white_static = np.array([0, 0, 200])
    upper_white_static = np.array([180, 50, 255])

    lower_yellow_static = np.array([18, 94, 140])
    upper_yellow_static = np.array([48, 255, 255])

    # Combinar los valores dinámicos y estáticos
    lower_white = np.maximum(lower_white_dynamic, lower_white_static)  # Tomamos el valor más alto entre estático y dinámico
    upper_white = np.minimum(upper_white_dynamic, upper_white_static)  # Tomamos el valor más bajo entre estático y dinámico

    lower_yellow = np.maximum(lower_yellow_dynamic, lower_yellow_static)
    upper_yellow = np.minimum(upper_yellow_dynamic, upper_yellow_static)


    return lower_white, upper_white, lower_yellow, upper_yellow

# Crear una máscara para los colores blanco y amarillo dinámicamente
def crear_mascara_color(hsv_frame, lower_white, upper_white, lower_yellow, upper_yellow):
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    return cv2.bitwise_or(mask_white, mask_yellow)

# Aplicar la corrección gamma
def correccion_gamma(frame, gamma=1.2):
    return np.array(255 * (frame / 255) ** gamma, dtype="uint8")

# Ciclo principal de procesamiento de video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Recortar la región de interés
    frame_recortado = recorte(frame)

    # Convertir a HSV
    hsv_frame = cv2.cvtColor(frame_recortado, cv2.COLOR_BGR2HSV)

    # Ajustar los rangos de color blanco y amarillo basados en el histograma del canal V
    lower_white, upper_white, lower_yellow, upper_yellow = ajustar_rangos_color_con_histograma(hsv_frame)

    # Crear la máscara para las líneas blancas y amarillas
    mask = crear_mascara_color(hsv_frame, lower_white, upper_white, lower_yellow, upper_yellow)

    # Aplicar corrección gamma
    frame_gamma = correccion_gamma(frame_recortado, gamma=1.2)

    # Mostrar el resultado con las líneas resaltadas
    resultado = cv2.bitwise_and(frame_gamma, frame_gamma, mask=mask)
    cv2.imshow('Líneas resaltadas', resultado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
