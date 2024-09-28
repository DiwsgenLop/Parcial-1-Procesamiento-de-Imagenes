import cv2
import numpy as np
import matplotlib.pyplot as plt
# Cargar el video
cap = cv2.VideoCapture('lineas.mp4')

# Verificar que se pueda abrir el video
if not cap.isOpened():
    print("Error al abrir el video")

# Obtener el tamaño del video original (ancho, alto)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter para guardar el video procesado
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('salida_lineas.mp4', fourcc, 30, (frame_width, frame_height))

# Recorte del frame
def recorte(frame):
    x1, y1 = 280, 400
    x2, y2 = 1280, 720
    return frame[y1:y2, x1:x2]

# Conversión de cada fotograma de BGR a HSV
def convertir_a_HSV(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Crear una máscara para detectar las líneas blancas y amarillas
def crear_mascara_color(hsv_frame):
    # Blanco
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Amarillo
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Combinamos ambas máscaras
    return cv2.bitwise_or(mask_white, mask_yellow)

# Aplicar histograma acumulativo y normalizado al canal V en HSV
def ajustar_contraste_con_histograma(hsv_frame, mask):
    # Aplicar la máscara a las líneas detectadas en el canal V
    h_channel, s_channel, v_channel = cv2.split(hsv_frame)
    v_channel_lineas = cv2.bitwise_and(v_channel, v_channel, mask=mask)
    '''
    plt.imshow(v_channel_lineas, cmap='gray')
    plt.show()
    '''
    # Calcular el histograma solo para las áreas detectadas por la máscara
    histograma = np.zeros(256, dtype=int)
    for pixel in v_channel_lineas.flatten():
        histograma[pixel] += 1

    # Normalizar el histograma
    num_pixels = v_channel_lineas.size
    hist_norm = histograma / num_pixels

    # Calcular el histograma acumulativo
    hist_acumulativo = np.cumsum(hist_norm)
    '''
    plt.plot(hist_acumulativo)
    plt.show()
    '''
    # Crear la Look-Up Table (LUT) para redistribuir el brillo solo en las áreas detectadas
    lut = np.uint8(255 * hist_acumulativo)

    # Aplicar la LUT al canal V para ajustar el contraste
    v_contraste = lut[v_channel]
    '''
    plt.imshow(v_contraste, cmap='gray')
    plt.show()
    '''
    # Recombinar los canales H, S, y el nuevo canal V ajustado
    hsv_contraste = cv2.merge([h_channel, s_channel, v_contraste])

    return hsv_contraste

# Aplicar la corrección gamma solo a las líneas detectadas
def correccion_gamma(hsv_frame, mask, gamma=5):
    # Separar los canales HSV
    h_channel, s_channel, v_channel = cv2.split(hsv_frame)

    # Aplicar la máscara solo en el canal V
    v_channel_lineas = cv2.bitwise_and(v_channel, v_channel, mask=mask)

    # Aplicar corrección gamma
    v_gamma = np.array(255 * (v_channel_lineas / 255) ** gamma, dtype="uint8")

    # Recombinar los canales con la corrección gamma aplicada al canal V
    hsv_gamma = cv2.merge([h_channel, s_channel, v_gamma])

    return hsv_gamma

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

    # Aplicar el ajuste de contraste usando el histograma acumulativo en el canal V
    hsv_contraste = ajustar_contraste_con_histograma(hsv_frame, mask)

    # Aplicar la corrección gamma solo a las líneas detectadas
    hsv_final = correccion_gamma(hsv_contraste, mask)

    # Convertir de nuevo a BGR para guardar el resultado
    frame_final = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)

    # Mostrar el resultado con las líneas resaltadas y el fondo negro
    resultado = cv2.bitwise_and(frame_final, frame_final, mask=mask)
    cv2.imshow('Líneas resaltadas', resultado)

    # Escribir el frame procesado en el archivo de video de salida
    frame_salida = cv2.resize(resultado, (frame_width, frame_height))
    out.write(frame_salida)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()
