import cv2
import numpy as np

# Cargar el video
cap = cv2.VideoCapture('lineas.mp4')

# Verificar que se pueda abrir el video
if not cap.isOpened():
    print("Error al abrir el video")

# Obtener el tamaño del video original (ancho, alto)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter para guardar el video procesado
# Prueba con un codec más compatible como 'XVID'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para el formato AVI
out = cv2.VideoWriter('salida_lineas.avi', fourcc, 30, (frame_width, frame_height))

# Recorte del frame
def recorte(frame):
    x1, y1 = 280, 400
    x2, y2 = 1280, 720
    return frame[y1:y2, x1:x2]

# Conversión de cada fotograma de BGR a HSV
def convertir_a_HSV(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Crear una máscara para los colores blanco y amarillo
def crear_mascara_color(hsv_frame):
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)

    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    return cv2.bitwise_or(mask_white, mask_yellow)

# Ajustar contraste basado en el histograma acumulativo del canal de brillo (V)
def ajustar_contraste_v(v_channel):
    histograma = np.zeros(256, dtype=int)
    for pixel in v_channel.flatten():
        histograma[pixel] += 1

    num_pixels = v_channel.size
    hist_norm = histograma / num_pixels

    hist_acumulativo = np.cumsum(hist_norm)

    lut = np.uint8(255 * hist_acumulativo)
    v_contraste = lut[v_channel]

    return v_contraste

# Aplicar la corrección gamma manualmente en el canal de brillo (V)
def correccion_gamma_v(v_channel):
    gamma = 5
    v_gamma = np.array(255 * (v_channel / 255) ** gamma, dtype="uint8")
    return v_gamma

# Ciclo principal de procesamiento de video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Recortar la región de interés
    frame_recortado = recorte(frame)

    # Convertir a HSV
    hsv_frame = convertir_a_HSV(frame_recortado)

    # Separar los canales HSV
    h_channel, s_channel, v_channel = cv2.split(hsv_frame)

    # Aplicar el ajuste de contraste al canal de brillo (V)
    v_contraste = ajustar_contraste_v(v_channel)

    # Aplicar corrección gamma al canal de brillo (V)
    v_final = correccion_gamma_v(v_contraste)
    
    # Recombinar los canales H, S, y el nuevo canal V ajustado
    hsv_final = cv2.merge([h_channel, s_channel, v_final])

    # Convertir de nuevo a BGR
    frame_final = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
    
    # Crear la máscara de las líneas blancas y amarillas
    mask = crear_mascara_color(hsv_frame)

    # Mostrar el resultado con las líneas resaltadas
    resultado = cv2.bitwise_and(frame_final, frame_final, mask=mask)
    cv2.imshow('Líneas resaltadas', resultado)

    # Escribir el frame procesado en el archivo de video de salida
    # Asegurarse de que el tamaño del frame sea igual al de salida (frame_width, frame_height)
    frame_salida = cv2.resize(resultado, (frame_width, frame_height))
    out.write(frame_salida)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()  # Cerrar el archivo de salida de video
cv2.destroyAllWindows()
