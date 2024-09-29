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

# Convertir el frame a HSV
def convertir_a_HSV(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Obtener el rango dinámico de un color
def obtener_rango_color(hsv_frame, lower_static, upper_static):
    # Filtrar los píxeles que estén dentro del rango estático (inicial)
    mask = np.all(hsv_frame >= lower_static, axis=2) & np.all(hsv_frame <= upper_static, axis=2)
    
    # Extraer los píxeles HSV correspondientes al color detectado
    color_pixels = hsv_frame[mask > 0]
    
    # Si no se detectan píxeles, devolver los valores originales
    if len(color_pixels) == 0:
        return lower_static, upper_static
    
    # Calcular los valores mínimo y máximo para H, S, y V en los píxeles detectados
    lower_dynamic = np.min(color_pixels, axis=0)
    upper_dynamic = np.max(color_pixels, axis=0)
    
    return lower_dynamic, upper_dynamic

# Crear una máscara manual para los colores blanco y amarillo usando NumPy
def crear_mascara_optimizada(hsv_frame, lower_color, upper_color):
    # Crear máscaras utilizando operaciones vectorizadas de NumPy
    mask = np.all(hsv_frame >= lower_color, axis=2) & np.all(hsv_frame <= upper_color, axis=2)
    # Convertir el resultado booleano a una máscara de 8 bits (0 o 255)
    return mask.astype(np.uint8) * 255

# Ajustar el contraste usando el histograma acumulativo
def ajustar_contraste_con_histograma(hsv_frame, mask):
    h_channel, s_channel, v_channel = cv2.split(hsv_frame)
    v_channel_lineas = cv2.bitwise_and(v_channel, v_channel, mask=mask)

    # Calcular el histograma solo en las áreas de las líneas
    histograma = np.zeros(256, dtype=int)
    for pixel in v_channel_lineas.flatten():
        histograma[pixel] += 1

    # Normalizar y calcular el histograma acumulativo
    hist_norm = histograma / v_channel_lineas.size
    hist_acumulativo = np.cumsum(hist_norm)
    lut = np.uint8(255 * hist_acumulativo)

    v_contraste = lut[v_channel]
    return cv2.merge([h_channel, s_channel, v_contraste])

# Aplicar la corrección gamma al canal V
def correccion_gamma(hsv_frame, mask, gamma=1.2):
    h_channel, s_channel, v_channel = cv2.split(hsv_frame)
    v_channel_lineas = cv2.bitwise_and(v_channel, v_channel, mask=mask)

    # Aplicar corrección gamma
    v_gamma = np.array(255 * (v_channel_lineas / 255) ** gamma, dtype="uint8")
    return cv2.merge([h_channel, s_channel, v_gamma])

# Ciclo principal de procesamiento de video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Recortar la región de interés
    frame_recortado = recorte(frame)

    # Convertir a HSV
    hsv_frame = convertir_a_HSV(frame_recortado)

    # Definir los rangos estáticos iniciales para blanco y amarillo
    lower_white_static = np.array([0, 0, 200])
    upper_white_static = np.array([180, 50, 255])
    lower_yellow_static = np.array([18, 94, 140])
    upper_yellow_static = np.array([48, 255, 255])

    # Obtener rangos dinámicos para el blanco y el amarillo en cada frame
    lower_white_dynamic, upper_white_dynamic = obtener_rango_color(hsv_frame, lower_white_static, upper_white_static)
    lower_yellow_dynamic, upper_yellow_dynamic = obtener_rango_color(hsv_frame, lower_yellow_static, upper_yellow_static)

    # Crear las máscaras basadas en los rangos dinámicos
    mask_white_dynamic = crear_mascara_optimizada(hsv_frame, lower_white_dynamic, upper_white_dynamic)
    mask_yellow_dynamic = crear_mascara_optimizada(hsv_frame, lower_yellow_dynamic, upper_yellow_dynamic)

    # Combinar las máscaras
    mask_dynamic = cv2.bitwise_or(mask_white_dynamic, mask_yellow_dynamic)

    # Ajustar el contraste usando el histograma acumulativo
    hsv_contraste = ajustar_contraste_con_histograma(hsv_frame, mask_dynamic)

    # Aplicar corrección gamma
    hsv_final = correccion_gamma(hsv_contraste, mask_dynamic)

    # Convertir de nuevo a BGR
    frame_final = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)

    # Aplicar la máscara y mostrar el resultado
    resultado = cv2.bitwise_and(frame_final, frame_final, mask=mask_dynamic)
    cv2.imshow('Líneas resaltadas', resultado)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
