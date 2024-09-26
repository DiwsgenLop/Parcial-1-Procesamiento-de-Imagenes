import cv2
import numpy as np

def recorte(frame):
    x1, y1 = 280, 400
    x2, y2 = 1280, 720
    return frame[y1:y2, x1:x2]

def union(frame, frame_recortado):
    x1, y1 = 280, 400
    x2, y2 = 1280, 720
    frame_negro = np.zeros_like(frame)
    frame_negro[y1:y2, x1:x2] = frame_recortado
    return frame_negro

# Función para calcular el histograma manualmente
def calcular_histograma_manual(frame_channel):
    histograma = np.zeros(256, dtype=int)
    for pixel in frame_channel.flatten():
        histograma[pixel] += 1
    return histograma

# Función para calcular el histograma acumulativo
def calcular_histograma_acumulativo(histograma):
    return np.cumsum(histograma)

# Función para normalizar el histograma
def normalizar_histograma(histograma_acumulativo, num_pixels):
    return (histograma_acumulativo * 255) / num_pixels

# Función para aplicar la ecualización del histograma
def ecualizar_histograma(channel):
    hist = calcular_histograma_manual(channel)
    hist_acum = calcular_histograma_acumulativo(hist)
    hist_norm = normalizar_histograma(hist_acum, len(channel.flatten()))

    # Transformar los píxeles del canal usando el histograma normalizado
    channel_ecualizado = hist_norm[channel]
    return channel_ecualizado.astype(np.uint8)

# Función para ajustar el contraste usando ecualización en el canal V (HSV)
def ajustar_contraste(frame):
    # Convertimos a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Aplicamos la ecualización al canal V (luminancia)
    v_channel = hsv[:, :, 2]
    v_ecualizado = ecualizar_histograma(v_channel)
    
    # Reemplazamos el canal V por el ecualizado
    hsv[:, :, 2] = v_ecualizado
    
    # Convertimos de nuevo a BGR
    frame_contraste = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return frame_contraste

# Función para detectar líneas amarillas y blancas
def detectar_lineas(frame_procesado):
    hsv = cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2HSV)
    
    # Rango para líneas amarillas
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # Rango para líneas blancas
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([180, 30, 255])

    # Crear máscaras
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Unión de ambas máscaras
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)

    return combined_mask

# Función para enmascarar las líneas con fondo negro
def aplicar_fondo_negro(frame_procesado, mask_lineas):
    # Aplicamos la máscara a la imagen original
    resultado = cv2.bitwise_and(frame_procesado, frame_procesado, mask=mask_lineas)
    
    # Creamos un fondo negro
    fondo_negro = np.zeros_like(frame_procesado)
    
    # Añadimos las líneas sobre el fondo negro
    frame_con_lineas = cv2.add(fondo_negro, resultado)
    
    return frame_con_lineas

# Función principal para procesar cada frame
def procesar_frame(frame):
    # Ajustamos el contraste con ecualización del histograma
    frame_contraste = ajustar_contraste(frame)
    
    # Detectamos las líneas blancas y amarillas
    mask_lineas = detectar_lineas(frame_contraste)
    
    # Aplicamos fondo negro dejando solo las líneas visibles
    frame_con_lineas = aplicar_fondo_negro(frame_contraste, mask_lineas)
    
    return frame_con_lineas

# Función para cargar y procesar el video frame a frame
def cargar_video(ruta_video, procesar_frame):
    cap = cv2.VideoCapture(ruta_video)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recortamos la región de interés
        frame_recortado = recorte(frame)

        # Procesamos el frame (ajustamos contraste y detectamos líneas)
        frame_procesado = procesar_frame(frame_recortado)

        # Unimos el frame procesado con el original
        frame_unido = union(frame, frame_procesado)

        # Mostramos el frame procesado
        cv2.imshow('Frame procesado', frame_unido)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Función principal
if __name__ == "__main__":
    ruta_video = 'lineas.mp4'
    cargar_video(ruta_video, procesar_frame)
