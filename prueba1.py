import cv2
import numpy as np

# Cargar el video
cap = cv2.VideoCapture('lineas.mp4')

# Verificar que se pueda abrir el video
if not cap.isOpened():
    print("Error al abrir el video")
    
# Conversión de cada fotograma de BGR a HSV
def convertir_a_HSV(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def crear_mascara_color(hsv_frame):
    # Rango para el blanco en HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Rango para el amarillo en HSV
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Combinamos ambas máscaras
    return cv2.bitwise_or(mask_white, mask_yellow)

def ajustar_contraste(frame):
    # Convertimos a escala de grises para simplificar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculamos el histograma
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Normalizamos el histograma
    hist_norm = hist / hist.sum()
    
    # Aplicamos una función de ajuste de contraste básico
    alpha = 1.5  # Factor de contraste
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha)
    return adjusted

def operador_negativo(frame):
    return 255 - frame

def operador_brillo(frame, beta=50):
    # Incrementa el brillo
    return cv2.convertScaleAbs(frame, beta=beta)

def correccion_gamma(frame, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

#Cuando nuestra aplicación se cierra, liberamos los recursos
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertimos a HSV
    hsv_frame = convertir_a_HSV(frame)

    # Creamos la máscara para líneas blancas y amarillas
    mask = crear_mascara_color(hsv_frame)

    # Aplicamos las transformaciones necesarias
    frame_contraste = ajustar_contraste(frame)
    frame_brillo = operador_brillo(frame_contraste, beta=20)
    frame_final = correccion_gamma(frame_brillo, gamma=0.5)

    # Mostramos el resultado
    cv2.imshow('Líneas resaltadas', cv2.bitwise_and(frame, frame, mask=mask))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos los recursos
cap.release()
cv2.destroyAllWindows()
