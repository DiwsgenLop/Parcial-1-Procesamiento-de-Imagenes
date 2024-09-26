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

def filtros(frame):
    gamma = 1.2
    frame_gamma = np.array(255 * (frame / 255) ** gamma, dtype="uint8")
    return frame_gamma

def detectar_lineas(frame_procesado):
    hsv = cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([180, 30, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
    return combined_mask

def aplicar_fondo_negro(frame_procesado, mask_lineas):
    resultado = cv2.bitwise_and(frame_procesado, frame_procesado, mask=mask_lineas)
    fondo_negro = np.zeros_like(frame_procesado)
    frame_con_lineas = cv2.add(fondo_negro, resultado)
    return frame_con_lineas

def procesar_frame(frame):
    frame_procesado = filtros(frame)
    mask_lineas = detectar_lineas(frame_procesado)
    frame_con_lineas = aplicar_fondo_negro(frame_procesado, mask_lineas)
    return frame_con_lineas

def cargar_video(ruta_video, procesar_frame):
    cap = cv2.VideoCapture(ruta_video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_recortado = recorte(frame)
        frame_procesado = procesar_frame(frame_recortado)
        frame_unido = union(frame, frame_procesado)
        cv2.imshow('Frame procesado', frame_unido)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ruta_video = 'lineas.mp4'
    cargar_video(ruta_video, procesar_frame)
