import cv2
import numpy as np
import matplotlib.pyplot as plt
# Cargar el video
cap = cv2.VideoCapture('lineas.mp4')

# Verificar que se pueda abrir el video
if not cap.isOpened():
    print("Error al abrir el video")

#Ahora obtenemos las dimensiones originales del video antes de recortar
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#Definimos el codec y creamos el objeto VideoWriter para guardar el video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('salida_procesada.mp4', fourcc, 30, (frame_width, frame_height))

# recortar del frame
def recortar(frame):
    x1, y1 = 280, 400
    x2, y2 = 1280, 720
    return frame[y1:y2, x1:x2]

# Convertir el frame a HSV
def convertir_a_HSV(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Obtener el rango dinámico de un color
def rango_dinamico(hsv_frame, lower_static, upper_static):
    # Filtrar los píxeles que estén dentro del rango estático (inicial)
    mask = np.all(hsv_frame >= lower_static, axis=2) & np.all(hsv_frame <= upper_static, axis=2)
    
    #Vemos la mascara dentro de los rangos estaticos
    plt.title("Frame con mascara con valores estaticos")
    plt.imshow(mask)
    plt.show()
    
    # Extraer los píxeles HSV correspondientes al color detectado
    color_pixels = hsv_frame[mask > 0]
    print("Color de pixel respecto a la mascara",color_pixels)
    
    # Si no se detectan píxeles, devolver los valores originales
    if len(color_pixels) == 0:
        return lower_static, upper_static
    
    # Calcular los valores mínimo y máximo para H, S, y V en los píxeles detectados
    lower_dynamic = np.min(color_pixels, axis=0)
    upper_dynamic = np.max(color_pixels, axis=0)
    
    return lower_dynamic, upper_dynamic

# Crear una máscara manual para los colores blanco y amarillo usando NumPy
def mascara_(hsv_frame, lower_color, upper_color):
    # Crear máscaras utilizando operaciones vectorizadas de NumPy
    mask = np.all(hsv_frame >= lower_color, axis=2) & np.all(hsv_frame <= upper_color, axis=2)
    # Convertir el resultado booleano a una máscara de 8 bits (0 o 255)
    return mask.astype(np.uint8) * 255

# Ajustar el contraste usando el histograma acumulativo
def ajuste_contraste(hsv_frame, mask):
    h_channel, s_channel, v_channel = cv2.split(hsv_frame)
    v_channel_lineas = cv2.bitwise_and(v_channel, v_channel, mask=mask)
    
    plt.title("Frame con mascara")
    plt.imshow(v_channel_lineas)
    plt.show()
    
    # Calcular el histograma solo en las áreas de las líneas
    histograma = np.zeros(256, dtype=int)
    for pixel in v_channel_lineas.flatten():
        histograma[pixel] += 1
    '''
    Version beta
    '''
    #Vemos el histograma de los pixeles de la mascara antes de normalizarlo
    plt.title("Histograma de los pixeles de la mascara")
    plt.plot(histograma)
    plt.show()
    '''
    Version beta
    '''
    # Normalizamos el histograma
    hist_norm = histograma / v_channel_lineas.size
    
    '''
    Version beta
    '''
    plt.title("Histograma normalizado")
    plt.plot(hist_norm)
    plt.show()
    '''
    Version beta
    '''
    # Calculamos el histograma acumulativo
    hist_acumulativo = np.cumsum(hist_norm)
    '''
    Version beta
    '''
    plt.title("Histograma acumulativo")
    plt.plot(hist_acumulativo)
    plt.show()
    '''
    Version beta
    '''
    # Creamos la tabla de búsqueda (LUT) para ajustar el contraste
    lut = np.uint8(255 * hist_acumulativo)
    # Aplicamos la LUT al canal V
    v_contraste = lut[v_channel]
    
    # Devolvemos la imagen con el contraste ajustado combinando los canales H y S con el canal V modificado
    return cv2.merge([h_channel, s_channel, v_contraste])

# Aplicar la corrección gamma al canal V
def correccion_gamma(hsv_frame, mask, gamma=5):
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
    '''
    Version beta
    '''
    print("Dimensiones del frame original",frame.shape)
    '''
    Version beta
    '''
    tempRGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.title("Frame original")
    plt.imshow(tempRGB, cmap='gray')
    plt.show()
    '''
    Version beta
    '''
    
    # Recortar la región de interés
    frame_recortado = recortar(frame)
    '''
    Version beta
    '''
    print("Valor de frame recortado",frame_recortado.shape)
    #Vemos el frame recortado y antes de los colores hsv
    tempRGB=cv2.cvtColor(frame_recortado, cv2.COLOR_BGR2RGB)
    plt.title("Frame recortado")
    plt.imshow(tempRGB, cmap='gray')
    plt.show()
    '''
    Version beta
    '''
    # Convertir a HSV
    hsv_frame = convertir_a_HSV(frame_recortado)
    '''
    Version beta
    '''
    #Ahora vemos el frame recortado en hsv
    plt.title("Frame recortado en hsv")
    plt.imshow(hsv_frame)
    plt.show()
    '''
    Version beta
    '''
    
    # Definir los rangos estáticos iniciales para blanco y amarillo
    lower_white_static = np.array([0, 0, 200])
    upper_white_static = np.array([180, 50, 255])
    lower_yellow_static = np.array([18, 94, 140])
    upper_yellow_static = np.array([48, 255, 255])

    # Obtener rangos dinámicos para el blanco y el amarillo en cada frame
    lower_white_dynamic, upper_white_dynamic = rango_dinamico(hsv_frame, lower_white_static, upper_white_static)
    lower_yellow_dynamic, upper_yellow_dynamic = rango_dinamico(hsv_frame, lower_yellow_static, upper_yellow_static)
    
    '''
    Version beta
    '''
    # Mostrar los valores dinámicos por frame
    print(f"Blanco dinámico - Mínimo: {lower_white_dynamic}, Máximo: {upper_white_dynamic}")
    print(f"Amarillo dinámico - Mínimo: {lower_yellow_dynamic}, Máximo: {upper_yellow_dynamic}")
    '''
    Version beta
    '''
    
    # Crear las máscaras basadas en los rangos dinámicos
    mask_white_dynamic = mascara_(hsv_frame, lower_white_dynamic, upper_white_dynamic)
    mask_yellow_dynamic = mascara_(hsv_frame, lower_yellow_dynamic, upper_yellow_dynamic)
    
    '''
    Version beta
    '''
    plt.title("Mascara blanca")
    plt.imshow(mask_white_dynamic)
    plt.show()
        
    plt.title("Mascara amarilla")
    plt.imshow(mask_yellow_dynamic)
    plt.show()
    
    '''
    Version beta
    '''

    # Combinar las máscaras
    mask_dynamic = cv2.bitwise_or(mask_white_dynamic, mask_yellow_dynamic)
    '''
    Version beta
    '''
    # Mostrar la máscara combinada
    plt.title("Mascara combinada")
    plt.imshow(mask_dynamic)
    plt.show()
    '''
    Version beta
    '''
    # Ajustar el contraste usando el histograma acumulativo
    hsv_contraste = ajuste_contraste(hsv_frame, mask_dynamic)
    
    '''
    Version beta
    '''
    #Vemos el frame con el contraste ajustado
    plt.title("Frame con contraste ajustado")
    plt.imshow(hsv_contraste)
    plt.show()
    '''
    Version beta
    '''
    
    # Aplicar corrección gamma
    hsv_final = correccion_gamma(hsv_contraste, mask_dynamic)
    '''
    Version beta
    '''
    #Vemos el frame con la correccion gamma
    plt.title("Frame con correccion gamma")
    plt.imshow(hsv_final)
    plt.show()
    '''
    Version
    '''
    # Convertir de nuevo a BGR
    frame_final = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
    
    '''
    Version beta
    '''
    #Vemos el frame final
    plt.title("Frame final")
    plt.imshow(frame_final)
    plt.show()
    '''
    Version beta
    '''
    
    # Aplicar la máscara y mostrar el resultado
    resultado = cv2.bitwise_and(frame_final, frame_final, mask=mask_dynamic)
    cv2.imshow('Lineas resaltadas', resultado)

    # Escribimos el frame procesado en el archivo de video de salida
    frame_salida = cv2.resize(resultado, (frame_width, frame_height))
    out.write(frame_salida)
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()
