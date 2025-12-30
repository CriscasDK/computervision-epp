import cv2
import time
import os

def extraer_frame_rtsp(url_rtsp, nombre_archivo_salida="frame_extraido.jpg"):
    """
    Conecta a una URL RTSP, captura el primer frame disponible y lo guarda como imagen.

    :param url_rtsp: La URL completa del stream RTSP (ej: "rtsp://usuario:contraseña@ip_camara:puerto/ruta").
    :param nombre_archivo_salida: El nombre del archivo donde se guardará la imagen (ej: "frame_extraido.jpg").
    """
    print(f"⌛ Intentando conectar a la URL RTSP: {url_rtsp}")

    # 1. Crear el objeto VideoCapture
    # Usar CAP_FFMPEG puede mejorar la compatibilidad con algunos streams RTSP.
    cap = cv2.VideoCapture(url_rtsp, cv2.CAP_FFMPEG)

    # Esperar un poco para asegurar la conexión.
    time.sleep(1)

    # 2. Verificar si la conexión fue exitosa
    if not cap.isOpened():
        print(f"❌ ¡Error! No se pudo abrir el stream RTSP de la URL: {url_rtsp}")
        return

    # 3. Leer un frame
    # 'ret' es un booleano: True si el frame se leyó correctamente.
    # 'frame' es el frame capturado (un arreglo NumPy).
    ret, frame = cap.read()

    # 4. Liberar el recurso inmediatamente
    cap.release()

    if ret:
        # 5. Guardar el frame como un archivo de imagen
        cv2.imwrite(nombre_archivo_salida, frame)
        print(f"✅ ¡Éxito! Frame extraído y guardado como: **{nombre_archivo_salida}**")
        print(f"   Tamaño del frame (alto, ancho, canales): {frame.shape}")
    else:
        print("⚠️ Advertencia: Se pudo conectar al stream pero no se pudo leer ningún frame.")


# --- Configuración y Uso ---
# IMPORTANTE: Reemplaza esta URL con la URL real de tu cámara o servidor RTSP.
RTSP_URL_DE_EJEMPLO = "rtsp://ANALITICA:4N4L1T1C42025+@192.168.104.55/Streaming/Channels/101"

# Para fines de prueba, puedes buscar URLs de ejemplo públicas si no tienes una cámara a mano.
# Sin embargo, la URL debe ser válida para que el script funcione.

# Ejemplo de uso:
extraer_frame_rtsp(RTSP_URL_DE_EJEMPLO, "camara_frame.png")

# Opcional: Para verificar que la imagen se guardó (si usas un entorno local)
# print(f"El archivo existe: {os.path.exists('camara_frame.png')}")