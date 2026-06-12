import numpy as np
import torch
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ---------------------------------- HARDWARE & MODELOS ----------------------------------
    # MODEL_OBJ: Modelo de Detección de Objetos y EPP (Boots, Gloves, Helmet, Person, etc.)
    MODEL_OBJ = "risk_detection/trained_model/yolo26m-yoloepp-sintest-aumentodatax3.pt"  
    
    # MODEL_POSE: Modelo de Estimación de Pose (YOLOv11-pose)
    MODEL_POSE = "risk_detection/trained_model/yolo26m-pose.pt"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ---------------------------------- FUENTE DE VIDEO ----------------------------------
    # VIDEO_SOURCE = os.getenv("RTSP_URL", "rtsp://admin:password@192.168.1.10:554/cam/realmonitor")
    # VIDEO_SOURCE = "risk_detection/videos/recorte_primer_minuto_epp_pruebas.mp4"
    VIDEO_SOURCE = "risk_detection/videos/dos_personas_agachan_alzar_tubular.mp4"
    
    # Redimensionamiento para inferencia y visualización
    # RESIZE = (2560, 1440)
    RESIZE = (1280, 720)
    
    # ¿Grabar salida a disco localmente?
    WRITE_OUTPUT = False
    OUTPUT_PATH = "output_session3.mp4"

    # ---------------------------------- UMBRALES E INFERENCIA ----------------------------------
    CONF_OBJ = 0.5       # Confianza para detección de objetos (EPP/Personas)
    CONF_POSE = 0.5      # Confianza para detección de pose
    
    # Tracking y Fusión
    IOU_TRACK = 0.5      # IoU para el tracker de objetos (ByteTrack interno de YOLO)
    IOU_FUSION = 0.4     # IoU mínimo para asociar una Detección(Persona) con una Pose(Esqueleto)
    IOU_EPP = 0.3        # IoU mínimo para asociar objetos EPP con una Persona

    PERSON_CLASS_ID = 3

    # ---------------------------------- ZONA DE TRABAJO ----------------------------------
    # Umbrales de confirmación para entrada/salida de zona (histéresis)
    ZONE_ENTRY_THRESHOLD = 30   # Frames consecutivos DENTRO para confirmar entrada
    ZONE_EXIT_THRESHOLD = 30    # Frames consecutivos FUERA para confirmar salida
    ZONE_SNAPSHOT_INTERVAL = 60  # Segundos entre snapshots de personas en zona

    # ---------------------------------- EPP ----------------------------------
    EPP_REQUIRED = ["helmet", "boots", "gloves"]  # EPP obligatorios (sin safety_glasses)
    EPP_CONFIRM_THRESHOLD = 200  #Frames consecutivos para confirmar faltante (~0.33s a 15 FPS)
    EPP_VALIDATION_MARGIN = 60  # Margen en píxeles hacia adentro para validar coordenadas (x,y) de keypoints


    # ---------------------------------- REBA ASSESSMENT SCORE A----------------------------------
    REBA_ENABLED = True
    REBA_MIN_CONFIDENCE = 0.5      # Confianza mínima de orientación (0.0-1.0)
    REBA_HISTORY_SIZE = 15          # Frames para suavizado temporal
    REBA_LOAD_SCORE = 1            # Load score fijo
    REBA_HIGH_THRESHOLD = 8        # Score >= 8 se considera alto riesgo
    
    REBA_WRIST_SCORE = 1 # score de wrist fijo (step 9)

    # ---------------------------------- MAC ASSESSMENT SCORE (Lifting) --------------------------
    MAC_ENABLED = True
    MAC_CONSTANT_A = 4 # Load weight/frequency (Orange Zone Assumed)
    MAC_CONSTANT_E = 0 # Postural constraints
    MAC_CONSTANT_F = 0 # Grip on the load
    MAC_CONSTANT_G = 1 # Floor surface
    MAC_CONSTANT_H = 1 # Environmental factors
    MAC_HIGH_THRESHOLD = 13  # Score MAC >= 13 se considera alto riesgo y se registra en SQLite

    # TABLA A para el cruce y calculo del Score A
    # Tabla A: [neck][trunk][leg] -> posture_score_a
    TABLE_A = {
        # Neck = 1
        1: {
            1: {1: 1, 2: 2, 3: 3, 4: 4},
            2: {1: 2, 2: 3, 3: 4, 4: 5},
            3: {1: 2, 2: 4, 3: 5, 4: 6},
            4: {1: 3, 2: 5, 3: 6, 4: 7},
            5: {1: 4, 2: 6, 3: 7, 4: 8},
        },
        # Neck = 2
        2: {
            1: {1: 1, 2: 2, 3: 3, 4: 4},
            2: {1: 3, 2: 4, 3: 5, 4: 6},
            3: {1: 4, 2: 5, 3: 6, 4: 7},
            4: {1: 5, 2: 6, 3: 7, 4: 8},
            5: {1: 6, 2: 7, 3: 8, 4: 9},
        },
        # Neck = 3
        3: {
            1: {1: 3, 2: 3, 3: 5, 4: 6},
            2: {1: 4, 2: 5, 3: 6, 4: 7},
            3: {1: 5, 2: 6, 3: 7, 4: 8},
            4: {1: 6, 2: 7, 3: 8, 4: 9},
            5: {1: 7, 2: 8, 3: 9, 4: 9},
        }
    }
    
    # TABLA B: [upper_arm][lower_arm][wrist] -> posture_score_b
    TABLE_B = {
        1: {  # Upper Arm Score = 1
            1: {1: 1, 2: 2, 3: 2},  # Lower Arm = 1, Wrist 1..3
            2: {1: 1, 2: 2, 3: 3},  # Lower Arm = 2, Wrist 1..3
        },
        2: {  # Upper Arm Score = 2
            1: {1: 1, 2: 2, 3: 3},
            2: {1: 2, 2: 3, 3: 4},
        },
        3: {  # Upper Arm Score = 3
            1: {1: 3, 2: 4, 3: 5},
            2: {1: 4, 2: 5, 3: 5},
        },
        4: {  # Upper Arm Score = 4
            1: {1: 4, 2: 5, 3: 5},
            2: {1: 5, 2: 6, 3: 7},
        },
        5: {  # Upper Arm Score = 5
            1: {1: 6, 2: 7, 3: 8},
            2: {1: 7, 2: 8, 3: 8},
        },
        6: {  # Upper Arm Score = 6
            1: {1: 7, 2: 8, 3: 8},
            2: {1: 8, 2: 9, 3: 9},
        },
    }
    
    # ---------------------------------- REBA ASSESSMENT SCORE TOTAL (TABLE C)----------------------------------
    # TABLA C para el cálculo del REBA Score Total
    # Tabla C: [score_a][score_b] -> reba_score_total
    # Score A: 1-12 (fila)
    # Score B: 1-12 (columna)
    # REBA Score Total: 1-12
    TABLE_C = {
        # Score A = 1
        1: {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 7, 12: 7},
        # Score A = 2
        2: {1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5, 8: 6, 9: 6, 10: 7, 11: 7, 12: 8},
        # Score A = 3
        3: {1: 2, 2: 3, 3: 3, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 7, 10: 8, 11: 8, 12: 8},
        # Score A = 4
        4: {1: 3, 2: 4, 3: 4, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 8, 10: 9, 11: 9, 12: 9},
        # Score A = 5
        5: {1: 4, 2: 4, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 8, 9: 9, 10: 9, 11: 9, 12: 9},
        # Score A = 6
        6: {1: 6, 2: 6, 3: 6, 4: 7, 5: 8, 6: 8, 7: 9, 8: 9, 9: 10, 10: 10, 11: 10, 12: 10},
        # Score A = 7
        7: {1: 7, 2: 7, 3: 7, 4: 8, 5: 9, 6: 9, 7: 9, 8: 10, 9: 10, 10: 11, 11: 11, 12: 11},
        # Score A = 8
        8: {1: 8, 2: 8, 3: 8, 4: 9, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 11, 11: 11, 12: 11},
        # Score A = 9
        9: {1: 9, 2: 9, 3: 9, 4: 10, 5: 10, 6: 10, 7: 11, 8: 11, 9: 11, 10: 12, 11: 12, 12: 12},
        # Score A = 10
        10: {1: 10, 2: 10, 3: 10, 4: 11, 5: 11, 6: 11, 7: 11, 8: 12, 9: 12, 10: 12, 11: 12, 12: 12},
        # Score A = 11
        11: {1: 11, 2: 11, 3: 11, 4: 11, 5: 12, 6: 12, 7: 12, 8: 12, 9: 12, 10: 12, 11: 12, 12: 12},
        # Score A = 12
        12: {1: 12, 2: 12, 3: 12, 4: 12, 5: 12, 6: 12, 7: 12, 8: 12, 9: 12, 10: 12, 11: 12, 12: 12},
    }

    # ---------------------------------- MAC (Manual Handling Assessment Charts) ----------------------------------
    MAC_ENABLED = True
    MAC_LIFTING_ZONE_POLY = np.array([
        [564, 421], [814, 435], [836, 655], [514, 643]
    ], dtype=np.int32)
    MAC_KNEE_ANGLE_THRESHOLD = 30    # Grados mínimos de flexión de rodilla para detectar lifting
    MAC_TRUNK_ANGLE_THRESHOLD = 30   # Grados mínimos de inclinación del tronco para detectar lifting
    MAC_ENTRY_THRESHOLD = 15          # Frames consecutivos para confirmar escenario de lifting
    MAC_EXIT_THRESHOLD = 60          # Frames consecutivos para confirmar escenario de salide de lifting

    # ---------------------------------- HORARIOS Y PAUSAS ----------------------------------
    # Horas en las que el sistema se pausa
    HOURS_PAUSE = [int(x) for x in os.environ.get("HOURS_SCHEDULER_ACTIVE", "[1, 4, 7, 10, 13, 16, 19, 22]").strip("[]").split(", ")]
    MINUTES_PAUSE = 1 # Minutos de duración de la pausa

    # ---------------------------------- DIRECTORIOS Y LOGS ----------------------------------
    LOG_DIR = os.environ.get("LOG_DIR", "logs")
    CLIPS_DIR = os.environ.get("CLIPS_DIR", "risk_clips")
    
    # Monitor de rendimiento (CPU/GPU)
    MONITOR_PERFORMANCE = False

    # ---------------------------------- VISUALIZACIÓN ----------------------------------
    # VISUALIZE = True if os.environ.get("VISUALIZE", "False") == "True" else False
    VISUALIZE = True 
    
    # Índices de Keypoints (Formato COCO estándar)
    KEYPOINT_INDICES = {
        "nose": 0,
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7,    "right_elbow": 8,
        "left_wrist": 9,    "right_wrist": 10,
        "left_hip": 11,     "right_hip": 12,
        "left_knee": 13,    "right_knee": 14,
        "left_ankle": 15,   "right_ankle": 16
    }

    FEET_IDXS = (17, 18) # Serían los nuevos puntos que han sido proyectados
    HAND_IDXS = (19, 20) # # Serían los nuevos puntos que han sido proyectados
    ARMS_IDXS = [
            (7, 9),  # Brazo Izquierdo (codo, muñeca)
            (8, 10)  # Brazo Derecho (codo, muñeca)
        ]
    KNEES_ANKLES_IDX = [
            (13, 15),  # Pierna izquierda (rodilla, tobillo)
            (14, 16)   # Pierna derecha (rodilla, tobillo)
        ]
    
    MANO_EXTENSION_FACTOR = 0.65     # Factor de extensión de la mano respecto al antebrazo (0.40 significa que la mano tendria de longitud el 40% del antebrazo)
    FOOT_EXTENSION_FACTOR = 0.3  # Factor de extensión del pie respecto a la rodilla y tobillo (0.30 significa que el pie tendria de longitud el 30% del antebrazo)
    

    # ---------------------------------- ZONAS DE TRABAJO ----------------------------------
    WORK_ZONE_POLY = np.array([
        [435, 292], [887, 292], [1017, 572],
        [1279, 574], [1279, 719], [105, 719]
    ], dtype=np.int32)

    # Cuántos frames debe mantenerse alguien en la zona para confirmar presencia
    ZONE_PRESENCE_PERSISTENCE = 5
    
    # ---------------------------------- BALIZA Y CLIPS ----------------------------------
    CLIP_ENABLED = True if os.environ.get("CLIP_ENABLED", "True") == "True" else False
    CLIP_PREROLL_SEC = int(os.environ.get("CLIP_PREROLL_SEC", 2))
    
    # ----------------------------------  Detección color de casco (helmet)----------------------------------
    HELMET_COLOR_ENABLED = True
    HELMET_CLASS_NAMES = ["helmet"]  # nombre exacto que viene en detections.data["class_name"] 

    # Perfil: "auto" (según hora), "day", "night"
    HELMET_COLOR_PROFILE = "auto"

    # Horario día/noche (America/Bogota)
    HELMET_DAY_START = 6   # 06:00
    HELMET_DAY_END = 17    # 17:00

    # Estabilización temporal
    HELMET_COLOR_HISTORY_LEN = 7
    HELMET_COLOR_MIN_VOTE = 3
    HELMET_COLOR_MAX_AGE_FRAMES = 60

    # Umbral asignación casco->persona (lógica actual por IoU/top-zone)
    HELMET_ASSIGN_SCORE_THRESHOLD = 0.18
    HELMET_TOP_RATIO = 0.60

    # HSV por perfil (solo white, yellow, orange)
    HELMET_HSV_PROFILES = {
        "day": {
            "white": {"s_max": 70, "v_min": 205},
            "orange": {"h_min": 11, "h_max": 23, "s_min": 90, "v_min": 90},
            "yellow": {"h_min": 24, "h_max": 38, "s_min": 80, "v_min": 110},
        },
        "night": {
            "white": {"s_max": 90, "v_min": 170},
            "orange": {"h_min": 11, "h_max": 23, "s_min": 80, "v_min": 70},
            "yellow": {"h_min": 24, "h_max": 38, "s_min": 70, "v_min": 80},
        },
    }