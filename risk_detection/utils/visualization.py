# risk_detection/utils/visualization.py
import cv2
import numpy as np

def draw_polygon(frame, poly_np, active=False):
    color = (0, 255, 0) if not active else (0, 0, 255)
    cv2.polylines(frame, [poly_np.astype(np.int32)], True, color, 2)

def draw_line(frame, p1, p2, active=False):
    color = (0, 255, 0) if not active else (0, 0, 255)
    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 3)

def put_text(frame, text, org=(20, 40), color=(255,255,255), scale=0.8, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

def draw_hud(frame, fps=None, lines=[], pose_result=None, final_detection=None, fused_entities=None):
    """
    Dibuja HUD y Esqueletos.
    Args:
        frame: Imagen.
        fps: Frames por segundo.
        lines: Lista de strings de estado.
        pose_result: Objeto raw de Ultralytics con keypoints.
    """
    if frame is None: return

    h, w = frame.shape[:2]

    # FONDO HUD (Barra Superior Semitransparente)
    header_height = max(80, 40 + (len(lines) * 30))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # TEXTO FPS
    if fps is not None:
        # fps_color = (0, 255, 0) if fps > 15 else (0, 0, 255)
        put_text(frame, f"FPS: {fps:.1f}", (w - 180, 40), (0, 255, 0), 0.8, 2)

    # TEXTO ESTADO DE ZONAS
    y_start = 35
    for s in lines:
        if "RIESGO" in s and "NO" not in s: col = (0, 0, 255)
        elif "NORMAL" in s or "OK" in s:    col = (0, 255, 0)
        else:                               col = (200, 200, 200)
        
        put_text(frame, s, (20, y_start), col, 0.6, 2)
        y_start += 30

  # DIBUJAR ESQUELETO
    # Normalizar input
    pose_obj = None
    if isinstance(pose_result, list) and len(pose_result) > 0:
        pose_obj = pose_result[0]
    elif pose_result is not None:
        pose_obj = pose_result
    # 1. Crear un diccionario para acceder rápido al REBA Score Total de cada persona usando su track_id
    reba_scores = {}
    if final_detection and final_detection.people:
        for p in final_detection.people:
            reba_scores[p.track_id] = p.reba_total
    # 2. Dibujar esqueletos usando las entidades fusionadas que ya tienen un ID asignado a cada "pose"
    if fused_entities is not None:
        SKELETON_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),        # Cabeza
            (5, 6),                                # Hombros
            (5, 7), (7, 9),                        # Brazo izq
            (6, 8), (8, 10),                       # Brazo der
            (5, 11), (6, 12),                      # Torso
            (11, 12),                              # Caderas
            (11, 13), (13, 15),                    # Pierna izq
            (12, 14), (14, 16),                    # Pierna der
            # --- (Pies Proyectados) ---
            (15, 17),                              # Tobillo Izq -> Punta Pie Izq (Nuevo índice 17)
            (16, 18),                              # Tobillo Der -> Punta Pie Derc (Nuevo índice 18)
            # --- (Manos Proyectadas) ---
            (9, 19),                               # Muñeca Izq -> Punta Mano Izq (Nuevo índice 19)
            (10, 20)                               # Muñeca Der -> Punta Mano Derc (Nuevo índice 20)
        ]
        hand_points = [19, 20]
        foot_points = [17, 18]
        COLOR_HAND = (0, 255, 255)
        COLOR_FOOT = (255, 0, 255)
        for entity in fused_entities:
            kp_set = entity.get("keypoints")
            track_id = entity.get("track_id")
            # Continuar solo si este individuo tiene keypoints detectados
            if kp_set is None or len(kp_set) == 0:
                continue
            # Extraer el REBA de esta persona (usa 0 por defecto si no fue calculado)
            reba_total = reba_scores.get(track_id, 0)
            # --- NUEVA LÓGICA DE SEMÁFORO ---
            # REBA Score igual a 0 (no calculado o fuera de condiciones): azul
            if reba_total == 0:
                COLOR_SKELETON = (255, 0, 0)
            # REBA Score entre 1 y 3 (inclusivo): verde
            elif 1 <= reba_total <= 3:
                COLOR_SKELETON = (0, 255, 0)
            # REBA Score entre 4 y 7: amarillo
            elif 4 <= reba_total <= 7:
                COLOR_SKELETON = (0, 255, 255) # Nota: el amarillo en OpenCV (BGR) es (0, 255, 255)
            # REBA Score mayor o igual a 8: rojo
            elif reba_total >= 8:
                COLOR_SKELETON = (0, 0, 255)
            # Dibujar Conexiones
            for idx1, idx2 in SKELETON_CONNECTIONS:
                if idx1 < len(kp_set) and idx2 < len(kp_set):
                    pt1 = (int(kp_set[idx1][0]), int(kp_set[idx1][1]))
                    pt2 = (int(kp_set[idx2][0]), int(kp_set[idx2][1]))
                    if pt1[0] != 0 and pt1[1] != 0 and pt2[0] != 0 and pt2[1] != 0:
                        cv2.line(frame, pt1, pt2, COLOR_SKELETON, 2)
            # Dibujar Puntos conservando los colores iniciales para manos y pies
            for idx, pt in enumerate(kp_set):
                x, y = int(pt[0]), int(pt[1]) # Extraemos la coordenada x, y
                if x == 0 and y == 0: continue
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                if idx in hand_points:
                    cv2.circle(frame, (x, y), 4, COLOR_HAND, -1)
                elif idx in foot_points:
                    cv2.circle(frame, (x, y), 6, COLOR_FOOT, -1)
                    
def draw_text_custom(frame, text_list, pos=(20, 20), color=(0, 255, 0), scale=0.6):
    """
    Dibuja una lista de textos con fondo semitransparente.
    """
    x, y = pos
    gap = 25
    
    for line in text_list:
        # Fondo negro para legibilidad
        cv2.putText(frame, line, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 2)
        # Texto color
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += gap

def draw_polygon_overlay(frame, poly_np, color, alpha=0.3):
    """
    Dibuja un polígono relleno transparente.
    """
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly_np], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


