import numpy as np
from utils.geometry_utils import calculate_iou_xyxy
from shapely.geometry import box as shapely_box

# Función helper para validar overlap EPP-Person
def _is_valid_epp_overlap(epp_box, person_box):
    """
    Determina si un EPP pertenece a una persona usando containment ratio.
    
    Criterios:
    1. Contención: Si el EPP está >50% contenido dentro de la persona
    2. IoU fallback: Si hay intersección significativa (>5%)
    
    Esto es mejor que solo IoU porque objetos pequeños (guantes) tienen
    IoU bajo aunque estén completamente dentro de la persona.
    """
    poly_epp = shapely_box(*epp_box)
    poly_person = shapely_box(*person_box)
    
    # Validación básica
    if poly_epp.area == 0 or poly_person.area == 0:
        return False
    
    # Área de intersección
    intersection_area = poly_epp.intersection(poly_person).area
    if intersection_area == 0:
        return False
    
    # 1. Chequeo de Contención (¿Qué % del EPP está dentro de la persona?)
    containment_ratio = intersection_area / poly_epp.area
    if containment_ratio > 0.5:
        return True
    
    # 2. Chequeo de IoU - Fallback para casos donde EPP sobresale
    union_area = poly_epp.union(poly_person).area
    iou = intersection_area / union_area
    if iou > 0.05:
        return True
    
    return False


def fuse_complete_detection(sv_detections, pose_results, target_class_id=3, 
                            iou_threshold_pose=0.4, iou_threshold_epp=0.3):
    """
    Fusión COMPLETA: Person + Pose + EPP en una sola pasada.
    
    Esta función optimiza el pipeline eliminando cálculos duplicados de IoU.
    En lugar de calcular IoU dos veces (EPPMonitor + HelmetColorTracker),
    lo hace una sola vez aquí.
    
    IMPORTANTE: Para EPP usa containment ratio (no solo IoU) porque objetos pequeños
    como guantes tienen IoU muy bajo aunque estén completamente dentro de la persona.
    
    Args:
        sv_detections: Objeto Detections de Supervision (con tracking)
        pose_results: Resultados raw de YOLO Pose
        target_class_id: ID de la clase persona (default: 3)
        iou_threshold_pose: Umbral IoU para matching pose (default: 0.4)
        iou_threshold_epp: Umbral IoU para matching EPP (default: 0.3, usado como fallback)
    
    Returns:
        List[Dict]: Lista de entidades fusionadas con estructura:
            {
                "track_id": int,
                "box": np.array [x1, y1, x2, y2],
                "class_name": str,
                "keypoints": np.array o None,
                "epp": {
                    "helmet": [[x1, y1, x2, y2], ...],
                    "boots": [[x1, y1, x2, y2], ...],
                    "gloves": [[x1, y1, x2, y2], ...],
                    "safety_glasses": [[x1, y1, x2, y2], ...]
                }
            }
    """
    
    fused_entities = []
    
    # Validar entradas
    if sv_detections.xyxy is None or len(sv_detections) == 0:
        return []
    
    # Desempaquetar datos de Supervision
    boxes = sv_detections.xyxy
    class_ids = sv_detections.class_id
    tracker_ids = sv_detections.tracker_id
    class_names = sv_detections.data["class_name"]
    
    # Datos del Modelo Pose
    if pose_results and pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
        pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
        pose_kps = pose_results[0].keypoints.data.cpu().numpy()
    else:
        pose_boxes = []
        pose_kps = []
    
    # PASO 1: Separar personas trackeadas de objetos EPP
    people_indices = []
    epp_indices = []
    epp_classes = ["helmet", "boots", "gloves", "safety_glasses"]
    
    for i in range(len(boxes)):
        cls_id = class_ids[i]
        tid = tracker_ids[i] if tracker_ids is not None else None
        cls_name = str(class_names[i]).lower()
        
        # Personas con tracking
        if tid is not None and cls_id == target_class_id:
            people_indices.append(i)
        # Objetos EPP (sin tracking)
        elif cls_name in epp_classes:
            epp_indices.append(i)
    
    # PASO 2: Para cada persona, hacer matching con Pose y EPP
    for person_idx in people_indices:
        tid = int(tracker_ids[person_idx])
        person_box = boxes[person_idx]
        
        # --- Matching Person ↔ Pose ---
        matched_kps = None
        best_iou_pose = 0
        best_pose_idx = -1
        
        for j, pose_box in enumerate(pose_boxes):
            iou = calculate_iou_xyxy(person_box, pose_box)
            if iou > best_iou_pose:
                best_iou_pose = iou
                best_pose_idx = j
        
        if best_iou_pose > iou_threshold_pose:
            matched_kps = pose_kps[best_pose_idx]
        
        # --- Matching Person ↔ EPP ---
        # --- Matching Person ↔ EPP---
        # Inicializar diccionario de EPP vacío
        epp_dict = {
            "helmet": [],
            "boots": [],
            "gloves": [],
            "safety_glasses": []
        }
        
        # Para cada objeto EPP detectado
        for epp_idx in epp_indices:
            epp_box = boxes[epp_idx]
            epp_class = str(class_names[epp_idx]).lower()
            
            # Usar containment-based matching en lugar de solo IoU
            if _is_valid_epp_overlap(epp_box, person_box):
                epp_dict[epp_class].append(epp_box.tolist())
        
        # --- Construir entidad fusionada ---
        entity = {
            "track_id": tid,
            "box": person_box,
            "class_name": class_names[person_idx],
            "keypoints": matched_kps,
            "raw_class_id": class_ids[person_idx],
            "epp": epp_dict
        }
        fused_entities.append(entity)
    
    return fused_entities

