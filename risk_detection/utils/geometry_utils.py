# risk_detection/utils/geometry_utils.py
import numpy as np
from shapely.geometry import Polygon, Point, LineString, box as shapely_box

def xyxy_to_polygon(xyxy):
    x1, y1, x2, y2 = xyxy
    return shapely_box(x1, y1, x2, y2)

def boxes_to_polys_by_name(det_obj, names_filter):
    """det_obj: sv.Detections con .xyxy y det_obj.data['class_name'] np.ndarray"""
    cls = det_obj.data.get("class_name", None)
    if cls is None:
        return {}
    mask = np.isin(cls, names_filter)
    polys = {}
    for name, box in zip(cls[mask], det_obj.xyxy[mask]):
        polys[name] = xyxy_to_polygon(box)
    return polys

def has_all_classes(det_obj, required):
    cls = det_obj.data.get("class_name", None)
    if cls is None:
        return False
    return set(required).issubset(set(cls))

def point_in_or_touch_poly(point_xy, poly: Polygon):
    p = Point(float(point_xy[0]), float(point_xy[1]))
    return p.within(poly) or p.touches(poly)

def feet_distance_to_geom(feet_points, geom, thresh):
    for x, y in feet_points:
        # print(Point(x, y).distance(geom))
        if Point(x, y).distance(geom) <= thresh:
            return True
    return False

def clamp_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def inner_box(x1, y1, x2, y2, margin=0.12):
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    return x1 + mx, y1 + my, x2 - mx, y2 - my


def calculate_iou_xyxy(box1, box2):
    """
    Calcula Intersection over Union (IoU) entre dos cajas en formato xyxy.
    
    Esta es la implementación consolidada y optimizada de IoU para todo el proyecto.
    Reemplaza las implementaciones anteriores en fusion_utils.py y geometry_utils.py.
    
    Args:
        box1: Caja en formato [x1, y1, x2, y2] (puede ser list, tuple, o np.array)
        box2: Caja en formato [x1, y1, x2, y2] (puede ser list, tuple, o np.array)
    
    Returns:
        float: Valor de IoU entre 0.0 y 1.0
            - 0.0: Sin intersección
            - 1.0: Cajas idénticas
    
    Example:
        >>> box_a = [0, 0, 100, 100]
        >>> box_b = [50, 50, 150, 150]
        >>> iou = calculate_iou_xyxy(box_a, box_b)
        >>> print(f"IoU: {iou:.2f}")
        IoU: 0.14
    """
    # Desempaquetar coordenadas
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2
    
    # Calcular coordenadas de intersección
    x1_inter = max(x1_a, x1_b)
    y1_inter = max(y1_a, y1_b)
    x2_inter = min(x2_a, x2_b)
    y2_inter = min(y2_a, y2_b)
    
    # Calcular área de intersección (0 si no hay overlap)
    inter_width = max(0.0, x2_inter - x1_inter)
    inter_height = max(0.0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height
    
    # Si no hay intersección, retornar 0
    if intersection_area == 0:
        return 0.0
    
    # Calcular áreas de cada caja
    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
    
    # Calcular unión
    union_area = area_a + area_b - intersection_area
    
    # Evitar división por cero
    if union_area <= 0:
        return 0.0
    
    # Retornar IoU
    return float(intersection_area / union_area)

def angle_between_vectors(v1, v2):
    """
    Calcula ángulo entre dos vectores en grados.
    
    Returns:
        angle: Ángulo en grados [0, 180]
    """
    #Primero calculamos el producto punto
    # A.B = |A||B|cos(theta)

    # Normalizar vectores
    #Esto hace que las flechas midan exactamente 1.0. Al medir 1, las magnitudes (|A| y |B|) desaparecen de la ecuación porque multiplicar por 1 no cambia nada.
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

    #Producto Punto Ahora, cos(theta) = A_norm.B_norm.
    
    # Producto punto
    dot_product = np.dot(v1_norm, v2_norm)
    
    # Clamp para evitar errores numéricos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Ángulo en radianes y luego grados
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg
