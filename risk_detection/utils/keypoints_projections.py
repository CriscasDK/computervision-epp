import torch
import numpy as np
from ultralytics.engine.results import Results

def _calculate_virtual_projection_point(punto_a, punto_b, EXTENSION_FACTOR):
    """
    Proyecta un punto basándose en el vector A -> B.
    Retorna np.array([x, y]).
    """
    a = np.array(punto_a[:2]) # Aseguramos tomar solo x,y
    b = np.array(punto_b[:2])
    
    vec = b - a
    length = np.linalg.norm(vec)
    
    if length < 1:
        return b # Evitar errores si puntos están pegados
    
    # Vector resultante extendido
    # Multiplicamos el vector unitario (netamente direccion) por la longitud que se quiere aumentar, obteniendo un vector con la misma dirección del unitario y con la longitud tomada del procentaje deseado
    vec_result = (vec / length) * (length * EXTENSION_FACTOR)
    
    # Nuevo punto (proyección desde B)
    new_point = b + vec_result
    return new_point

def _get_virtual_points_for_person(person_kps_np, pairs, extension_factor):
    """
    Genera los puntos virtuales para una sola persona.
    Retorna una lista de arrays [x, y, conf].
    """
    new_points = []
    
    for idx_a, idx_b in pairs:
        # Puntos originales (x, y, conf)
        p1 = person_kps_np[idx_a]
        p2 = person_kps_np[idx_b]
        
        # Calculamos x, y proyectados
        virtual_xy = _calculate_virtual_projection_point(p1, p2, extension_factor)
        
        # GESTIÓN DE LA CONFIANZA:
        # Tomamos la confianza del punto "b"  ya que el pie o mano proyectado depende de que el tobillo o codo sea visible.
        conf = p2[2] if len(p2) > 2 else 0.5 
        
        # Creamos el punto con formato [x, y, conf]
        virtual_point_data = np.array([virtual_xy[0], virtual_xy[1], conf])
        new_points.append(virtual_point_data)
        
    return new_points

def add_virtual_keypoints_to_results(res_pose, pairs, extension_factor):
    """
    Modifica el objeto res_pose in-place agregando keypoints virtuales.
    """
    if not res_pose:
        return res_pose

    for result in res_pose:
        if result.keypoints is None:
            continue
            
        # Obtenemos los keypoints originales
        # original_kps_tensor es un TENSOR de PyTorch
        original_kps_tensor = result.keypoints.data
        original_kps_np = original_kps_tensor.cpu().numpy()
        
        all_virtual_kps = []
        
        for person_idx in range(len(original_kps_np)):
            person_kps = original_kps_np[person_idx]
            virtual_points = _get_virtual_points_for_person(person_kps, pairs, extension_factor)
            all_virtual_kps.append(virtual_points)
            
        if not all_virtual_kps:
            continue


        new_kps_tensor = torch.tensor(
            np.array(all_virtual_kps), 
            device=original_kps_tensor.device,
            dtype=original_kps_tensor.dtype
        )
        
        # Concatenamos
        combined_kps = torch.cat([original_kps_tensor, new_kps_tensor], dim=1)
        
        # Inyectamos de vuelta
        result.keypoints.data = combined_kps
        
        # Limpieza de caché si existe
        if hasattr(result.keypoints, '_xy'):
            del result.keypoints._xy 
            
    return res_pose