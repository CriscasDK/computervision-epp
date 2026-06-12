# risk_detection/utils/pose_utils.py
import numpy as np

def iter_keypoints(res_pose):
    if not res_pose or not hasattr(res_pose[0], "keypoints") or res_pose[0].keypoints is None:
        return []
    return res_pose[0].keypoints.xy.cpu().numpy()  # [N,17,2]

def iter_feet(res_pose, feet_idxs=(15,16)):
    kps = iter_keypoints(res_pose)
    for kp_set in kps:
        for idx in feet_idxs:
            x, y = kp_set[idx]
            yield float(x), float(y)

def normalize_keypoints(keypoints):
    """
    Normaliza keypoints usando altura del torso como referencia.
    Esto hace que los ángulos sean invariantes a la distancia de cámara.
    
    Returns:
        (normalized_kps, torso_height)
    """
    # Separar coordenadas y confianza
    coords = keypoints[:, :2]
    conf = keypoints[:, 2:] # Mantiene forma (N, 1)

    # Calcular puntos medios
    shoulder_mid = (coords[5] + coords[6]) / 2
    hip_mid = (coords[11] + coords[12]) / 2
    
    # Altura del torso
    torso_height = np.linalg.norm(shoulder_mid - hip_mid)
    
    if torso_height == 0:
        return keypoints, 0
    
    # Normalizar (opcional, por ahora solo retornamos la altura)
    # Los ángulos no necesitan normalización si usamos vectores

    # 3. NORMALIZACIÓN REAL:
    # Normalizar solo las coordenadas x, y
    # Restamos hip_mid para que la cadera sea el origen (0,0) 
    # y dividimos por torso_height para que el torso siempre mida 1 unidad.
    normalized_coords = (coords - hip_mid) / torso_height
    # Volver a unir con la confianza original
    normalized_kps = np.hstack([normalized_coords, conf])

    return normalized_kps, torso_height
