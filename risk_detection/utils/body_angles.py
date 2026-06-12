# risk_detection/utils/body_angles.py
"""
Funciones compartidas para cálculo de ángulos corporales.

Usadas por:
- REBAEvaluatorScoreA (Neck, Trunk, Legs)
- MACSceneDetector (Trunk, Knees)
"""
import numpy as np
from utils.geometry_utils import angle_between_vectors

def calculate_neck_angle(keypoints):
    """
    Calcula ángulo del cuello respecto al vector del torso.
    
    Línea Base: Vector del torso (hip_mid → shoulder_mid)
    Vector Cuello: desde punto medio hombros → oreja con mayor confianza
    
    Returns:
        angle: Ángulo en grados (positivo=flexión, negativo=extensión)
    """
    # Puntos del torso
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]
    
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    
    # Vector del torso (línea base)
    torso_vector = shoulder_mid - hip_mid
    
    # Seleccionar oreja con mayor confianza
    left_ear = keypoints[3]  # [x, y, conf]
    right_ear = keypoints[4]  # [x, y, conf]

    # Usar la oreja con mayor confianza
    if left_ear[2] > right_ear[2]:
        ear_point = left_ear[:2]
    else:
        ear_point = right_ear[:2]
    
    # Vector del cuello (desde hombros a oreja)
    neck_vector = ear_point - shoulder_mid
    
    # Calcular ángulo entre vector del torso y vector del cuello
    angle = angle_between_vectors(neck_vector, torso_vector)
    
    # # Determinar si es flexión (+) o extensión (-)
    # # Proyección del vector del cuello sobre la perpendicular al torso
    # # Si la oreja está adelante del torso → flexión
    # # Si la oreja está atrás del torso → extensión
    
    # # Perpendicular al torso (rotación 90° en sentido horario)
    # # torso_vector = [dx, dy] → perpendicular = [dy, -dx]
    # torso_perp = np.array([torso_vector[1], -torso_vector[0]])
    
    # # Proyección del cuello sobre la perpendicular
    # projection = np.dot(neck_vector, torso_perp)
    
    # # Si projection > 0 → flexión (adelante)
    # # Si projection < 0 → extensión (atrás)
    # if projection > 0:
    #     return angle
    # else:
    #     return -angle
    return angle


def calculate_trunk_angle(keypoints):
    """
    Calcula ángulo del tronco respecto al vector de las piernas.
    
    Línea Base: Vector de las piernas (projected_feet_mid → hip_mid)
    Vector Tronco: desde punto medio caderas → punto medio hombros
    
    Args:
        keypoints: Array de keypoints normalizados con pies proyectados (17, 18)
    
    Returns:
        angle: Ángulo en grados (positivo=flexión)
    """
    # Puntos del torso
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]
    
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    
    # Vector del torso
    trunk_vector = shoulder_mid - hip_mid
    
    # Usar pies proyectados (índices 17, 18)
    left_foot = keypoints[17][:2]
    right_foot = keypoints[18][:2]
    
    feet_mid = (left_foot + right_foot) / 2
    
    # Vector de las piernas (línea base)
    leg_vector = hip_mid - feet_mid
    
    # Calcular ángulo entre vector de las piernas y vector del torso
    angle = angle_between_vectors(trunk_vector, leg_vector)
    
    return angle


def calculate_knee_angle(keypoints, side='left'):
    """
    Calcula el ángulo de flexión de la rodilla.
    
    Ángulo formado por: hip -> knee -> ankle
    
    Args:
        keypoints: Array de keypoints
        side: 'left' o 'right'
    
    Returns:
        angle: Ángulo de flexión en grados, o None si no es válido
    """
    if side == 'left':
        hip_idx = 11
        knee_idx = 13
        ankle_idx = 15
    else:  # right
        hip_idx = 12
        knee_idx = 14
        ankle_idx = 16
    
    # Verificar que los keypoints existan y tengan confianza suficiente
    min_conf = 0.5
    
    if len(keypoints) <= ankle_idx:
        return None
    
    hip = keypoints[hip_idx]
    knee = keypoints[knee_idx]
    ankle = keypoints[ankle_idx]
    
    # Verificar confianza
    if hip[2] < min_conf or knee[2] < min_conf or ankle[2] < min_conf:
        return None
    
    # Vectores
    hip_point = hip[:2]
    knee_point = knee[:2]
    ankle_point = ankle[:2]
    
    # Vector de muslo (hip -> knee)
    thigh_vector = knee_point - hip_point
    
    # Vector de pantorrilla (knee -> ankle)
    calf_vector = ankle_point - knee_point
    
    # Calcular ángulo entre vectores
    angle = angle_between_vectors(thigh_vector, calf_vector)
    
    return angle


def calculate_legs_angles(keypoints):
    """
    Calcula el ángulo de flexión de ambas rodillas.
    
    Returns:
        tuple: (left_knee_angle, right_knee_angle) en grados.
               Cada valor puede ser None si no es válido.
    """
    left_knee_angle = calculate_knee_angle(keypoints, side='left')
    right_knee_angle = calculate_knee_angle(keypoints, side='right')
    
    return left_knee_angle, right_knee_angle

def calculate_upper_arm_angle(keypoints, side='left'):
    """
    Calcula el ángulo del brazo superior respecto al tronco (Step 7).
    
    Línea Base: Vector del torso (hip_mid → shoulder_mid)
    Vector Brazo Superior: desde hombro → codo
    
    Args:
        keypoints: Array de keypoints
        side: 'left' o 'right'
    
    Returns:
        angle: Ángulo en grados, o None si no es válido
    """
    if side == 'left':
        shoulder_idx = 5
        elbow_idx = 7
    else:  # right
        shoulder_idx = 6
        elbow_idx = 8
    
    # Puntos del torso
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]
    
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    
    # Vector del torso (línea base, apuntando hacia arriba)
    torso_vector = shoulder_mid - hip_mid
    
    # Puntos del brazo
    shoulder_point = keypoints[shoulder_idx][:2]
    elbow_point = keypoints[elbow_idx][:2]
    
    # Vector del brazo superior (hombro → codo)
    upper_arm_vector = elbow_point - shoulder_point
    
    # Calcular ángulo entre vector del torso y vector del brazo superior
    angle = angle_between_vectors(upper_arm_vector, torso_vector)
    
    return angle
    
def calculate_lower_arm_angle(keypoints, side='left'):
    """
    Calcula el ángulo de la proyección de la línea hombro-codo con 
    la línea codo-muñeca (Step 8).
    
    Vector 1: hombro → codo (brazo superior)
    Vector 2: codo → muñeca (brazo inferior/antebrazo)
    
    Args:
        keypoints: Array de keypoints
        side: 'left' o 'right'
    
    Returns:
        angle: Ángulo en grados, o None si no es válido
    """
    if side == 'left':
        shoulder_idx = 5
        elbow_idx = 7
        wrist_idx = 9
    else:  # right
        shoulder_idx = 6
        elbow_idx = 8
        wrist_idx = 10
    
    # Puntos
    shoulder_point = keypoints[shoulder_idx][:2]
    elbow_point = keypoints[elbow_idx][:2]
    wrist_point = keypoints[wrist_idx][:2]
    
    # Vector brazo superior (hombro → codo)
    upper_arm_vector = elbow_point - shoulder_point
    
    # Vector brazo inferior (codo → muñeca)
    lower_arm_vector = wrist_point - elbow_point
    
    # Calcular ángulo entre ambos vectores
    angle = angle_between_vectors(upper_arm_vector, lower_arm_vector)
    
    return angle
