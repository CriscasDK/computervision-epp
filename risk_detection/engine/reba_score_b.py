"""
REBA (Rapid Entire Body Assessment) Score B Evaluator

Este módulo calcula el REBA Score B para evaluar riesgo ergonómico de brazos
usando keypoints de pose 2D. Incluye cálculo de confianza basado en orientación
de la persona (lateral vs frontal).

El Score B evalúa:
- Upper Arm Position (Step 7): Score 1-4 por brazo
- Lower Arm Position (Step 8): Score 1-2 por brazo
- Wrist Position (Step 9): Score 1 (simplificado)
- Posture Score B = Lookup Table B(Upper Arm Score, Lower Arm Score, Wrist Score)
- Score B Final = Posture Score B + Wrist Score (1)

Cuando ambos brazos cumplen el umbral de confianza, se suman los scores individuales.

Autor: Sistema de Evaluación de eventos de Riesgos
"""

import numpy as np
from collections import deque
from statistics import mode, StatisticsError
from engine.base_scene import BaseScene
from engine.models import Detection
from utils.geometry_utils import angle_between_vectors
from utils.pose_utils import normalize_keypoints
from utils.body_angles import calculate_upper_arm_angle, calculate_lower_arm_angle


class REBAEvaluatorScoreB(BaseScene):
    """
    Evaluador de REBA Score B basado en pose keypoints.
    
    Calcula:
    - Upper Arm Score (1-4) por brazo - Step 7
    - Lower Arm Score (1-2) por brazo - Step 8
    - Wrist Score (1) - Step 9 (siempre +1)
    - Posture Score B (lookup Table B)
    - Score B Final (Posture B + Wrist Score)
    - Confidence Score (orientación lateral vs frontal)
    
    Si ambos brazos cumplen confianza, se suman los scores.
    """
    name = "reba_score_b"
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        # Historial de scores para suavizado temporal
        self.score_history = {}  # track_id -> deque
        self.confidence_history = {}  # track_id -> deque
        self.history_size = self.cfg.REBA_HISTORY_SIZE
        
        # Tabla B hardcodeada
        self.TABLE_B = self.cfg.TABLE_B
        
        # Umbrales de confianza
        self.min_confidence = self.cfg.REBA_MIN_CONFIDENCE
        self.min_kp_confidence = 0.5  # Confianza mínima para keypoints individuales
        
        # Wrist score fijo (Step 9 siempre 1)
        self.wrist_score = self.cfg.REBA_WRIST_SCORE
    
    def evaluate(self, fused_entities, detection_data: Detection, **kwargs):
        """
        Calcula REBA Score B para cada persona con keypoints válidos.
        
        Solo calcula el Score B si la persona tiene is_valid_pose=True y 
        is_confidence=True (validados previamente en REBA Score A).
        
        Si ambos brazos cumplen el umbral de confianza, se suman los scores.
        
        Args:
            fused_entities: Lista de entidades con keypoints
            detection_data: Objeto Detection con Person objects
            
        Returns:
            detection_data: Actualizado con reba_score_b y reba_score_b_conf
        """
        # Mapear Person por track_id
        persons_map = {p.track_id: p for p in detection_data.people}
        
        for entity in fused_entities:
            track_id = entity["track_id"]
            keypoints = entity["keypoints"]
            
            # Validar que existan keypoints
            if keypoints is None or len(keypoints) == 0:
                continue
            
            # Validar que la persona exista en detection_data
            if track_id not in persons_map:
                continue
            
            person = persons_map[track_id]
            
            # 1. Validar que la pose sea válida (calculada previamente en Score A)
            if not person.is_valid_pose:
                person.reba_score_b = 0
                continue
            
            # 2. Validar que la confianza sea suficiente (calculada previamente en Score A)
            if not person.is_confidence:
                person.reba_score_b = 0
                continue
            
            # 3. Normalizar keypoints (invariante a distancia de cámara)
            keypoints, torso_height = normalize_keypoints(keypoints)
            
            if torso_height == 0:
                person.reba_score_b = 0
                continue
            
            # # 4. Calcular confianza de orientación (usar la misma del Score A)
            # confidence = self._calculate_orientation_confidence(keypoints, torso_height)
            
            # 5. Solo calcular REBA B si la confianza es suficiente
            if person.reba_score_conf < self.min_confidence:
                person.reba_score_b = 0
                continue
            
            # 6. Calcular scores para cada brazo
            # print(f"Track id - {track_id}")
            left_upper_score, left_lower_score = self._calculate_arm_scores(keypoints, side='left')
            right_upper_score, right_lower_score = self._calculate_arm_scores(keypoints, side='right')
            
            # 7. Sumar scores si ambos brazos son válidos, o usar el que esté disponible
            upper_arm_score = 0
            lower_arm_score = 0
            
            left_valid = left_upper_score is not None and left_lower_score is not None
            right_valid = right_upper_score is not None and right_lower_score is not None
            
            if left_valid and right_valid:
                # Ambos brazos válidos: sumar scores
                upper_arm_score = (left_upper_score + right_upper_score) // 2
                lower_arm_score = (left_lower_score + right_lower_score) // 2
            elif left_valid:
                # Solo brazo izquierdo válido
                upper_arm_score = left_upper_score
                lower_arm_score = left_lower_score
            elif right_valid:
                # Solo brazo derecho válido
                upper_arm_score = right_upper_score
                lower_arm_score = right_lower_score
            else:
                # Ningún brazo válido
                person.reba_score_b = 0
                continue
            
            # Limitar scores a rangos válidos
            upper_arm_score = min(upper_arm_score, 6)  # Max 6 
            lower_arm_score = min(lower_arm_score, 2)  # Max 2 
            
            # 8. Lookup Posture Score B en Tabla B
            # Tabla B: [lower_arm][upper_arm][wrist]
            posture_score_b = self._lookup_table_b(upper_arm_score, lower_arm_score, self.wrist_score)
            
            # 9. Score B final
            score_b = posture_score_b
            
            # print(f"Track_id: {track_id} - UpperArmScore: {upper_arm_score} - LowerArmScore: {lower_arm_score} - "
            #       f"ScoreB: {score_b}")
            
            # 10. Suavizado temporal
            stable_score = self._get_stable_score(track_id, score_b)
            
            # 11. Actualizar Person
            person.reba_score_b = stable_score
        
        return detection_data
    
    # ========== CONFIANZA DE ORIENTACIÓN ==========
    
    def _calculate_orientation_confidence(self, keypoints, torso_height):
        """
        Calcula confianza de que la persona está en vista lateral.
        
        Combina 3 métricas:
        1. Shoulder width ratio
        2. Hip width ratio
        3. Torso aspect ratio
        
        Returns:
            confidence: 0.0 (frontal/espaldas) a 1.0 (lateral perfecto)
        """
        # 1. Shoulder width ratio
        shoulder_ratio = self._calculate_shoulder_width_ratio(keypoints, torso_height)
        
        # 2. Hip width ratio
        hip_ratio = self._calculate_hip_width_ratio(keypoints, torso_height)
        
        # 3. Torso aspect ratio
        aspect = self._calculate_torso_aspect_ratio(keypoints)
        
        # Convertir ratios a scores
        # Menor ratio = mayor confianza (más lateral)
        shoulder_conf = np.clip(1.0 - (shoulder_ratio - 0.1) / 0.5, 0.0, 1.0)
        hip_conf = np.clip(1.0 - (hip_ratio - 0.1) / 0.5, 0.0, 1.0)
        aspect_conf = np.clip(1.0 - (aspect - 0.2) / 0.6, 0.0, 1.0)
        
        # Combinar (promedio ponderado)
        confidence = (
            0.4 * shoulder_conf +  # Peso mayor a hombros
            0.3 * hip_conf +
            0.3 * aspect_conf
        )
        
        return float(confidence)
    
    def _calculate_shoulder_width_ratio(self, keypoints, torso_height):
        """Calcula ratio de ancho de hombros normalizado."""
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        ratio = shoulder_width if torso_height > 0 else 0
        
        return ratio
    
    def _calculate_hip_width_ratio(self, keypoints, torso_height):
        """Calcula ratio de ancho de caderas normalizado."""
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        
        hip_width = np.linalg.norm(right_hip - left_hip)
        ratio = hip_width if torso_height > 0 else 0
        
        return ratio
    
    def _calculate_torso_aspect_ratio(self, keypoints):
        """Calcula relación ancho/alto del torso."""
        # Ancho promedio
        shoulder_width = np.linalg.norm(keypoints[6][:2] - keypoints[5][:2])
        hip_width = np.linalg.norm(keypoints[12][:2] - keypoints[11][:2])
        avg_width = (shoulder_width + hip_width) / 2
        
        # Alto
        shoulder_mid = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hip_mid = (keypoints[11][:2] + keypoints[12][:2]) / 2
        torso_height = np.linalg.norm(shoulder_mid - hip_mid)
        
        aspect = avg_width if torso_height > 0 else 0
        
        return aspect
    
    # ========== CÁLCULO DE SCORES POR BRAZO ==========
    
    def _calculate_arm_scores(self, keypoints, side='left'):
        """
        Calcula Upper Arm Score y Lower Arm Score para un brazo específico.
        
        Args:
            keypoints: Array de keypoints normalizados
            side: 'left' o 'right'
        
        Returns:
            (upper_arm_score, lower_arm_score) o (None, None) si no es válido
        """
        if side == 'left':
            shoulder_idx = 5
            elbow_idx = 7
            wrist_idx = 9
        else:  # right
            shoulder_idx = 6
            elbow_idx = 8
            wrist_idx = 10
        
        # Verificar que los keypoints existan y tengan confianza suficiente
        shoulder = keypoints[shoulder_idx]
        elbow = keypoints[elbow_idx]
        wrist = keypoints[wrist_idx]
        
        if (shoulder[2] < self.min_kp_confidence or 
            elbow[2] < self.min_kp_confidence or 
            wrist[2] < self.min_kp_confidence):
            return None, None
        
        # Calcular ángulos
        upper_arm_angle = calculate_upper_arm_angle(keypoints, side)
        lower_arm_angle = calculate_lower_arm_angle(keypoints, side)

        # print(f"upper_arm_angle_{side}: {upper_arm_angle} - lower_arm_angle_{side}: {upper_arm_angle}")
        
        if upper_arm_angle is None or lower_arm_angle is None:
            return None, None
        
        # Calcular scores
        upper_arm_score = self._get_upper_arm_score(upper_arm_angle)
        lower_arm_score = self._get_lower_arm_score(lower_arm_angle)

        # print(f"upper_arm_score_{side}: {upper_arm_score} - lower_arm_score_{side}: {lower_arm_score}")
        
        return upper_arm_score, lower_arm_score
    
    # ========== SCORES INDIVIDUALES ==========
    
    def _get_upper_arm_score(self, angle):
        """
        Calcula Upper Arm Score basado en ángulo (Step 7).
        
        Reglas REBA según usuario:
        - -20° a 20° (neutral/extension ligera): +1
        - < -20° (extensión): +2
        - 20° a 45° (flexión leve): +2
        - 45° a 90° (flexión moderada): +3
        - > 90° (flexión severa): +4
        
        Args:
            angle: Ángulo en grados entre brazo superior y torso
        
        Returns:
            score: 1-4
        """
        if -20 <= angle <= 20:
            # Brazo al lado del cuerpo o ligera flexión/extensión
            return 1
        elif angle < -20:
            # Extensión hacia atrás
            return 2
        elif angle <= 45:
            # Flexión leve (20-45°)
            return 2
        elif angle <= 90:
            # Flexión moderada (45-90°)
            return 3
        else:
            # Flexión severa (>90°)
            return 4
    
    def _get_lower_arm_score(self, angle):
        """
        Calcula Lower Arm Score basado en ángulo (Step 8).
        
        Reglas REBA según usuario:
        - 60-100° (posición neutral): +1
        - 0-60° (flexión excesiva): +2
        - 100-180° (extensión excesiva): +2
        
        Args:
            angle: Ángulo en grados entre brazo superior y brazo inferior
        
        Returns:
            score: 1-2
        """
        # Rango neutral: 60-100°
        if 60 <= angle <= 100:
            # Posición neutral
            return 1
        else:
            # Flexión o extensión excesiva
            return 2
    
    # ========== TABLA B LOOKUP ==========
    
    def _lookup_table_b(self, upper_arm_score, lower_arm_score, wrist_score):
        """
        Busca Posture Score B en Tabla B.
        
        Tabla B estructura: [lower_arm][upper_arm][wrist]
        
        Args:
            upper_arm_score: 1-6
            lower_arm_score: 1-2
            wrist_score: 1-3 (siempre 1 en nuestra implementación)
        
        Returns:
            posture_score_b: 1-9
        """
        try:
            return self.TABLE_B[upper_arm_score][lower_arm_score][wrist_score]
        except KeyError:
            # Fallback si hay valores fuera de rango
            return 1
    
    # ========== SUAVIZADO TEMPORAL ==========
    
    def _get_stable_score(self, track_id, current_score):
        """
        Suaviza score usando historial para evitar pestañeos.
        
        Usa moda (valor más frecuente) de los últimos N frames.
        """
        if track_id not in self.score_history:
            self.score_history[track_id] = deque(maxlen=self.history_size)
        
        self.score_history[track_id].append(current_score)
        
        try:
            stable_score = mode(self.score_history[track_id])
        except StatisticsError:
            # Si no hay moda clara, usar el último
            stable_score = current_score
        
        return stable_score
    
    def _get_stable_confidence(self, track_id, current_confidence):
        """
        Suaviza confidence usando promedio móvil.
        """
        if track_id not in self.confidence_history:
            self.confidence_history[track_id] = deque(maxlen=self.history_size)
        
        self.confidence_history[track_id].append(current_confidence)
        
        # Promedio móvil
        stable_confidence = np.mean(self.confidence_history[track_id])
        
        return float(stable_confidence)
    
    # ========== CLEANUP ==========
    
    def cleanup_old_tracks(self, current_track_ids):
        """
        Limpia historial de tracks que ya no están presentes.
        """
        # Score history
        old_ids = set(self.score_history.keys()) - set(current_track_ids)
        for tid in old_ids:
            self.score_history.pop(tid, None)
        
        # Confidence history
        old_ids = set(self.confidence_history.keys()) - set(current_track_ids)
        for tid in old_ids:
            self.confidence_history.pop(tid, None)