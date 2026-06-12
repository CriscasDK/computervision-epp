"""
REBA (Rapid Entire Body Assessment) Score A Evaluator

Este módulo calcula el REBA Score A para evaluar riesgo ergonómico usando
keypoints de pose 2D. Incluye cálculo de confianza basado en orientación
de la persona (lateral vs frontal).

Autor: Sistema de Evaluación de eventos de Riesgos
"""

import numpy as np
from collections import deque
from statistics import mode, StatisticsError
from engine.base_scene import BaseScene
from engine.models import Detection
from utils.geometry_utils import angle_between_vectors
from utils.pose_utils import normalize_keypoints
from utils.body_angles import calculate_trunk_angle, calculate_knee_angle, calculate_legs_angles, calculate_neck_angle


class REBAEvaluatorScoreA(BaseScene):
    """
    Evaluador de REBA Score A basado en pose keypoints.
    
    Calcula:
    - Neck Score (1-3)
    - Trunk Score (1-5)
    - Leg Score (1-4)
    - Posture Score A (lookup Table A)
    - Score A Final (Posture A + Load Score)
    - Confidence Score (orientación lateral vs frontal)
    """
    name = "reba_score_a"
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        # Historial de scores para suavizado temporal
        self.score_history = {}  # track_id -> deque
        self.confidence_history = {}  # track_id -> deque
        self.history_size = self.cfg.REBA_HISTORY_SIZE

        self.TABLE_A = self.cfg.TABLE_A
        
        # Umbrales de confianza
        self.min_confidence = self.cfg.REBA_MIN_CONFIDENCE
        
        # Load score fijo
        self.load_score = self.cfg.REBA_LOAD_SCORE
    
    def evaluate(self, fused_entities, detection_data: Detection, **kwargs):
        """
        Calcula REBA Score A para cada persona con keypoints válidos.
        
        Args:
            fused_entities: Lista de entidades con keypoints
            detection_data: Objeto Detection con Person objects
            
        Returns:
            detection_data: Actualizado con reba_score_a y reba_score_a_conf
        """
        # Mapear Person por track_id
        persons_map = {p.track_id: p for p in detection_data.people}
        
        for entity in fused_entities:
            track_id = entity["track_id"]
            keypoints = entity["keypoints"]

            # print(f"Keypoints: {keypoints}")
            
            # Validar que existan keypoints
            if keypoints is None or len(keypoints) == 0:
                continue
            
            # Validar que la persona exista en detection_data
            if track_id not in persons_map:
                continue
            
            person = persons_map[track_id]
            
            # # 1. Validar pose (keypoints necesarios presentes)
            # print(f"Track_id: {track_id} - Es valida: {self._is_valid_pose(keypoints)}")
            # print(f"Track_id: {track_id}")
            if not self._is_valid_pose(keypoints, track_id):
                # print(f"Track_id con _is_valid_pose en False - {track_id}")
                person.is_valid_pose = False
                person.reba_score_a = 0
                person.reba_score_conf = 0
                continue

            # 2. Normalizar keypoints (invariante a distancia de cámara)
            keypoints, torso_height = normalize_keypoints(keypoints)
            
            if torso_height == 0:
                person.reba_score_a = 0
                person.reba_score_conf = 0.0
                continue
            
            # 3. Calcular confianza de orientación
            confidence = self._calculate_orientation_confidence(keypoints, torso_height)
            
            # 4. Solo calcular REBA si la confianza es suficiente
            if confidence < self.min_confidence:
                person.is_confidence = False
                person.reba_score_a = 0
                person.reba_score_conf = confidence
                continue
            
            # 5. Calcular ángulos
            neck_angle = calculate_neck_angle(keypoints)
            trunk_angle = calculate_trunk_angle(keypoints)
            legs_angle = calculate_legs_angles(keypoints)
            
            # 6. Calcular scores individuales
            neck_score = self._get_neck_score(neck_angle)
            trunk_score = self._get_trunk_score(trunk_angle)
            leg_score = self._get_leg_score(legs_angle)
            
            # 7. Lookup Posture Score A en Tabla A
            posture_score_a = self._lookup_table_a(neck_score, trunk_score, leg_score)
            # print(f"Track_id: {track_id} - Trunk: {trunk_angle:.1f}° - Trunk_score: {trunk_score} - Leg_angle: {legs_angle}° - Leg_score: {leg_score}")
            # print(f"Track_id: {track_id} - Neck: {neck_angle:.1f}° - Trunk: {trunk_angle:.1f}° - Leg_angle: {legs_angle} - Posture_score_a: {posture_score_a} - Conf: {confidence:.2f}")
            # print(f"Track_id: {track_id} - Neck_score: {neck_score:.1f} - Trunk_score: {trunk_score:.1f} - Leg_score: {leg_score} - Posture_score_a: {posture_score_a} - Conf: {confidence:.2f}")
            
            # 8. Score A final = Posture A + Load Score
            score_a = posture_score_a + self.load_score
            
            # 9. Suavizado temporal
            stable_score = self._get_stable_score(track_id, score_a)
            stable_confidence = self._get_stable_confidence(track_id, confidence)
            
            # 10. Actualizar Person
            person.reba_score_a = stable_score
            person.reba_score_conf = stable_confidence
            person.is_valid_pose = True
            person.is_confidence = True
        
        return detection_data
    
    # ========== VALIDACIÓN ==========
    
    def _is_valid_pose(self, keypoints, track_id):
        """
        Verifica que los keypoints necesarios estén presentes con confianza suficiente.
        
        Keypoints requeridos:
        - 3, 4: ears (al menos una)
        - 5, 6: shoulders
        - 11, 12: hips
        - 13, 14: knees
        """
        # Keypoints que DEBEN estar presentes
        required_indices = [5, 6, 11, 12, 13, 14, 15, 16]
        min_conf = 0.6
         
        for idx in required_indices:
            kp = keypoints[idx]
            
            # Verificar confianza
            if kp[2] < min_conf:
                # print(f"Track_id {track_id} - Keypoint invalido: {idx} / {kp}")
                return False
        
        # Al menos una oreja debe tener buena confianza
        left_ear_conf = keypoints[3][2]
        right_ear_conf = keypoints[4][2]
        
        if left_ear_conf < min_conf and right_ear_conf < min_conf:
            return False
        
        return True
    
    # ========== CONFIANZA DE ORIENTACIÓN ==========
    
    def _calculate_orientation_confidence(self, keypoints, torso_height):
        """
        Calcula confianza de que la persona está en vista lateral.
        Considera la confianza de detección (v) de los puntos de YOLO.
        
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

        # 4. Visibilidad de la cara (Puntos 1-4: ojos y oídos)
        # De perfil puro, solo se ve un lado de la cara.
        # Si detectamos ambos ojos con alta confianza, definitivamente NO estamos de perfil.
        # left_eye_conf = keypoints[1][2]
        # right_eye_conf = keypoints[2][2]
        # face_frontal_factor = min(left_eye_conf, right_eye_conf) # Si ambos son altos, es frontal
        
        # Convertir ratios a scores de confianza
        # Menor ratio = mayor confianza (más lateral)
        
        # Convertir ratios a scores (Ajustados para ser más estrictos)
        #3. Shoulder confidence:
        # El ratio de 0.1 es un perfil perfecto. El ratio de 0.5 es frontal.
        shoulder_conf = np.clip(1.0 - (shoulder_ratio - 0.1) / 0.5, 0.0, 1.0)
        
        # Hip confidence
        hip_conf = np.clip(1.0 - (hip_ratio - 0.1) / 0.5, 0.0, 1.0)
        
        # Aspect confidence
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
    
    # ========== SCORES INDIVIDUALES ==========
    
    def _get_neck_score(self, angle):
        """
        Calcula Neck Score basado en ángulo.
        
        Reglas:
        - Flexión 0-20°: +1
        - Flexión >20°: +2
        - Extensión: +2
        
        Args:
            angle: Ángulo en grados (+ flexión, - extensión)
        
        Returns:
            score: 1-3
        """
        # if angle < 0:
        #     # Extensión
        #     return 2
        if angle <= 20:
            # Flexión leve
            return 1
        else:
            # angle > 20:
            # Flexión moderada
            return 2
    
    def _get_trunk_score(self, angle):
        """
        Calcula Trunk Score basado en ángulo.
        
        Reglas:
        - Erguido (0°): +1
        - Flexión 0-20°: +2
        - Flexión 20-60°: +3
        - Flexión >60°: +4
        
        Args:
            angle: Ángulo en grados
        
        Returns:
            score: 1-5
        """
        if angle <= 10:
            # Erguido o extensión - Se deja hasta 10 por variaciones, pero segun la teoria deberia ser 0
            return 1
        elif angle <= 20:
            # Flexión leve hacia delante
            return 2
        elif angle <= 60:
            # Flexión moderada
            return 3
        else:
            # Flexión severa
            return 4
    
    def _get_leg_score(self, legs_angle):
        """
        Calcula Leg Score basado en flexión de rodillas según REBA Step 3.
        
        Reglas REBA:
        - Piernas casi rectas (0-30° flexión): +1
        - Rodillas flexionadas (30-60° flexión): +2
        
        Ajustes:
        - Add +1 si flexión 30-60°
        - Add +2 si flexión >60°
        
        Returns:
            score: 1-4
        """
        # Angulos de la pierna izquiera y derecha
        left_knee_angle = legs_angle[0]
        right_knee_angle = legs_angle[1]
        
        # Usar el ángulo disponible o el promedio
        # Usamos el promedio de ambas rodillas si están disponibles
        if left_knee_angle is not None and right_knee_angle is not None:
            knee_angle = (left_knee_angle + right_knee_angle) / 2
        elif left_knee_angle is not None:
            knee_angle = left_knee_angle
        elif right_knee_angle is not None:
            knee_angle = right_knee_angle
        else:
            # No hay ángulos válidos, retornar score mínimo
            return 1
        
        # Calcular score base según flexión
        if knee_angle <= 30:
            # Piernas casi rectas
            base_score = 1
            adjustment = 0
        elif knee_angle <= 60:
            # Flexión moderada (30-60°)
            base_score = 1
            adjustment = 1  # Add +1
        else:
            # Flexión alta (>60°)
            base_score = 1
            adjustment = 2  # Add +2
        
        total_score = base_score + adjustment
        
        # Limitar a rango válido [1, 4]
        return min(max(total_score, 1), 4)
    
    # NOTA: _calculate_knee_angle ha sido extraído a utils/body_angles.py
    # Se importa como: calculate_knee_angle
    
    # ========== TABLA A LOOKUP ==========
    
    def _lookup_table_a(self, neck_score, trunk_score, leg_score):
        """
        Busca Posture Score A en Tabla A.
        
        Args:
            neck_score: 1-3
            trunk_score: 1-5
            leg_score: 1-4
        
        Returns:
            posture_score_a: 1-9
        """
        try:
            return self.TABLE_A[neck_score][trunk_score][leg_score]
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
