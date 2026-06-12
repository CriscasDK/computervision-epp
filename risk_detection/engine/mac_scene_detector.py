"""
MAC (Manual Handling Assessment Charts) - Detección de Escenario de Levantamiento

Este módulo detecta cuándo un trabajador está realizando un levantamiento de carga
verificando 4 condiciones simultáneas:
1. Al menos 1 pie proyectado (kp 17/18) dentro del polígono de lifting
2. Pose válida (is_valid_pose == True, ya evaluado por REBA Score A)
3. Al menos 1 rodilla flexionada > umbral
4. Ángulo de tronco > umbral

Cuando las 4 condiciones se confirman durante N frames consecutivos (histéresis),
se marca mac_lifting_detected = True para habilitar la evaluación MAC posterior.
"""

import numpy as np
from shapely.geometry import Point, Polygon
from engine.base_scene import BaseScene
from engine.models import Detection
from utils.body_angles import calculate_trunk_angle, calculate_knee_angle
from utils.pose_utils import normalize_keypoints


class MACSceneDetector(BaseScene):
    """
    Detecta escenario de levantamiento de carga para evaluación MAC.
    
    Verifica las 4 condiciones simultáneas por track_id con histéresis
    temporal para evitar falsos positivos por parpadeos de pose.
    """
    name = "mac_scene_detector"
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.polygon = Polygon(cfg.MAC_LIFTING_ZONE_POLY)
        self.poly_np = cfg.MAC_LIFTING_ZONE_POLY
        
        # Umbrales configurables
        self.knee_angle_threshold = cfg.MAC_KNEE_ANGLE_THRESHOLD
        self.trunk_angle_threshold = cfg.MAC_TRUNK_ANGLE_THRESHOLD
        self.entry_threshold = cfg.MAC_ENTRY_THRESHOLD
        self.exit_threshold = cfg.MAC_EXIT_THRESHOLD
        
        # Índices de pies proyectados
        self.projected_feet_indices = [17, 18]
        
        # Estado de histéresis por track_id
        # {track_id: {'consecutive_frames': int, 'confirmed': bool}}
        self.lifting_state = {}
    
    def _is_foot_in_zone(self, keypoints):
        """
        Verifica si al menos un pie proyectado (kp 17/18) está dentro del polígono de lifting.
        
        Args:
            keypoints: Array de keypoints (sin normalizar, coordenadas originales de imagen)
        
        Returns:
            bool: True si al menos un pie proyectado está dentro del polígono
        """
        for idx in self.projected_feet_indices:
            if idx >= len(keypoints):
                continue
            
            pt = keypoints[idx]
            # Verificar confianza mínima del keypoint
            if pt[2] < 0.6:
                continue
            
            point = Point(float(pt[0]), float(pt[1]))
            # Verificar si está dentro o en el borde del polígono
            if self.polygon.contains(point) or self.polygon.distance(point) < 1.0:
                return True
        
        return False
    
    def _check_knee_flexion(self, keypoints_normalized):
        """
        Verifica si al menos una rodilla tiene flexión superior al umbral.
        
        Args:
            keypoints_normalized: Keypoints normalizados para cálculo de ángulos
        
        Returns:
            bool: True si al menos una rodilla supera el umbral de flexión
        """
        left_knee = calculate_knee_angle(keypoints_normalized, side='left')
        right_knee = calculate_knee_angle(keypoints_normalized, side='right')
        
        if left_knee is not None and left_knee > self.knee_angle_threshold:
            return True
        if right_knee is not None and right_knee > self.knee_angle_threshold:
            return True
        
        return False
    
    def _check_trunk_inclination(self, keypoints_normalized):
        """
        Verifica si el ángulo de tronco supera el umbral.
        
        Args:
            keypoints_normalized: Keypoints normalizados para cálculo de ángulos
        
        Returns:
            bool: True si el tronco está inclinado más allá del umbral
        """
        trunk_angle = calculate_trunk_angle(keypoints_normalized)
        return trunk_angle > self.trunk_angle_threshold
    
    def _update_lifting_hysteresis(self, track_id, all_conditions_met):
        """
        Actualiza la histéresis temporal para confirmar entrada/salida de lifting.
        
        Usa contadores ascendentes independientes (mismo patrón que WorkZoneMonitor):
        - ENTRADA: N frames consecutivos con las 4 condiciones → confirmed = True
        - SALIDA: N frames consecutivos SIN las 4 condiciones → confirmed = False
        
        Args:
            track_id: ID del track
            all_conditions_met: True si las 4 condiciones se cumplen en este frame
        
        Returns:
            bool: True si el lifting está confirmado
        """
        if track_id not in self.lifting_state:
            self.lifting_state[track_id] = {
                'frames_active': 0,     # Frames consecutivos cumpliendo condiciones
                'frames_inactive': 0,   # Frames consecutivos SIN cumplir condiciones
                'confirmed': False
            }
        
        state = self.lifting_state[track_id]
        
        if all_conditions_met:
            # Condiciones cumplidas: incrementar contador de entrada, resetear salida
            state['frames_active'] += 1
            state['frames_inactive'] = 0
        else:
            # Condiciones NO cumplidas: incrementar contador de salida, resetear entrada
            state['frames_inactive'] += 1
            state['frames_active'] = 0
        
        # Lógica de histéresis con umbrales independientes
        if not state['confirmed']:
            # Estado actual: NO lifting → verificar entrada
            if state['frames_active'] >= self.entry_threshold:
                state['confirmed'] = True
        else:
            # Estado actual: SÍ lifting → verificar salida
            if state['frames_inactive'] >= self.exit_threshold:
                state['confirmed'] = False
        
        return state['confirmed']
    
    def evaluate(self, fused_entities, detection_data: Detection, **kwargs):
        """
        Evalúa escenario de levantamiento de carga para cada persona.
        
        Verifica las 4 condiciones simultáneas con histéresis temporal:
        1. Pie proyectado dentro del polígono de lifting
        2. Pose válida (is_valid_pose == True)
        3. Rodilla flexionada > umbral
        4. Tronco inclinado > umbral
        
        Args:
            fused_entities: Lista de entidades fusionadas con keypoints
            detection_data: Objeto Detection con Person objects
        
        Returns:
            Detection: detection_data actualizado con mac_lifting_detected
        """
        if not detection_data.people:
            return detection_data
        
        # Mapear Person por track_id
        persons_map = {p.track_id: p for p in detection_data.people}
        
        # Track IDs activos en este frame (para limpieza posterior)
        active_track_ids = set()
        
        for entity in fused_entities:
            track_id = entity["track_id"]
            keypoints_raw = entity["keypoints"]
            active_track_ids.add(track_id)
            
            # La persona debe existir en detection_data
            if track_id not in persons_map:
                continue
            
            person = persons_map[track_id]
            
            # Si no tiene keypoints, no puede evaluar
            if keypoints_raw is None or len(keypoints_raw) == 0:
                person.mac_lifting_detected = self._update_lifting_hysteresis(track_id, False)
                continue
            
            # ── CONDICIÓN 1: Pie proyectado dentro del polígono de lifting ──
            foot_in_zone = self._is_foot_in_zone(keypoints_raw)
            
            # ── CONDICIÓN 2: Pose válida ──
            valid_pose = person.is_valid_pose
            
            # Si no pasa las condiciones básicas, no calcular ángulos
            if not foot_in_zone or not valid_pose:
                person.mac_lifting_detected = self._update_lifting_hysteresis(track_id, False)
                continue
            
            # Normalizar keypoints para cálculo de ángulos
            keypoints_norm, torso_height = normalize_keypoints(keypoints_raw)
            
            if torso_height == 0:
                person.mac_lifting_detected = self._update_lifting_hysteresis(track_id, False)
                continue
            
            # ── CONDICIÓN 3: Rodilla flexionada > umbral ──
            knee_flexed = self._check_knee_flexion(keypoints_norm)
            
            # ── CONDICIÓN 4: Tronco inclinado > umbral ──
            trunk_inclined = self._check_trunk_inclination(keypoints_norm)
            
            # ── Verificar las 4 condiciones simultáneas ──
            all_conditions = foot_in_zone and valid_pose and knee_flexed and trunk_inclined
            
            # Actualizar histéresis y asignar resultado
            person.mac_lifting_detected = self._update_lifting_hysteresis(track_id, all_conditions)
            
            if person.mac_lifting_detected:
                print(f"🏋️ [MAC] Lifting DETECTADO: track_id={track_id}")
        
        # Limpiar estados de tracks que ya no están presentes
        self._cleanup_old_tracks(active_track_ids)
        
        return detection_data
    
    def _cleanup_old_tracks(self, active_track_ids):
        """Limpia historial de tracks que ya no están presentes."""
        old_ids = set(self.lifting_state.keys()) - active_track_ids
        for tid in old_ids:
            self.lifting_state.pop(tid, None)
