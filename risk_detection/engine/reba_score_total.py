"""
REBA (Rapid Entire Body Assessment) Score Total Evaluator

Este módulo calcula el REBA Score Total para evaluar riesgo ergonómico usando
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


class REBAEvaluatorScoreTotal(BaseScene):
    """
    Evaluador de REBA Score Total basado en pose keypoints.
    
    Calcula:
    - Posture Score Total (lookup Table C)
    """
    name = "reba_score_a"
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.TABLE_C = self.cfg.TABLE_C

    def _lookup_table_c(self, score_a, score_b):
        """
        Busca REBA Score Total en Tabla C.
        
        Args:
            score_a: 1-12 (fila)
            score_b: 1-12 (columna)
        
        Returns:
            reba_total: 1-12
        """
        try:
            # Limitar a rangos válidos
            score_a = min(max(score_a, 1), 12)
            score_b = min(max(score_b, 1), 12)
            return self.TABLE_C[score_a][score_b]
        except KeyError:
            # Fallback si hay valores fuera de rango
            return 1
    
    def evaluate(self, detection_data: Detection, **kwargs):
        """
        Calcula REBA Score Total para cada persona a partir de los Score A y B calculados anteriormente.
        
        Args:
            fused_entities: Lista de entidades con keypoints
            detection_data: Objeto Detection con Person objects
            
        Returns:
            detection_data: Actualizado con reba_score_a y reba_score_a_conf
        """
        # Mapear Person por track_id
        # persons_map = {p.track_id: p for p in detection_data.people}
        
        # FASE 6: Calcular REBA Score Total usando Tabla C
        # Tabla C: [score_a][score_b] -> reba_total
        for person in detection_data.people:
            if person.reba_score_a > 0 and person.reba_score_b > 0:
                person.reba_total = self._lookup_table_c(
                    person.reba_score_a, 
                    person.reba_score_b
                )
                raw_bad_pose = person.reba_total >= 8
                
                person.bad_pose["bad_pose_reba"] = self.update_hysteresis(
                    track_id=person.track_id, 
                    condition=raw_bad_pose, 
                    pos_threshold=30, 
                    neg_threshold=50
                )
            else:
                person.reba_total = 0
                person.bad_pose["bad_pose_reba"] = self.update_hysteresis(
                    track_id=person.track_id, 
                    condition=False, 
                    pos_threshold=30, 
                    neg_threshold=50
                )
        
        return detection_data