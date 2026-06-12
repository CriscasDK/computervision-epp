from abc import ABC, abstractmethod
from datetime import datetime
import pytz

class BaseScene(ABC):
    """
    Clase abstracta base para lógica de visión.
    Define la estructura que deben tener todos los monitores de zona o eventos.
    """
    name: str = "base_scene"

    def __init__(self, cfg):
        self.cfg = cfg
        self.active = False
        
        self.track_states = {}
        
        self.tz = pytz.timezone("America/Bogota")

    @abstractmethod
    def evaluate(self, tracks, frame, detection_data, **kwargs):
        """
        Args:
            tracks: Resultados del tracking de YOLO (Boxes + Keypoints + IDs)
            frame: Imagen actual (para debug/dibujo)
            **kwargs: Argumentos extra (ej: raw_detections para EPP)
        Return:
            dict: Resultados procesados
        """
        raise NotImplementedError("Implementar evaluate()")
    
    def update_hysteresis(self, track_id: int, condition: bool, pos_threshold: int = 20, neg_threshold: int = 100) -> bool:
        """
        Calcula la histéresis por cada persona (track_id) independientemente.
        """
        if track_id not in self.track_states:
             self.track_states[track_id] = {'active_pos': 0, 'active_neg': 0, 'state': False}
             
        state = self.track_states[track_id]
        
        if condition:
            state['active_pos'] += 1
            state['active_neg'] = 0
        else:
            state['active_neg'] += 1
            state['active_pos'] = 0

        if state['active_pos'] >= pos_threshold:
            state['state'] = True
        elif state['active_neg'] >= neg_threshold:
            state['state'] = False
            
        return state['state']

    def make_result(self, in_zone_ids, risk_ids, count):
        """Formato estandarizado de respuesta"""
        return {
            "timestamp": datetime.now(self.tz).isoformat(),
            "scene_name": self.name,
            "people_in_zone_count": count,
            "people_ids_in_zone": in_zone_ids, # Lista de IDs
            "people_ids_bad_pose": risk_ids,   # Lista de IDs con mala postura
            "scene_active": count > 0,         # La escena está "activa" si hay gente
            "event_active": len(risk_ids) > 0   # Evento si hay mala postura
        }