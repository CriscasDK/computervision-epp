from shapely.geometry import box
from .base_scene import BaseScene
from .models import Person, Detection
import numpy as np

class EPPMonitor(BaseScene):
    name = "monitor_epp_inventario"

    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Umbral de confianza para keypoints de extremidades
        self.kp_conf_threshold = 0.7
        self.low_conf_threshold = 0.2
        
        # Wrists (para guantes): 9=left_wrist, 10=right_wrist
        # Ankles (para botas): 15=left_ankle, 16=right_ankle
        # Head (para casco): 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
        self.wrist_indices = [9, 10]
        self.ankle_indices = [15, 16]
        self.head_indices = [0, 1, 2, 3, 4]
        
        # Histéresis de EPP Faltante
        self.epp_required = set(cfg.EPP_REQUIRED)
        self.epp_confirm_threshold = cfg.EPP_CONFIRM_THRESHOLD
        self.epp_noncompliant_tracks = {}  # {track_id: {'pending_frames': int, 'missing': set()}}
        self.epp_registered_tracks = set() # track_ids ya alertados


    def _check_extremity_visibility(self, keypoints):
        """
        Verifica si las extremidades relevantes para EPP son visibles y están dentro del recuadro seguro.
        
        Retorna:
            dict: con booleanos individuales para 'wrists_evaluable' y 'ankles_evaluable'
        """
        if keypoints is None or len(keypoints) == 0:
            return {'wrists_evaluable': False, 'ankles_evaluable': False}
        
        # Calcular los límites del "sub-recuadro"
        margin = self.cfg.EPP_VALIDATION_MARGIN
        w, h = self.cfg.RESIZE
        x_min, y_min = margin, margin
        x_max, y_max = w - margin, h - margin
        
        def is_valid_point(idx):
            if idx >= len(keypoints): return False
            pt = keypoints[idx]
            
            # Usar umbral bajo si el punto pertenece a la cabeza
            thresh = self.low_conf_threshold if idx in self.head_indices else self.kp_conf_threshold
            if pt[2] <= thresh: return False
            
            x, y = int(pt[0]), int(pt[1])
            # Validar que caiga estrictamente dentro del margen
            return (x_min < x < x_max) and (y_min < y < y_max)
        
        wrists_evaluable = any(is_valid_point(idx) for idx in self.wrist_indices)
        ankles_evaluable = any(is_valid_point(idx) for idx in self.ankle_indices)
        head_evaluable = any(is_valid_point(idx) for idx in self.head_indices)
        
        return {
            'wrists_evaluable': wrists_evaluable,
            'ankles_evaluable': ankles_evaluable,
            'head_evaluable': head_evaluable
        }
    
    
    def evaluate(self, fused_entities, frame, detection_data: Detection, **kwargs):
        """
        Lee EPP de la estructura fusionada (sin calcular IoU).
        Valida visibilidad de extremidades antes de asignar EPP.
        
        Args:
            fused_entities: Lista de entidades fusionadas con EPP incluido
            frame: Frame actual (no usado, mantenido por compatibilidad)
            detection_data: Objeto Detection con lista de Person
        
        Returns:
            Detection: Objeto actualizado con Person.epp llenado
        """
        # Si no hay personas en el objeto, devolvemos tal cual
        if not detection_data.people:
            return detection_data
        
        # Crear mapas track_id -> epp_dict y track_id -> keypoints
        epp_map = {}
        kps_map = {}
        for entity in fused_entities:
            tid = entity["track_id"]
            epp_map[tid] = entity["epp"]
            kps_map[tid] = entity.get("keypoints")
        
        # Llenar Person.epp desde fusión
        for person in detection_data.people:
            tid = person.track_id
            
            if tid not in epp_map:
                person.epp = []
                person.epp_evaluable = False
                continue
            
            # Verificar visibilidad de extremidades usando keypoints
            keypoints = kps_map.get(tid)
            person.epp_evaluable = self._check_extremity_visibility(keypoints)
            
            # Convertir dict de listas a lista de nombres de EPP presentes
            epp_items = epp_map[tid]
            current_person_epp = [
                epp_type 
                for epp_type, boxes in epp_items.items() 
                if boxes  # Si tiene al menos 1 caja de este tipo
            ]
            
            # --- EVALUACIÓN DE EPP FALTANTE CON HISTÉRESIS DENTRO DE MARGEN ---
            if tid not in self.epp_registered_tracks:
                current_epp = set(current_person_epp)
                current_missing = self.epp_required - current_epp
                
                # Filtrar faltantes irreales (fuera de la cámara o de baja confianza)
                verified_missing = set()
                for missing_item in current_missing:
                    # Guantes requiere ambas condiciones válidas (confianza y dentro del margen)
                    if missing_item == 'gloves' and not person.epp_evaluable.get('wrists_evaluable', False):
                        continue 
                    # Botas requiere ambas condiciones válidas (confianza y dentro del margen)
                    if missing_item == 'boots' and not person.epp_evaluable.get('ankles_evaluable', False):
                        continue
                    # Casco requiere que la cabeza sea evaluable (ojos/orejas/nariz visibles en recuadro)
                    if missing_item == 'helmet' and not person.epp_evaluable.get('head_evaluable', False):
                        continue
                        
                    verified_missing.add(missing_item)
                
                if verified_missing:
                    # Faltan EPPs: acumular frames
                    if tid not in self.epp_noncompliant_tracks:
                        self.epp_noncompliant_tracks[tid] = {'pending_frames': 1, 'missing': verified_missing}
                    else:
                        # Si el conjunto faltante cambió de repente o se mantuvo, actualizamos para no alertar mixto
                        if self.epp_noncompliant_tracks[tid]['missing'] != verified_missing:
                            self.epp_noncompliant_tracks[tid] = {'pending_frames': 1, 'missing': verified_missing}
                        else:
                            self.epp_noncompliant_tracks[tid]['pending_frames'] += 1
                        
                        if self.epp_noncompliant_tracks[tid]['pending_frames'] >= self.epp_confirm_threshold:
                            # ¡Alerta disparada!
                            person.epp_alert_triggered = True
                            person.missing_epps_str = '_'.join(verified_missing)

                            current_person_epp_compliant = self.epp_required - verified_missing
                            person.epp = list(current_person_epp_compliant)
                            
                            self.epp_registered_tracks.add(tid)
                            del self.epp_noncompliant_tracks[tid]
                else:
                    # Posee todos los EPP (o los faltantes no son evaluables)
                    if tid in self.epp_noncompliant_tracks:
                        del self.epp_noncompliant_tracks[tid]
                 
        return detection_data