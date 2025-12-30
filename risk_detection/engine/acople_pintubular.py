# risk_detection/engine/acople_pintubular.py
import time, numpy as np
from shapely.geometry import Polygon, Point
from .base_scene import BaseScene
from utils.geometry_utils import has_all_classes, boxes_to_polys_by_name, make_line_from_stickout_to_llavetm, point_in_or_touch_poly, feet_distance_to_geom
from utils.pose_utils import iter_feet
from utils.visualization import draw_polygon, draw_line, put_text

class AcoplePintubular(BaseScene):
    name = "acople_pintubular"

    def __init__(self, cfg):
        super().__init__(cfg)

        # Contador de frames que recuerda si hubo un pintubular cerca recientemente
        self.pintubular_proximity_memory = 0

        # Esto permite que si el pintubular deja de detectarse justo al acoplarse, aún recordemos que estaba ahí.
        self.MEMORY_PERSISTENCE = 30
    
    def _check_pintubular_context(self, det_obj):
        """
        Verifica si hay un 'pintubular' cerca del 'stickout'.
        Si lo hay, recarga la memoria de proximidad.
        """
        req = ["stickout", "pintubular"]

        # Si no detectamos ambos, simplemente decrementamos la memoria y salimos
        if not has_all_classes(det_obj, req):
            if self.pintubular_proximity_memory > 0:
                self.pintubular_proximity_memory -= 1
            return

        polys = boxes_to_polys_by_name(det_obj, req)
        stickout = polys["stickout"]
        pintubular = polys["pintubular"]

        # Calcular distancia entre stickout y pintubular
        dist = stickout.distance(pintubular)
        # print(dist)
        
        # Umbral de cercanía para considerar que van a acoplarse (en píxeles)
        if dist < self.cfg.ACOPLE_PROXIMITY_THRESHOLD:
            # Si el pintubular esta cerca del stickout recargamos la memoria al máximo
            self.pintubular_proximity_memory = self.MEMORY_PERSISTENCE
        elif self.pintubular_proximity_memory > 0:
            # Si se alejan o se pierde la detección, la memoria decae gradualmente
            self.pintubular_proximity_memory -= 1
    
    def _analyze_height_change(self, det_obj):
        """
        Analiza el cambio de altura del stickout usando MEDIANA para evitar ruido.
        Retorna (True/False) si hubo un incremento significativo.
        """
        req = ["stickout"]
        if not has_all_classes(det_obj, req):
            return False, 0

        polys = boxes_to_polys_by_name(det_obj, req)
        s = polys["stickout"]
        h = (s.bounds[3] - s.bounds[1]) # Altura actual
        area = s.area

        # Agregar al buffer
        self.heights_stickout.append(h)
        if len(self.heights_stickout) > self.cfg.ACOPLE_HEIGHT_BUFFER:
            self.heights_stickout.pop(0)

        # Se necesitan suficientes datos para comparar
        if len(self.heights_stickout) < 5:
            return False, area

        # La mediana ignora los valores atípicos (outliers/parpadeos).
        # Usamos los valores anteriores (excluyendo los más recientes para tener contraste)
        altura_base = np.median(self.heights_stickout[:-3])
        
        # Evitar división por cero
        altura_base = max(altura_base, 1.0)

        # Calcular incremento relativo
        inc_rel = (h - altura_base) / altura_base
        # print(inc_rel)

        # Condición de salto:
        # 1. El incremento es mayor al umbral
        # 2. El área es suficiente (evita falsos positivos con stickouts muy lejanos/pequeños)
        is_jump = (inc_rel > self.cfg.ACOPLE_INC_MIN) and (area > self.cfg.ACOPLE_AREA_MIN_STICKOUT)
        
        return is_jump, area
    
    def _check_people_nearby(self, det_obj, res_pose):
        """
        Verifica si hay personas cerca del stickout.
        El acople es manual, así que debe haber humanos.
        """
        if not res_pose or len(res_pose) == 0:
            return False
            
        req = ["stickout"]
        if not has_all_classes(det_obj, req):
            return False
            
        polys = boxes_to_polys_by_name(det_obj, req)
        if polys:
            stickout_poly = polys["stickout"]
            # Verificar si algún pie o punto clave está cerca
            feet = list(iter_feet(res_pose, self.cfg.FEET_IDXS))
            risk = feet_distance_to_geom(feet, stickout_poly, self.cfg.ACOPLE_PIE_PROX_PX)
        else:
            risk = False

        if risk:
            return True
                
        return False
    
    def _confirm_scene(self, is_height_jump, has_pintubular_context, people_nearby):
        """
        Lógica central de decisión.
        Solo activa la escena si se cumplen TODAS las condiciones de robustez.
        """
        # 1. Hubo un salto de altura en el stickout
        # 2. Y ADEMÁS, recordamos haber visto un pintubular cerca recientemente
        # 3. Y ADEMÁS, hay gente cerca operando
        
        valid_scene_trigger = is_height_jump and (self.pintubular_proximity_memory > 0) and people_nearby

        self.increment_scene_active_pos_neg(valid_scene_trigger)

        # print(valid_scene_trigger,self.scene_active_pos)

        # Confirmar acople si la condición se mantiene estable
        if not self.scene_active and self.scene_active_pos >= self.cfg.ACOPLE_SCENE_ON:
            self.activate_scene()
            self.initialize_time() # Guardar t0 para la ventana de tiempo

    def _window_remaining(self):
        if not self.scene_active or self.t0 is None:
            return 0.0
        left = self.cfg.ACOPLE_WINDOW_SEC - (time.time() - self.t0)
        # print(left)
        return max(0.0, left)

    def _risk_window_polygon(self, res_pose, frame):
        poly_np = self.cfg.POLIGONO_RIESGO_STICKOUT_LLAVETM120
        poly = Polygon(poly_np)
        risk = False

        # Verificar pies en polígono
        for x, y in iter_feet(res_pose, self.cfg.FEET_IDXS):
            if point_in_or_touch_poly([x,y], poly):
                risk = True

        if frame is not None and self.cfg.VISUALIZE:
            draw_polygon(frame, poly_np, active=risk)
            rem = int(self._window_remaining())
            debug_color = (0, 255, 0) if self.pintubular_proximity_memory > 0 else (0, 0, 255)
            put_text(frame, f"Ventana: {rem}s | Memoria Pintubular: {self.pintubular_proximity_memory}", (20, 120), color=debug_color)

        return risk

    def evaluate(self, det_obj, res_pose, frame):
        # 1. Actualizar contexto (¿Dónde está el pintubular?)
        self._check_pintubular_context(det_obj)
        
        # 2. Analizar cambio físico (¿Creció el stickout?)
        is_jump, area = self._analyze_height_change(det_obj)

        # print(is_jump)
        
        # 3. Analizar contexto humano (¿Hay operarios?)
        people_nearby = self._check_people_nearby(det_obj, res_pose)

        # 4. Validar activación de escena
        self._confirm_scene(is_jump, self.pintubular_proximity_memory > 0, people_nearby)

        risk = False
        
        # Solo evaluamos riesgo si la escena está activa y dentro de la ventana de tiempo
        if self.scene_active and self._window_remaining() > 0:
            risk = self._risk_window_polygon(res_pose, frame)

            self.increment_risk_active_pos_neg(risk)

            if self.risk_active_pos >= self.cfg.ACOPLE_RISK_ON:
                self.activate_risk()
            elif self.risk_active_neg >= self.cfg.ACOPLE_RISK_OFF:
                self.deactivate_risk()

        elif self.scene_active and self._window_remaining() <= 0:
            # Se acabó el tiempo de la ventana de riesgo (ej. pasaron 20s desde el acople)
            self.deactivate_scene()
            # Reiniciar memoria de pintubular para evitar reactivación inmediata falsa
            self.pintubular_proximity_memory = 0 

        self.log_state()
        return self.make_result(self.scene_active, self.risk_active)