# risk_detection/engine/extraccion_stickout.py
import numpy as np
from shapely.geometry import Point, Polygon
from .base_scene import BaseScene
from utils.geometry_utils import has_all_classes, boxes_to_polys_by_name, point_in_or_touch_poly
from utils.pose_utils import iter_feet
from utils.visualization import draw_polygon, put_text

class ExtraccionStickout(BaseScene):
    name = "extraccion_stickout"

    def __init__(self, cfg):
        super().__init__(cfg)
        # Buffer para almacenar la posición Y del centroide del brazotaladro
        self.brazo_y_history = []
        # Tamaño de la ventana de tiempo para analizar el movimiento (15 frames = ~1 seg)
        self.MOVEMENT_WINDOW = 30 

    def _instant_condition(self, det_obj):
        """
        Detecta la escena de 'Extracción de Stickout' usando geometría con Shapely.
        
        Se evalúan:
        - Solapamiento espacial entre el stickout y el brazo.
        - Distancia vertical entre sus cajas.
        - Alineación horizontal aproximada.
        """
        
        req = ["stickout", "brazotaladro"]
        if not has_all_classes(det_obj, req):
            return False, None
        polys = boxes_to_polys_by_name(det_obj, req)
        s, b = polys["stickout"], polys["brazotaladro"]

        # Devolvemos el centroide Y del brazo para el análisis de movimiento
        brazo_centroid_y = b.centroid.y

        inter = s.intersection(b).area
        amin = max(min(s.area, b.area), 1.0)
        ratio = inter / amin

        if ratio > self.cfg.EXTR_OVERLAP_MIN:
            return True, brazo_centroid_y

        dist = s.distance(b)
        cx_s = s.centroid.x; cx_b = b.centroid.x
        w_avg = ((s.bounds[2]-s.bounds[0]) + (b.bounds[2]-b.bounds[0]))/2.0
        aligned = abs(cx_s - cx_b) < w_avg * self.cfg.EXTR_ALIGN_RATIO

        is_close = (dist <= self.cfg.EXTR_DIST_PX) and aligned

        return is_close, brazo_centroid_y
    
    def _is_extracting(self, current_y):
        """
        Determina si el movimiento es de EXTRACCIÓN (Hacia Arriba).
        En imagen: Y disminuye al subir.
        """
        if current_y is None:
            return False

        # Agregar al historial
        self.brazo_y_history.append(current_y)
        if len(self.brazo_y_history) > self.MOVEMENT_WINDOW:
            self.brazo_y_history.pop(0)

        # Necesitamos un mínimo de historia para juzgar
        if len(self.brazo_y_history) < 5:
            return False

        # Comparamos el valor actual con el promedio de los primeros frames del buffer.
        # Esto suaviza el ruido "jitter" de la detección frame a frame.
        y_past = np.mean(self.brazo_y_history[:5]) # El pasado (inicio del buffer)
        y_now = np.mean(self.brazo_y_history[-5:]) # El presente (final del buffer)

        delta = y_now - y_past

        print(f"{y_now} - {y_past} = {delta}")

        # Si delta es negativo significativo -> Se mueve hacia arriba (Y disminuye) -> EXTRACCIÓN
        # Si delta es positivo significativo -> Se mueve hacia abajo (Y aumenta) -> INSERCIÓN
        
        MOVEMENT_THRESHOLD = 5.0 # Píxeles de movimiento neto necesarios

        if delta < -MOVEMENT_THRESHOLD:
            return True # Extracción (Subiendo)
        elif delta > MOVEMENT_THRESHOLD:
            return False # Inserción (Bajando)
        else:
            # Si está quieto (delta pequeño), mantenemos el estado anterior de la escena
            # o retornamos True si asumimos que extracción incluye pausas.
            # Si no sube, no es extracción activa.
            # Pero para evitar parpadeos cuando la máquina para un segundo, 
            # podemos retornar True si la escena YA estaba activa.
            return self.scene_active 

    def _risk_polygon(self, res_pose):
        """
        Detecta si algún landmark del pie (izquierdo o derecho)
        está dentro o sobre el polígono de riesgo definido.
        """
        poly_np = self.cfg.POLIGONO_RIESGO_STICKOUT

        poly = Polygon(poly_np)
        for x, y in iter_feet(res_pose, self.cfg.FEET_IDXS):
            if point_in_or_touch_poly([x,y], poly):
                return True
        return False

    def evaluate(self, det_obj, res_pose, frame):
        # 1. Evaluar geometría estática
        is_geometrically_valid, brazo_y = self._instant_condition(det_obj)

        is_moving_up = False
        
        if is_geometrically_valid:
            # 2. Evaluar dinámica (dirección del movimiento)
            is_moving_up = self._is_extracting(brazo_y)

        # 3. La escena es válida SOLO SI geometría OK + movimiento hacia arriba
        scene_active_now = is_geometrically_valid and is_moving_up

        self.increment_scene_active_pos_neg(scene_active_now)

        if self.scene_active_pos >= self.cfg.EXTR_SCENE_ON:
            self.activate_scene()
        elif self.scene_active_neg >= self.cfg.EXTR_SCENE_OFF:
            self.deactivate_scene()

        risk = False

        if self.scene_active:
            risk = self._risk_polygon(res_pose)
            self.increment_risk_active_pos_neg(risk)

            if self.risk_active_pos >= self.cfg.EXTR_RISK_ON:
                self.activate_risk()
            elif self.risk_active_neg >= self.cfg.EXTR_RISK_OFF:
                self.deactivate_risk()

            if frame is not None and self.cfg.VISUALIZE:
                draw_polygon(frame, self.cfg.POLIGONO_RIESGO_STICKOUT, active=self.risk_active)

            # trend_text = "ESTATICO"
            if len(self.brazo_y_history) > 5:
                delta = self.brazo_y_history[-1] - self.brazo_y_history[0]
                if delta < -5: trend_text = "SUBIENDO (Extraccion)"
                elif delta > 5: trend_text = "BAJANDO (Insercion)"
                else: trend_text = "ESTATICO (estatico)"
                put_text(frame, f"Brazo: {trend_text}", (20, 150), color=(255, 255, 0))
                
        self.log_state()
        # print(f"Escena activa: {self.scene_active}, Riesgo activo: {self.risk_active}, Frames positivos: {self.risk_active_pos}, Frames_negativos: {self.risk_active_neg}")
        return self.make_result(self.scene_active, self.risk_active)
