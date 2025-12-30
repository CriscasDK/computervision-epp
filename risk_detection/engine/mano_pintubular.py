import numpy as np
import cv2
from shapely.geometry import Polygon, Point
from .base_scene import BaseScene
from utils.geometry_utils import has_all_classes, boxes_to_polys_by_name
from utils.pose_utils import iter_keypoints
from utils.visualization import draw_polygon, draw_line, put_text

class mano_pintubular(BaseScene):
    name = "mano_pintubular"

    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _instant_condition(self, det_obj):
        """
        Calcula la distancia entre:
        - Stickout: Esquina Superior Derecha (Top-Right)
        - PinTubular: Esquina Inferior Izquierda (Bottom-Left)
        """
        req = ["stickout", "pintubular"]
        if not has_all_classes(det_obj, req):
            # Retornamos None en los puntos si no hay objetos
            return False, None, None, 0.0
        
        polys = boxes_to_polys_by_name(det_obj, req)
        stick = polys["stickout"]
        pin = polys["pintubular"]

        # Stickout bounds (minx, miny, maxx, maxy)
        # Esquina Superior Derecha -> X máximo, Y mínimo
        s_minx, s_miny, s_maxx, s_maxy = stick.bounds
        p1 = (s_maxx, s_miny)

        # PinTubular bounds
        # Esquina Inferior Izquierda -> X mínimo, Y máximo
        p_minx, p_miny, p_maxx, p_maxy = pin.bounds
        p2 = (p_minx, p_maxy)

        # Calcular distancia
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        
        # La escena se activa si la distancia es menor al umbral
        is_close = dist < self.cfg.MANO_PIN_DIST_MIN

        return is_close, p1, p2, dist

    def _risk_hand_in_zone(self, res_pose, det_obj):
        """
        Genera el rectángulo de riesgo dentro del PinTubular y detecta manos.
        """
        req = ["pintubular"]
        # Nota: Si llegamos aquí, ya sabemos que existe pintubular por la condición de escena,
        # pero validamos por seguridad.
        if not has_all_classes(det_obj, req):
            return False, None

        polys = boxes_to_polys_by_name(det_obj, req)
        pin = polys["pintubular"]
        
        # Obtener dimensiones del PinTubular
        p_minx, p_miny, p_maxx, p_maxy = pin.bounds
        h_total = p_maxy - p_miny
        
        # Calcular altura del rectángulo de riesgo
        # Ejemplo: si RATIO es 0.4, el rectángulo ocupa el 40% inferior
        h_risk = h_total * self.cfg.MANO_PIN_RISK_HEIGHT_RATIO

        # Definir coordenadas del rectángulo de riesgo
        # Lado inferior, izquierdo y derecho coinciden con el PinTubular
        # Solo varía el lado superior (Y mínimo del rectángulo)
        r_minx = p_minx
        r_maxx = p_maxx
        r_maxy = p_maxy
        r_miny = p_maxy - h_risk 

        poly_points = np.array([
            [r_minx, r_miny], # Top-Left
            [r_maxx, r_miny], # Top-Right
            [r_maxx, r_maxy], # Bottom-Right
            [r_minx, r_maxy]  # Bottom-Left
        ], dtype=np.int32)
        
        poly_geom = Polygon(poly_points)
        risk_detected = False

        # Verificar manos dentro del polígono
        if res_pose:
            for kp in iter_keypoints(res_pose):
                for idx in self.cfg.HAND_IDXS:
                    if idx < len(kp):
                        x, y = kp[idx][:2]
                        conf = kp[idx][2] if len(kp[idx]) > 2 else 1.0
                        if conf > 0.5: # Filtrar detecciones de baja confianza
                            if Point(x, y).within(poly_geom):
                                risk_detected = True
                                break
                if risk_detected: break

        return risk_detected, poly_points

    def evaluate(self, det_obj, res_pose, frame):
        # 1. Evaluar Escena
        scene_active, p1, p2, dist_val = self._instant_condition(det_obj)
        self.increment_scene_active_pos_neg(scene_active)

        if self.scene_active_pos >= self.cfg.MANO_PIN_SCENE_ON:
            self.activate_scene()
        elif self.scene_active_neg >= self.cfg.MANO_PIN_SCENE_OFF:
            self.deactivate_scene()
        
        risk = False
        poly_np = None

        # 2. Evaluar Riesgo (Solo si la escena está activa)
        if self.scene_active:
            risk, poly_np = self._risk_hand_in_zone(res_pose, det_obj)
            self.increment_risk_active_pos_neg(risk)

            if self.risk_active_pos >= self.cfg.MANO_PIN_RISK_ON:
                self.activate_risk()
            elif self.risk_active_neg >= self.cfg.MANO_PIN_RISK_OFF:
                self.deactivate_risk()

        # ---------------- VISUALIZACIÓN ----------------
        if frame is not None and self.cfg.VISUALIZE:
            
            # A. Visualizar Línea de Distancia (Siempre que detecte los objetos)
            if p1 is not None and p2 is not None:
                # Color de la línea:
                # Rojo si la escena está activa (distancia < umbral)
                # Verde si es segura (distancia > umbral)
                line_color = (0, 0, 255) if self.scene_active else (0, 255, 0)
                
                # Usamos cv2.line directo para controlar color exacto o usamos draw_line con 'active'
                # Aquí uso cv2 para forzar los colores según lógica de escena
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), line_color, 2)
                
                # Opcional: Escribir la distancia en pantalla cerca de la línea
                mx, my = int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)
                cv2.putText(frame, f"{int(dist_val)}px", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

            # B. Visualizar Rectángulo de Riesgo (Solo si escena activa)
            if self.scene_active and poly_np is not None:
                # El rectángulo se dibuja:
                # Rojo (active=True) si hay riesgo (mano dentro)
                # Verde (active=False) si es zona segura (sin mano)
                draw_polygon(frame, poly_np, active=self.risk_active)
        # -----------------------------------------------

        self.log_state()
        return self.make_result(self.scene_active, self.risk_active)
