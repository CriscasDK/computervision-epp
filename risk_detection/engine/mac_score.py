import cv2
import numpy as np
from engine.base_scene import BaseScene
from engine.models import Detection
from utils.body_angles import calculate_trunk_angle

class MACEvaluator(BaseScene):
    """
    Evaluador de MAC (Manual Handling Assessment Charts) Score para Lifting.
    Se asumen valores fijos: A=4, E=0, F=0, G=0, H=0.
    Este módulo calcula las componentes B, C y D usando geometría simple (2D) 
    sobre los keypoints de pose.
    """
    name = "mac_score"
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def evaluate(self, fused_entities, frame, detection_data: Detection, **kwargs):
        if not detection_data.people or not self.cfg.MAC_ENABLED:
            return detection_data
            
        kps_map = {ent["track_id"]: ent.get("keypoints") for ent in fused_entities}

        for person in detection_data.people:
            kps = kps_map.get(person.track_id)
            if kps is None or len(kps) < 17:
                continue
                
            # Solo calcular el MAC score si el detector de escena confirmó un evento de levantamiento
            # [COMENTADO TEMPORALMENTE PARA DEBUG]
            # if not person.mac_lifting_detected:
            #     person.mac_total = 0
            #     continue
                
            # Calcular pasos MAC dependientes de posturas
            score_b, trunk_flex, norm_dist, w_avg_x, lb_x, lb_y, is_lateral = self._calculate_b(kps, person)
            score_c, wrists_y, knees_y, elbows_y = self._calculate_c(kps)
            score_d, lateral_bend_deg, ratio = self._calculate_d(kps)
            
            # Constantes Asumidas según lo indicado
            score_a = self.cfg.MAC_CONSTANT_A
            score_e = self.cfg.MAC_CONSTANT_E
            score_f = self.cfg.MAC_CONSTANT_F
            score_g = self.cfg.MAC_CONSTANT_G
            score_h = self.cfg.MAC_CONSTANT_H
            
            mac_total = score_a + score_b + score_c + score_d + score_e + score_f + score_g + score_h
            
            person.mac_score_b = score_b
            person.mac_score_c = score_c
            person.mac_score_d = score_d
            person.mac_total = mac_total

            person.bad_pose["bad_pose_mac"] = self.update_hysteresis(
                track_id=person.track_id, 
                condition=mac_total >= self.cfg.MAC_HIGH_THRESHOLD, 
                pos_threshold=30,
                neg_threshold=50
            )
            
            # Si visualize está activo y tenemos un frame válido, pintamos la lógica (Debug visual)
            if self.cfg.VISUALIZE and frame is not None:
                self._draw_debug(frame, kps, person.track_id, 
                                 score_b, trunk_flex, norm_dist, w_avg_x, lb_x, lb_y, is_lateral,
                                 score_c, wrists_y, knees_y, elbows_y, 
                                 score_d, lateral_bend_deg, ratio)

        return detection_data

    def _calculate_b(self, kps, person):
        """
        B: Hand distance from the lower back (Horizontal distance).
        Proyecta la ubicación de las palmas o usa las proyecciones globales existentes (19, 20).
        Reutiliza la orientación lateral calculada por el REBA score para
        decidir si la distancia medida horizontalmente es confiable.
        """
        trunk_flex = calculate_trunk_angle(kps)
        
        # 1. Torso points
        h_mid_x = (kps[11][0] + kps[12][0]) / 2
        h_mid_y = (kps[11][1] + kps[12][1]) / 2
        s_mid_x = (kps[5][0] + kps[6][0]) / 2
        s_mid_y = (kps[5][1] + kps[6][1]) / 2
        
        torso_len = np.sqrt((s_mid_x - h_mid_x)**2 + (s_mid_y - h_mid_y)**2)
        if torso_len == 0: torso_len = 1e-5
        
        # 2. Restricción de Lateralidad (Reutilizando la de REBA, >= 0.50 como umbral)
        is_lateral = person.reba_score_conf >= 0.50
        
        # 3. Ubicación de las palmas usando proyecciones globales provistas al vector de kps
        # kps[19], kps[20] son mano izquierda y mano derecha proyectadas
        w_avg_x = h_mid_x
        if len(kps) > 20: 
            palms_xs = []
            if kps[19][2] > 0.3: palms_xs.append(kps[19][0])
            if kps[20][2] > 0.3: palms_xs.append(kps[20][0])
            if palms_xs: w_avg_x = sum(palms_xs)/len(palms_xs)
        else:
            # Fallback a muñecas (9, 10)
            wrists_xs = []
            if kps[9][2] > 0.3: wrists_xs.append(kps[9][0])
            if kps[10][2] > 0.3: wrists_xs.append(kps[10][0])
            if wrists_xs: w_avg_x = sum(wrists_xs)/len(wrists_xs)
        
        # MAC Tool: "Observe the *horizontal* distance" (x-axis in lateral view)
        horiz_dist_px = abs(w_avg_x - h_mid_x)
        
        norm_dist = horiz_dist_px / torso_len
        
        # Si no es lateral, la información visual es defectuosa. Devolvemos Score 0 
        # para no levantar falsos eventos de riesgo por mala visibilidad de palancas.
        effective_dist = norm_dist if is_lateral else 0.0
        
        if not is_lateral:
            score_b = 0
        elif trunk_flex >= 60 or effective_dist >= 0.7:
            score_b = 6
        elif trunk_flex >= 30 or effective_dist >= 0.3:
            score_b = 3
        else:
            score_b = 0
            
        return score_b, trunk_flex, effective_dist, w_avg_x, h_mid_x, h_mid_y, is_lateral

    def _calculate_c(self, kps):
        """
        C: Vertical lift zones.
        Comparamos posicion vertical (eje Y) de las manos respecto a rodillas/codos.
        El Origen (0) de OpenCV en Y es Arriba, por ende mayor Y significa estar mas abajo.
        """
        def get_avg_y(idx1, idx2):
            y1, c1 = kps[idx1][1], kps[idx1][2]
            y2, c2 = kps[idx2][1], kps[idx2][2]
            ys = []
            if c1 > 0.3: ys.append(y1)
            if c2 > 0.3: ys.append(y2)
            return sum(ys)/len(ys) if ys else None
            
        # Priorizar palmas proyectadas (indices 19, 20)
        if len(kps) > 20 and (kps[19][2] > 0.3 or kps[20][2] > 0.3):
            wrists_y = get_avg_y(19, 20)
        else:
            wrists_y = get_avg_y(9, 10)
            
        elbows_y = get_avg_y(7, 8)
        knees_y = get_avg_y(13, 14)
        
        # Priorizar pies proyectados (indices 17, 18)
        if len(kps) > 18 and (kps[17][2] > 0.3 or kps[18][2] > 0.3):
            ankles_y = get_avg_y(17, 18)
        else:
            ankles_y = get_avg_y(15, 16)
            
        nose_y = kps[0][1] if kps[0][2] > 0.3 else None
        
        if not wrists_y or not elbows_y or not knees_y:
            return 0, wrists_y, knees_y, elbows_y
            
        floor_level_y = ankles_y if ankles_y else (knees_y + 100)
        head_level_y = nose_y if nose_y else (elbows_y - 100)
        
        score_c = 0
        # Red checks (Suelo o Cabeza/Arriba de ella)
        if wrists_y >= floor_level_y - 30: # 30px param margen suelo
            score_c = 3
        elif wrists_y <= head_level_y + 20: 
            score_c = 3
        # Amber checks (Debajo rodilla O arriba codo)
        elif wrists_y > knees_y: 
            score_c = 1
        elif wrists_y < elbows_y: 
            score_c = 1
        # Green check (Entre codo y rodilla)
        else:
            score_c = 0
            
        return score_c, wrists_y, knees_y, elbows_y

    def _calculate_d(self, kps):
        """
        D: Torso twisting and sideways bending.
        Torso twisted AND bent sideways (Red/2).
        Torso twisted OR bent sideways (Amber/1).
        """
        s_mid_x = (kps[5][0] + kps[6][0]) / 2
        s_mid_y = (kps[5][1] + kps[6][1]) / 2
        h_mid_x = (kps[11][0] + kps[12][0]) / 2
        h_mid_y = (kps[11][1] + kps[12][1]) / 2
        
        dx = s_mid_x - h_mid_x
        dy = h_mid_y - s_mid_y
        if dy == 0: dy = 1e-5
        
        lateral_bend_deg = abs(np.degrees(np.arctan(dx / dy)))
        
        shoulder_w = abs(kps[6][0] - kps[5][0])
        hip_w = abs(kps[12][0] - kps[11][0])
        ratio = shoulder_w / (hip_w + 1e-5)
        
        is_twisted = ratio < 0.75  # Hombros se empujan por torsión
        is_bent = lateral_bend_deg > 30  # Inclinación diagonal del torso
        
        if is_twisted and is_bent:
            score_d = 2 # Red
        elif is_twisted or is_bent:
            score_d = 1 # Amber
        else:
            score_d = 0 # Green
            
        return score_d, lateral_bend_deg, ratio

    def _draw_debug(self, frame, kps, track_id, 
                    score_b, trunk_flex, norm_dist, w_avg_x, lb_x, lb_y, is_lateral,
                    score_c, wrists_y, knees_y, elbows_y, 
                    score_d, lateral_bend_deg, ratio):
        """Dibuja guías explicativas sobre el frame cuando VISUALIZE=True."""
        try:
            # Obtener origen usando la cabeza o el cuello
            org_x = int(kps[0][0]) if kps[0][2] > 0.3 else int((kps[5][0] + kps[6][0])/2)
            org_y = int(kps[0][1]) if kps[0][2] > 0.3 else int((kps[5][1] + kps[6][1])/2)
            
            # Textos MAC
            lat_txt = "LATERAL OK" if is_lateral else "NO LATERAL"
            cv2.putText(frame, f"ID {track_id} MAC B:{score_b} (Dist:{norm_dist:.2f} | Flex:{trunk_flex:.1f}) [{lat_txt}]", 
                        (org_x - 50, org_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame, f"ID {track_id} MAC C:{score_c}", 
                        (org_x - 50, org_y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame, f"ID {track_id} MAC D:{score_d} (Bend:{lateral_bend_deg:.1f})", 
                        (org_x - 50, org_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # --- Líneas C (Vertical Zones) ---
            if wrists_y and knees_y and elbows_y:
                # Muñecas
                wy = int(wrists_y)
                cv2.line(frame, (org_x - 40, wy), (org_x + 40, wy), (255, 0, 0), 1) 
                cv2.putText(frame, "Wrist", (org_x + 45, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                
                # Codos y Rodillas (Los "límites")
                ey = int(elbows_y)
                ky = int(knees_y)
                cv2.line(frame, (org_x - 60, ey), (org_x + 60, ey), (0, 255, 0), 1) 
                cv2.line(frame, (org_x - 60, ky), (org_x + 60, ky), (0, 255, 0), 1) 
                
            # --- Líneas B y D (Postura de Tronco y Flexión Lateral) ---
            # Línea de la espalda (Cuello a Pelvis)
            s_mid_x = int((kps[5][0] + kps[6][0]) / 2)
            s_mid_y = int((kps[5][1] + kps[6][1]) / 2)
            h_mid_x = int((kps[11][0] + kps[12][0]) / 2)
            h_mid_y = int((kps[11][1] + kps[12][1]) / 2)
            
            # Espina real (Amarilla para torsión/bend)
            cv2.line(frame, (s_mid_x, s_mid_y), (h_mid_x, h_mid_y), (0, 255, 255), 2)
            # Vertical de referencia desde la pelvis
            cv2.line(frame, (h_mid_x, h_mid_y), (h_mid_x, h_mid_y - 100), (255, 255, 255), 1, cv2.LINE_AA)
            
            # Distancia horizontal de manos a lower back (MAC B)
            if w_avg_x and lb_x and lb_y:
                cv2.line(frame, (int(lb_x), int(lb_y)), (int(w_avg_x), int(lb_y)), (200, 100, 255), 2)

        except Exception as e:
            # Tolerancia de errores al pintar
            pass
