import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from .base_scene import BaseScene
from .models import Person, Detection
from utils.visualization import draw_text_custom

class WorkZoneMonitor(BaseScene):
    name = "zona_trabajo_principal"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.polygon = Polygon(cfg.WORK_ZONE_POLY)
        self.poly_np = cfg.WORK_ZONE_POLY
        
        # Tracking de estado por track_id usando histéresis
        # Cada track_id tiene: {
        #   'in_zone': bool,           # Estado confirmado actual
        #   'frames_inside': int,      # Frames consecutivos dentro
        #   'frames_outside': int      # Frames consecutivos fuera
        # }
        self.zone_state = {}


        # Tracking de estado por track_id usando histéresis
        # Cada track_id tiene: {
        #   'in_zone': bool,           # Estado confirmado actual
        #   'frames_inside': int,      # Frames consecutivos dentro
        #   'frames_outside': int      # Frames consecutivos fuera
        # }
        self.zone_state = {}

    def _get_clamped_bottom_center(self, box, img_h, img_w):
        """
        Calcula el punto de referencia (pies) asegurando que esté dentro de la imagen.
        Si el box se sale, lo limitamos (clamp) al borde máximo de la resolución.
        """
        x1, y1, x2, y2 = box
        
        # 1. Clamping (Limitar coordenadas al tamaño de imagen)
        # Nos aseguramos que x2 y y2 no sean mayores que el ancho/alto real
        # Restamos 1 pixel para estar seguros de estar 'dentro' del indice del array
        x2_clamped = min(x2, img_w - 1)
        y2_clamped = min(y2, img_h - 1)
        
        # El centro X también debe estar limitado
        center_x = (x1 + x2_clamped) / 2
        center_x = max(0, min(center_x, img_w - 1))

        return (center_x, y2_clamped)

    def _is_inside_zone(self, box, img_h, img_w):
        """
        Verifica ubicación usando el punto clampeado.
        """
        px, py = self._get_clamped_bottom_center(box, img_h, img_w)
        point = Point(px, py)
        
        # Verificación estándar de geometría
        # Si el punto está justo en el borde, shapely puede devolver False en .contains
        # Usamos .distance(point) < 1.0 para ser tolerantes con el borde exacto
        is_inside = self.polygon.contains(point) or self.polygon.distance(point) < 1.0
        
        return is_inside

    def evaluate(self, fused_entities, frame, **kwargs):
        """
        Evalúa qué personas están dentro de la zona de trabajo.
        
        Implementa confirmación temporal con histéresis usando contadores ascendentes:
        - ENTRADA: Contador ascendente de frames DENTRO hasta alcanzar umbral
        - SALIDA: Contador ascendente de frames FUERA hasta alcanzar umbral
        
        Esto permite umbrales independientes y lógica más clara.
        """
        current_detection = Detection()

        current_ids_in_zone = []
        current_ids_bad_pose = []
        color_zone = (0, 255, 0)  # Verde

        # Obtener dimensiones reales para el clamping
        img_h, img_w = frame.shape[:2] if frame is not None else (720, 1280)

        for entity in fused_entities:
            track_id = entity["track_id"]
            box = entity["box"]
            kps = entity["keypoints"]
            
            # 1. Verificar si está físicamente dentro de la zona AHORA
            inside_now = self._is_inside_zone(box, img_h, img_w)
            
            # 2. Inicializar estado si es nuevo track_id
            if track_id not in self.zone_state:
                self.zone_state[track_id] = {
                    'in_zone': False,
                    'frames_inside': 0,
                    'frames_outside': 0
                }
            
            state = self.zone_state[track_id]
            
            # 3. Actualizar contadores con histéresis (contadores ascendentes)
            if inside_now:
                # Está dentro: incrementar contador de entrada, resetear salida
                state['frames_inside'] += 1
                state['frames_outside'] = 0
            else:
                # Está fuera: incrementar contador de salida, resetear entrada
                state['frames_outside'] += 1
                state['frames_inside'] = 0
            
            # 4. Determinar estado confirmado con umbrales independientes
            entry_threshold = self.cfg.ZONE_ENTRY_THRESHOLD
            exit_threshold = self.cfg.ZONE_EXIT_THRESHOLD
            
            # Lógica de histéresis:
            # - Si NO está en zona y acumula N frames dentro → ENTRA
            # - Si SÍ está en zona y acumula N frames fuera → SALE
            if not state['in_zone']:
                # Estado actual: FUERA
                # Condición de entrada: frames_inside >= entry_threshold
                if state['frames_inside'] >= entry_threshold:
                    state['in_zone'] = True
            else:
                # Estado actual: DENTRO
                # Condición de salida: frames_outside >= exit_threshold
                if state['frames_outside'] >= exit_threshold:
                    state['in_zone'] = False
            
            # 5. Agregar a lista de IDs en zona si está confirmado
            is_confirmed_in_zone = state['in_zone']
            
            if is_confirmed_in_zone:
                current_ids_in_zone.append(track_id)
                color_zone = (0, 165, 255)  # Naranja
                
                # Visualizar punto de anclaje (Debug)
                if frame is not None and self.cfg.VISUALIZE:
                    bx, by = self._get_clamped_bottom_center(box, img_h, img_w)
                    cv2.circle(frame, (int(bx), int(by)), 5, (255, 255, 0), -1)
            
            # 7. Crear Person para TODOS los track_id detectados
            person_obj = Person(
                track_id=track_id,
                in_zone=is_confirmed_in_zone,
                epp=[]
            )
            current_detection.people.append(person_obj)

        # Visualizar polígono de zona
        if self.cfg.VISUALIZE and frame is not None:
            cv2.polylines(frame, [self.poly_np], isClosed=True, color=color_zone, thickness=2)

        return self.make_result(
            in_zone_ids=current_ids_in_zone, 
            risk_ids=current_ids_bad_pose, 
            count=len(current_ids_in_zone)
        ), current_detection

