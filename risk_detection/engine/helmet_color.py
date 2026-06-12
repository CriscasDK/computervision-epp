# risk_detection/utils/helmet_color.py
import cv2
import pytz
import numpy as np
from datetime import datetime
from collections import deque, Counter
from .models import Person, Detection
from .base_scene import BaseScene
from utils.geometry_utils import clamp_box_xyxy, inner_box, calculate_iou_xyxy

class HelmetColorTracker(BaseScene):
    """
    Asocia cascos a track_id de personas y estabiliza el color por historial.
    """
    name = "monitor_helmet_color"
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.person_class_id = int(self.cfg.PERSON_CLASS_ID)
        self.helmet_class_names = set([str(x).lower() for x in self.cfg.HELMET_CLASS_NAMES])

        self.history_len = int(self.cfg.HELMET_COLOR_HISTORY_LEN)
        self.min_vote = int(self.cfg.HELMET_COLOR_MIN_VOTE)
        self.max_age_frames = int(self.cfg.HELMET_COLOR_MAX_AGE_FRAMES)

        self.score_threshold_assign = float(self.cfg.HELMET_ASSIGN_SCORE_THRESHOLD)
        self.top_ratio = float(self.cfg.HELMET_TOP_RATIO)

        self.hsv_profiles = self.cfg.HELMET_HSV_PROFILES or {}
        self.bogota = pytz.timezone("America/Bogota")

        self._color_hist = {}  # track_id -> deque[(color, score)]
        self._last_seen = {}   # track_id -> frame_idx
    
    def classify_helmet_color_bgr(self, helmet_crop_bgr, hsv_profile):
        """
        Clasifica SOLO {white, yellow, orange} usando un perfil HSV.

        hsv_profile ejemplo:
        {
            "white": {"s_max": 70, "v_min": 205},
            "orange": {"h_min": 11, "h_max": 23, "s_min": 90, "v_min": 90},
            "yellow": {"h_min": 24, "h_max": 38, "s_min": 80, "v_min": 110},
        }

        Retorna:
        color_name: str
        score: float  (proporción de pixeles válidos que soportan el color ganador)
        """
        if helmet_crop_bgr is None or helmet_crop_bgr.size == 0:
            return "unknown", 0.0

        if helmet_crop_bgr.shape[0] < 6 or helmet_crop_bgr.shape[1] < 6:
            return "unknown", 0.0

        crop = cv2.GaussianBlur(helmet_crop_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        h = hsv[:, :, 0].astype(np.uint8)  # 0..179
        s = hsv[:, :, 1].astype(np.uint8)  # 0..255
        v = hsv[:, :, 2].astype(np.uint8)  # 0..255

        # Píxeles válidos mínimos (evita negro absoluto / ruido)
        valid = (v > 20)
        total_valid = int(valid.sum())
        if total_valid < 80:
            return "unknown", 0.0

        # WHITE: baja saturación + alto brillo
        wcfg = hsv_profile.get("white", {})
        white = valid & (s <= int(wcfg.get("s_max", 70))) & (v >= int(wcfg.get("v_min", 205)))

        # ORANGE: H en rango + S/V mínimos
        ocfg = hsv_profile.get("orange", {})
        orange = (
            valid
            & (h >= int(ocfg.get("h_min", 11))) & (h <= int(ocfg.get("h_max", 23)))
            & (s >= int(ocfg.get("s_min", 90)))
            & (v >= int(ocfg.get("v_min", 90)))
        )

        # YELLOW: H en rango + S/V mínimos
        ycfg = hsv_profile.get("yellow", {})
        yellow = (
            valid
            & (h >= int(ycfg.get("h_min", 24))) & (h <= int(ycfg.get("h_max", 38)))
            & (s >= int(ycfg.get("s_min", 80)))
            & (v >= int(ycfg.get("v_min", 110)))
        )

        counts = {
            "white": int(white.sum()),
            "orange": int(orange.sum()),
            "yellow": int(yellow.sum()),
        }

        best_color = max(counts.keys(), key=lambda k: counts[k])
        best_count = counts[best_color]
        score = best_count / float(total_valid + 1e-6)

        # Filtro mínimo: evita que “gane” por unos pocos píxeles
        if best_count < 120 and score < 0.20:
            return "unknown", float(score)

        return best_color, float(score)
    
    def _define_profile_day(self):
        "Define el perfil del dia a partir de la hora actual"
        
        profile_mode = getattr(self.cfg, "HELMET_COLOR_PROFILE", "auto")

        if profile_mode in ("day", "night"):
            profile = profile_mode
        else:
            now = datetime.now(self.bogota)
            day_start = int(getattr(self.cfg, "HELMETDAYSTART", 6))
            day_end = int(getattr(self.cfg, "HELMETDAYEND", 17))
            profile = "day" if (day_start <= now.hour < day_end) else "night"

        return profile

    def _push_color(self, track_id, color, score):
        dq = self._color_hist.get(track_id)
        if dq is None:
            dq = deque(maxlen=self.history_len)
            self._color_hist[track_id] = dq
        dq.append((color, float(score)))

    def _stable_color(self, track_id):
        dq = self._color_hist.get(track_id, None)
        if not dq:
            return "unknown", 0.0

        colors = [c for (c, _) in dq if c != "unknown"]
        if not colors:
            avg = float(np.mean([s for (_, s) in dq])) if dq else 0.0
            return "unknown", avg

        counts = Counter(colors)
        top_color, top_votes = counts.most_common(1)[0]

        if top_votes < self.min_vote:
            last_color, last_score = dq[-1]
            return last_color, float(last_score)

        scores = [s for (c, s) in dq if c == top_color]
        return top_color, float(np.mean(scores)) if scores else 0.0

    def cleanup_old_tracks(self, frame_idx):
        to_del = []
        for tid, last in self._last_seen.items():
            if (frame_idx - last) > self.max_age_frames:
                to_del.append(tid)
        for tid in to_del:
            self._last_seen.pop(tid, None)
            self._color_hist.pop(tid, None)

    def evaluate(self, frame_bgr, fused_entities, detection_data: Detection, frame_idx):
        """
        Lee helmet_box de la estructura fusionada
        
        Args:
            frame_bgr: Frame en BGR para clasificación de color
            fused_entities: Lista de entidades fusionadas con EPP incluido
            detection_data: Objeto Detection con lista de Person
            frame_idx: Índice del frame actual
        
        Returns:
            (Detection, dict): Objeto actualizado y estado de helmet por track_id
        """
        if frame_bgr is None or not detection_data.people:
            return detection_data, {}

        h_img, w_img = frame_bgr.shape[:2]
        profile = self._define_profile_day()
        hsv_profile = self.hsv_profiles.get(profile, self.hsv_profiles.get("day", {}))

        # Crear mapa track_id -> helmet_boxes desde fusión
        helmet_map = {
            entity["track_id"]: entity["epp"]["helmet"]
            for entity in fused_entities
            if entity["epp"]["helmet"]  # Solo si tiene al menos un casco
        }
        
        # Mapeamos los objetos Person por track_id
        persons_map = {p.track_id: p for p in detection_data.people}

        out = {}
        for person in detection_data.people:
            tid = person.track_id
            self._last_seen[tid] = int(frame_idx)

            # 1. Intentar clasificar color si hay un casco asociado en la fusión
            if tid in helmet_map and helmet_map[tid]:
                # Tomar el primer casco (o el de mayor área si hay múltiples)
                helmet_boxes = helmet_map[tid]
                
                # Seleccionar casco de mayor área
                helmet_box = max(
                    helmet_boxes, 
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
                )
                
                # Clasificar color
                x1, y1, x2, y2 = clamp_box_xyxy(helmet_box, w_img, h_img)
                x1i, y1i, x2i, y2i = inner_box(x1, y1, x2, y2, margin=0.12)
                x1i, y1i, x2i, y2i = clamp_box_xyxy((x1i, y1i, x2i, y2i), w_img, h_img)

                crop = frame_bgr[y1i:y2i, x1i:x2i]
                color, score = self.classify_helmet_color_bgr(crop, hsv_profile)
                self._push_color(tid, color, score)

            # 2. Obtener el color estable (con memoria)
            stable_color, stable_score = self._stable_color(tid)

            # 3. ACTUALIZAR EL OBJETO PERSON
            if tid in persons_map:
                persons_map[tid].helmet_color = stable_color

            out[int(tid)] = {
                "color": stable_color,
                "score": float(stable_score),
                "helmet_box": helmet_map.get(tid, [None])[0] if tid in helmet_map else None,
                "person_box": persons_map.get(tid, None),
                "profile": profile,
            }

        self.cleanup_old_tracks(int(frame_idx))
        return detection_data, out