# main_realtime.py

import time
import cv2
import signal
import sys
import logging
import pytz
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from supervision import Detections, BoxAnnotator, LabelAnnotator, ColorLookup, ByteTrack
import shutil
import os
from config import Config
from in_out.work_zone_logger import WorkZoneLogger
from in_out.video_clip_writer import VideoClipWriter
from in_out.performance import PerformanceMonitor

# from in_out.helmet_color_logger import HelmetColorLogger
# from utils.helmet_color import HelmetColorTracker

from utils.keypoints_projections import *
from utils.visualization import draw_hud
from risk_engine import RiskEngine
from utils.fusion_utils import fuse_complete_detection


try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


# =============================
# Configuración del Logger
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================
# Clase principal del sistema
# =============================
class RiskDetectionApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.keep_running = True
        self.bogota = pytz.timezone("America/Bogota")

        self.previous_risk_states = {}
        self.video_writer = None
        self.cap = None

        self.engine = RiskEngine(cfg)
        self.clip_writer = None
        self.work_zone_logger = WorkZoneLogger(cfg)  # Nuevo logger event-based

        self.pre_roll_buffer = deque()
        self.fps_smoothed = None
        self.monitor = PerformanceMonitor() if cfg.MONITOR_PERFORMANCE else None

        # Inicializamos ByteTrack para tener control total
        self.tracker = ByteTrack()

        # Contador de frames para logs y estabilidad
        self.frame_idx = 0

        # Estado por persona (track_id -> {color,score,...})
        self.helmet_state = {}
        # self.helmet_tracker = HelmetColorTracker(cfg)

    # -------------------------
    # Inicialización
    # -------------------------
    def setup(self):
        logger.info("Iniciando Sistema de Vision (EPP + Pose + Zonas)...")
        self._load_models()
        self._setup_db_logger()
        # self._setup_helmet_logger()
        self._setup_clip_writer()
        self._setup_video_capture()
        logger.info("Sistema completamente inicializado.")

    def _load_models(self):
        start = time.time()

        # Modelo de Objetos (EPP: Helmet, Boots, Person, etc.) - Usado para Tracking
        self.model_obj = YOLO(self.cfg.MODEL_OBJ)

        # Modelo de Pose (Esqueleto) - Usado para Ergonomía
        self.model_pose = YOLO(self.cfg.MODEL_POSE)

        logger.info(f"Modelos cargados en {time.time() - start:.2f}s")

    def _setup_db_logger(self):
        # Iniciar WorkZoneLogger (event-based)
        db_path = self.work_zone_logger.start_logger(output_dir=self.cfg.LOG_DIR)
        logger.info(f"Logs de WorkZone (event-based) -> {db_path}")
        
        # db_logger antiguo
        # db_path = self.db_logger.start_logger(output_dir=self.cfg.LOG_DIR)
        # logger.info(f"Logs de BBDD -> {db_path}")

    def _setup_helmet_logger(self):
        enabled = self.cfg.HELMET_COLOR_ENABLED
        if not enabled:
            logger.info("Helmet color logger deshabilitado por configuracion.")
            return
        # path = self.helmet_logger.start_logger(output_dir=self.cfg.LOG_DIR)
        # logger.info(f"Logs de color de casco -> {path}")

    def _setup_clip_writer(self):
        if self.cfg.CLIP_ENABLED:
            self.clip_writer = VideoClipWriter(self.cfg)
            self.clip_writer.start_controller()

    def _setup_video_capture(self):
        logger.info(f"Conectando a video fuente: {self.cfg.VIDEO_SOURCE}")
        self.cap = cv2.VideoCapture(self.cfg.VIDEO_SOURCE)

        self.cap.set(cv2.CAP_PROP_POS_MSEC, 30000)

        if not self.cap.isOpened():
            logger.error(f"No se pudo abrir la fuente: {self.cfg.VIDEO_SOURCE}")
            raise RuntimeError(f"No se pudo abrir fuente: {self.cfg.VIDEO_SOURCE}")

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        pre_roll_size = int(fps * self.cfg.CLIP_PREROLL_SEC)
        self.pre_roll_buffer = deque(maxlen=pre_roll_size)
        logger.info(f"FPS: {fps:.2f} | Buffer pre-roll: {pre_roll_size} frames")

        if self.cfg.WRITE_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(self.cfg.OUTPUT_PATH, fourcc, fps, self.cfg.RESIZE)
            logger.info(f"Grabacion activada -> {self.cfg.OUTPUT_PATH}")

    # -------------------------
    # Horarios y señales
    # -------------------------
    def is_within_schedule(self):
        now = datetime.now(self.bogota)
        if now.hour in self.cfg.HOURS_PAUSE and now.minute < self.cfg.MINUTES_PAUSE:
            return False
        return True

    def graceful_shutdown(self, signum, frame):
        logger.warning("Senal de apagado recibida. Deteniendo ejecucion...")
        self.keep_running = False

    def _merge_detections(self, dets_person, dets_other):
        """
        Fusiona detecciones de personas (con tracking) y objetos EPP (sin tracking).
        
        Usa early returns para simplificar la lógica y mejorar legibilidad.
        
        Args:
            dets_person: Detecciones de personas con tracker_id asignado
            dets_other: Detecciones de objetos EPP (cascos, botas, etc.)
        
        Returns:
            Detections: Objeto fusionado con todas las detecciones
        """
        # Caso 1: Sin detecciones de ningún tipo
        if len(dets_person) == 0 and len(dets_other) == 0:
            return Detections.empty()
        
        # Caso 2: Solo objetos EPP (sin personas)
        if len(dets_person) == 0:
            dets_other.tracker_id = np.full(len(dets_other), None)
            return dets_other
        
        # Caso 3: Solo personas (sin objetos EPP)
        if len(dets_other) == 0:
            return dets_person
        
        # Caso 4: Ambos tipos presentes -> Merge
        dets_other.tracker_id = np.full(len(dets_other), None)
        return Detections.merge([dets_person, dets_other])

    # -------------------------
    # Bucle principal
    # -------------------------
    def run(self):
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)

        fps_timer = time.time()

        while self.keep_running:
            if not self.is_within_schedule():
                logger.info(f"Pausa por {self.cfg.MINUTES_PAUSE} minutos...")
                try:
                    self.cleanup()
                except Exception as e:
                    logger.error(f"Error ejecutando cleanup dentro de is_within_schedule(): {e}")
                self.move_database()
                time.sleep(60 * self.cfg.MINUTES_PAUSE)
                return

            ok, frame = self.cap.read()
            if not ok:
                logger.warning("No se pudo leer frame.")
                break

            # frame = cv2.resize(frame, self.cfg.RESIZE, interpolation=cv2.INTER_AREA)
            
            # --- PROCESAMIENTO ---
            self._process_frame(frame)

            # --- FPS ---
            now = time.time()
            inst_fps = 1.0 / max(now - fps_timer, 1e-6)
            fps_timer = now
            self.fps_smoothed = inst_fps if self.fps_smoothed is None else (self.fps_smoothed * 0.9 + inst_fps * 0.1)

            if self.monitor:
                self.monitor.update(self.fps_smoothed)

            if self.cfg.VISUALIZE:
                cv2.imshow("RiskEngine (EPP + Pose)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Cierre manual (tecla 'q')")
                    break

            self.frame_idx += 1

        self.cleanup()

    # -------------------------
    # Procesamiento por frame
    # -------------------------
    def _process_frame(self, frame):
        # Inferencia: Obtener detecciones EPP y Poses RAW
        inference_outputs = self._run_inference(frame)
        detections_sv = inference_outputs["detections_sv"]

        # Fusión COMPLETA: Person + Pose + EPP en una sola pasada
        # Esto elimina cálculos duplicados de IoU en EPPMonitor y HelmetColorTracker
        fused_entities = fuse_complete_detection(
            sv_detections=detections_sv,
            pose_results=inference_outputs["raw_pose"],
            target_class_id=self.cfg.PERSON_CLASS_ID,
            iou_threshold_pose=self.cfg.IOU_FUSION,
            iou_threshold_epp=self.cfg.IOU_EPP,
        )

        # Engine: Lógica de Zona y Ergonomía
        # El engine modifica 'frame' in-place (dibujando polígonos/alertas) si VISUALIZE=True
        engine_results, self.helmet_state, final_detection = self.engine.process(
            fused_entities,
            frame if self.cfg.VISUALIZE else None,
            raw_detections_sv=inference_outputs["detections_sv"],
            frame_idx = self.frame_idx
            )
        
        # print(engine_results)

        # Manejo de eventos (I/O, Clips)
        self._handle_risks(final_detection)

        # Visualización de Detecciones EPP (Cajas, Etiquetas) + HUD + color casco
        self._visualize(
            frame, 
            detections_sv, 
            engine_results, 
            raw_pose=inference_outputs["raw_pose"],
            final_detection=final_detection,
            fused_entities=fused_entities 
        )
        # Guardado de Buffers
        if self.cfg.CLIP_ENABLED:
            self.pre_roll_buffer.append(frame)
            self.clip_writer.put_frame(frame)

        if self.video_writer:
            self.video_writer.write(frame)

    def _run_inference(self, frame):
        """
        Ejecuta ambos modelos.

        Retorna:
          - raw_obj: resultados crudos de YOLO EPP
          - raw_pose: resultados crudos de YOLO Pose
          - detections_sv: Detecciones formateadas para Supervision
        """
        res_obj = self.model_obj.predict(
            frame,
            device=self.cfg.DEVICE,
            conf=self.cfg.CONF_OBJ,
            verbose=False,
        )
        
        # Convertir a Supervision
        detections = Detections.from_ultralytics(res_obj[0])
        names = res_obj[0].names

        if detections.class_id is not None:
            detections.data["class_name"] = np.array([names[i] for i in detections.class_id.astype(int)])

        # Separar y trackear solo personas
        if detections.class_id is None or len(detections) == 0:
            detections_final = Detections.empty()
        else:
            # Separar personas de otros objetos EPP
            mask_person = (detections.class_id == self.cfg.PERSON_CLASS_ID)
            dets_person = detections[mask_person]
            dets_other = detections[~mask_person]

            # Trackear solo personas (si existen)
            if len(dets_person) > 0:
                dets_person = self.tracker.update_with_detections(dets_person)
            
            # Fusionar usando el método helper simplificado
            detections_final = self._merge_detections(dets_person, dets_other)


        # Inferencia de Pose
        res_pose = self.model_pose(
            frame,
            device=self.cfg.DEVICE,
            conf=self.cfg.CONF_POSE,
            verbose=False,
        )
        res_pose = add_virtual_keypoints_to_results(
                    res_pose, 
                    pairs=self.cfg.KNEES_ANKLES_IDX, 
                    extension_factor=self.cfg.FOOT_EXTENSION_FACTOR
                )
        res_pose = add_virtual_keypoints_to_results(
                    res_pose, 
                    pairs=self.cfg.ARMS_IDXS, 
                    extension_factor=self.cfg.MANO_EXTENSION_FACTOR
                )

        return {
            "raw_obj": res_obj,
            "raw_pose": res_pose,
            "detections_sv": detections_final,
        }

    def _handle_risk_transition(self, person_id, risk_type, scene_name, is_active, 
                                  person, timestamp, ts_str, log_fn, clip_duration=8.0):
        """
        Lógica genérica de transición de evento (Start/Stop clip + registro SQLite).
        
        Detecta transiciones de estado (False→True / True→False) para iniciar/detener
        clips y registrar eventos en SQLite mientras el evento esté activo.
        
        Args:
            person_id: Identificador único de la persona (ej: "person_5")
            risk_type: Tipo de evento para el nombre del archivo ("REBA", "MAC")
            scene_name: Nombre de escena para el clip writer
            is_active: Estado actual del evento (True/False)
            person: Objeto Person con atributos del trabajador
            timestamp: Timestamp ISO del frame
            ts_str: Timestamp formateado para nombres de archivo
            log_fn: Función de logging (log_reba_event, log_mac_event)
            clip_duration: Duración del clip en segundos
        """
        state_key = f"{person_id}_{risk_type.lower()}"
        prev_active = self.previous_risk_states.get(state_key, False)
        video_file = None
        
        # Transición: Inicio de evento (False → True)
        if is_active and not prev_active:
            video_file = f"{person_id}_{risk_type}_{ts_str}.mp4"
            if self.cfg.CLIP_ENABLED:
                logger.info(f"[Clip] START: {scene_name} ({risk_type} Alto detectado)")
                self.clip_writer.start_clip(scene_name, list(self.pre_roll_buffer), video_file, duration_sec=clip_duration)
        
        # Transición: Fin de evento (True → False)
        elif not is_active and prev_active:
            if self.cfg.CLIP_ENABLED:
                logger.info(f"[Clip] STOP: {scene_name}")
                self.clip_writer.stop_clip(scene_name)
        
        # Mientras el evento esté activo: registrar en SQLite
        if is_active:
            log_fn(
                person=person,
                timestamp=timestamp,
                video_file=video_file if self.cfg.CLIP_ENABLED else None
            )
        
        # Actualizar estado
        self.previous_risk_states[state_key] = is_active

    def _handle_risks(self, final_detection):
        timestamp = datetime.now(self.bogota).isoformat()
        ts_str = timestamp.replace(":", "-").replace(".", "_")
        
        for person in final_detection.people:
            person_id = f"person_{person.track_id}"

            # ── REBA: Transición de riesgo ergonómico ──
            self._handle_risk_transition(
                person_id=person_id,
                risk_type="REBA",
                scene_name=f"REBA_{person_id}",
                is_active=person.bad_pose["bad_pose_reba"],
                person=person,
                timestamp=timestamp,
                ts_str=ts_str,
                log_fn=self.work_zone_logger.log_reba_event,
                clip_duration=8.0
            )

            # ── MAC: Transición de riesgo por levantamiento ──
            self._handle_risk_transition(
                person_id=person_id,
                risk_type="MAC",
                scene_name=f"MAC_{person_id}",
                is_active=person.bad_pose["bad_pose_mac"],
                person=person,
                timestamp=timestamp,
                ts_str=ts_str,
                log_fn=self.work_zone_logger.log_mac_event,
                clip_duration=8.0
            )
            
            # ── EPP: Registro de incumplimiento (evento único, sin transición) ──
            if person.epp_alert_triggered:
                missing_str = person.missing_epps_str or 'unknown'
                epp_filename = f"{person_id}_EPP_{missing_str}_{ts_str}.mp4"
                
                self.work_zone_logger.log_epp_event(
                    person=person, 
                    missing_epps_str=missing_str, 
                    timestamp=timestamp, 
                    video_file=epp_filename
                )

                if self.cfg.CLIP_ENABLED:
                    logger.info(f"[Clip] START: EPP_{person_id} (EPP faltante: {missing_str})")
                    self.clip_writer.start_clip(f"EPP_{person_id}", list(self.pre_roll_buffer), epp_filename, duration_sec=3.0)
        
        # Snapshot periódico de personas en zona
        people_in_zone = sum(1 for p in final_detection.people if p.in_zone)
        self.work_zone_logger.log_zone_snapshot(
            people_in_zone_count=people_in_zone,
            timestamp=timestamp
        )
    
    def _visualize(self, frame, detections_sv, engine_results, raw_pose=None, final_detection=None, fused_entities=None):
        # Dibujar Cajas de EPP (Cascos, Botas, Gafas, Guantes y Personas) usando Supervision
        box_annot = BoxAnnotator(thickness=1, color_lookup=ColorLookup.INDEX)
        lab_annot = LabelAnnotator(
            color_lookup=ColorLookup.INDEX,
            text_padding=3,
            text_scale=0.35,
            text_thickness=1,
            smart_position=True,
        )

        # Generar etiquetas para las cajas
        labels = []
        if detections_sv.class_id is not None and len(detections_sv) > 0:
            for i in range(len(detections_sv)):
                cls_id = int(detections_sv.class_id[i])
                cls_name = str(detections_sv.data["class_name"][i])
                tid = detections_sv.tracker_id[i] if detections_sv.tracker_id is not None else None
                tid_txt = "N/A" if tid is None else str(int(tid))

                # Si es persona, anexar color de casco + REBA Total
                if tid is not None and cls_id == int(self.cfg.PERSON_CLASS_ID):
                    info = self.helmet_state.get(int(tid), {})
                    hc = info.get("color", "unknown")
                    hs = info.get("score", 0.0)
                    
                    # Buscar REBA/MAC scores en final_detection
                    extra_info = ""
                    if final_detection and final_detection.people:
                        person = next((p for p in final_detection.people if p.track_id == int(tid)), None)
                        if person:
                            if person.reba_total > 0:
                                extra_info += f" | REBA:{person.reba_total}"
                            if person.mac_total > 0:
                                extra_info += f" | MAC:{person.mac_total}"
                    
                    labels.append(f"#{tid_txt} {cls_name} | helmet={hc}{extra_info}")
                else:
                    labels.append(f"{cls_name}")

        frame = box_annot.annotate(scene=frame, detections=detections_sv)
        frame = lab_annot.annotate(scene=frame, detections=detections_sv, labels=labels)

        # Dibujar sub-recuadro morado de validación de EPP
        margin = self.cfg.EPP_VALIDATION_MARGIN
        w, h = self.cfg.RESIZE
        # cv2.rectangle(frame, (margin, margin), (w - margin, h - margin), (255, 0, 255), 2)

        # HUD + Esqueleto
        lines = []
        for scene, res in engine_results.items():
            count = res.get("people_in_zone_count", 0)
            risk = res.get("risk_active", False)
            status = "RIESGO" if risk else "NORMAL"
            lines.append(f"{scene}: {status} | Personas en zona: {count}")
        
        # Agregar línea con REBA scores de todas las personas detectadas
        if final_detection and final_detection.people:
            reba_lines = []
            for p in final_detection.people:
                if p.reba_total > 0:
                    reba_lines.append(f"P#{p.track_id}:REBA{p.reba_total}")
            if reba_lines:
                lines.append(f"Scores: {' | '.join(reba_lines)}")

        # Pasar raw_pose a draw_hud para dibujar esqueleto
        draw_hud(frame, self.fps_smoothed, lines, pose_result=raw_pose, final_detection=final_detection, fused_entities=fused_entities) 
        
        # Dibujar polígono de zona de levantamiento MAC
        if self.cfg.MAC_ENABLED:
            # Color dinámico: cian si no hay lifting, magenta si hay lifting detectado
            mac_lifting_active = False
            if final_detection and final_detection.people:
                mac_lifting_active = any(p.mac_lifting_detected for p in final_detection.people)
            
            mac_color = (255, 0, 255) if mac_lifting_active else (255, 255, 0)  # Magenta / Cian
            cv2.polylines(frame, [self.cfg.MAC_LIFTING_ZONE_POLY], isClosed=True, color=mac_color, thickness=2)
    
    def move_database(self):
        logger.info("Ejecutando rotacion de Base de Datos...")

        db_path = os.path.join(self.cfg.LOG_DIR, "person_events.db")
        upload_path = os.path.join(self.cfg.LOG_DIR, "person_events_upload.db")

        if os.path.exists(db_path):
            try:
                if os.path.exists(upload_path):
                    logger.warning("DB antigua encontrada. Sobrescribiendo.")
                    os.remove(upload_path)
                # shutil.move(db_path, upload_path)
                shutil.copy2(db_path, upload_path) # solución provisional
                logger.info(f"DB lista para subida: {upload_path}")
            except Exception as e:
                logger.error(f"Error moviendo DB: {e}")
        else:
            logger.warning("No se encontro archivo .db para rotar.")

    def cleanup(self):
        logger.info("Finalizando y liberando recursos...")

        if self.clip_writer:
            self.clip_writer.stop()
        
        # Detener WorkZoneLogger (event-based)
        if self.work_zone_logger:
            self.work_zone_logger.stop_logger()

        # DEPRECADO: db_logger antiguo
        # self.db_logger.stop_logger()
        # self.helmet_logger.stop_logger()

        if self.cap:
            self.cap.release()

        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()

        if self.monitor:
            self.monitor.finalize(output_dir=self.cfg.LOG_DIR)

        logger.info("Sesion finalizada correctamente.")


# =============================
# Punto de entrada
# =============================
if __name__ == "__main__":
    import glob
    
    # --- EJECUCIÓN TIEMPO REAL ---
    cfg = Config()
    app = RiskDetectionApp(cfg)
    
    while app.keep_running:
        try:
            app.setup()
            app.run()
        except Exception as e:
            logger.exception(f"Error en ejecucion principal: {e}")
            logger.info("Reiniciando en 15 segundos...")
            time.sleep(15)
    
        if not app.keep_running:
            break

    # --- EJECUCIÓN BATCH DE VIDEOS ---
    # default_videos_dir = os.path.join(os.path.dirname(__file__), "videos", "batch")
    # os.makedirs(default_videos_dir, exist_ok=True)
    # video_files = glob.glob(os.path.join(default_videos_dir, "*.mp4"))
    
    # if not video_files:
    #     logger.warning(f"No se encontraron videos .mp4 en {default_videos_dir}")
    # else:
    #     logger.info(f"Se encontraron {len(video_files)} videos para procesar en lote.")
        
    #     # Procesar registro por registro
    #     for idx, video_path in enumerate(video_files, 1):
    #         logger.info(f"#" * 50)
    #         logger.info(f"Iniciando video {idx}/{len(video_files)}: {os.path.basename(video_path)}")
    #         logger.info(f"#" * 50)
            
    #         cfg = Config()
    #         cfg.VIDEO_SOURCE = video_path
    #         cfg.VISUALIZE = False
            
    #         try:
    #             start_time = time.time()
    #             app = RiskDetectionApp(cfg)
    #             app.setup()
    #             app.run()
                
    #             elapsed_time = time.time() - start_time
    #             logger.info(f"✅ Procesamiento finalizado con éxito para: {os.path.basename(video_path)}")
    #             logger.info(f"⏱️ Tiempo de ejecución: {elapsed_time:.2f} segundos\n")
                
    #         except KeyboardInterrupt:
    #             logger.warning("⏸️ Ejecución en lote detenida forzosamente por el usuario (Ctrl + C).")
    #             app.keep_running = False
    #         except Exception as e:
    #             logger.error(f"❌ Error crítico procesando {video_path}: {e}", exc_info=True)
    #         finally:
    #             # Importante: liberar hilos de clips, logger y opencv.
    #             if 'app' in locals():
    #                 app.cleanup()
                
    #         # Comprobar si la aplicación se detuvo de manera controlada por señal del usuario
    #         if not app.keep_running:
    #             logger.warning("⏸️ Bucle principal detenido por señal. Abortando el resto del batch.")
    #             break
                
    #     logger.info("🎉 Procesamiento en lote completado.")

    sys.exit(0)
