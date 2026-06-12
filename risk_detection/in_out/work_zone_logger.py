import sqlite3
import queue
import threading
import os
from datetime import datetime
import pytz
import time

# Cola que comunica el hilo principal con el hilo de escritura en DB
event_queue = queue.Queue()

# Variables globales para manejar el hilo y la ruta del archivo
worker_thread = None
db_path = None

class _DummyPerson:
    """Objeto Person mínimo para tracks que desaparecieron sin datos reales."""
    def __init__(self, track_id, in_zone=False):
        self.track_id = track_id
        self.epp = []
        self.helmet_color = 'unknown'
        self.reba_score_a = 0
        self.reba_score_b = 0
        self.reba_total = 0
        self.reba_score_conf = 0.0
        self.bad_pose = {"bad_pose_reba": False, "bad_pose_mac": False}
        self.in_zone = in_zone
        self.mac_score_b = 0
        self.mac_score_c = 0
        self.mac_score_d = 0
        self.mac_total = 0

class WorkZoneLogger:
    """
    Logger optimizado para eventos de zona de trabajo, EPP y REBA.
    
    Características:
    - Event-based logging (solo registra cambios de estado, no frame-by-frame)
    - Threading para no bloquear el hilo principal
    - Tabla única normalizada para todas las caracterizaciones
    - Eventos soportados: zone_snapshot, reba_high, epp_noncompliant
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        # Estado para REBA
        self.high_reba_tracks = {}  # {track_id: {'start_time': timestamp, 'last_seen': frame_idx, 'peak_reba': int}}
        self.grace_period_frames = 30  # Frames de gracia antes de confirmar desaparición (~1 segundo a 30 FPS)
        self.reba_threshold = cfg.REBA_HIGH_THRESHOLD
        # Estado para EPP
        self.epp_noncompliant_tracks = {}  # {track_id: {'pending_frames': int}} para confirmación
        self.epp_registered_tracks = set()  # track_ids ya registrados como no-compliant
        self.epp_required = set(cfg.EPP_REQUIRED)
        self.epp_confirm_threshold = cfg.EPP_CONFIRM_THRESHOLD
        # Estado para Zona Snapshot
        self.zone_snapshot_interval = cfg.ZONE_SNAPSHOT_INTERVAL  # Segundos
        self.last_zone_snapshot_time = None  # Se inicializa con el primer timestamp recibido
    
    # ──────────────────────────────────────────────────────────────
    # Métodos auxiliares DRY
    # ──────────────────────────────────────────────────────────────
    
    def _build_event_data(self, event_type, person, timestamp, video_file=None, **extra):
        """
        Construye el diccionario de datos común para todos los eventos.
        
        Args:
            event_type: Tipo de evento (zone_snapshot, reba_high, etc.)
            person: Objeto Person con atributos del trabajador (o None para snapshots)
            timestamp: Timestamp ISO del evento
            video_file: Archivo de video asociado (opcional)
            **extra: Campos adicionales específicos del evento
        
        Returns:
            dict: Datos listos para encolar
        """
        if person is not None:
            data = {
                'event_type': event_type,
                'track_id': person.track_id,
                'timestamp': timestamp,
                'in_zone': getattr(person, 'in_zone', False),
                'helmet': 'helmet' in person.epp,
                'gloves': 'gloves' in person.epp,
                'boots': 'boots' in person.epp,
                'safety_glasses': 'safety_glasses' in person.epp,
                'helmet_color': person.helmet_color,
                'reba_score_a': person.reba_score_a,
                'reba_score_b': person.reba_score_b,
                'reba_total': person.reba_total,
                'confidence': person.reba_score_conf,
                'bad_pose_reba': person.bad_pose["bad_pose_reba"],
                'bad_pose_mac': person.bad_pose["bad_pose_mac"],
                'mac_score_b': person.mac_score_b,
                'mac_score_c': person.mac_score_c,
                'mac_score_d': person.mac_score_d,
                'mac_total': person.mac_total,
                'video_file': video_file,
                'people_in_zone': None,
            }
        else:
            # Para eventos agregados como zone_snapshot
            data = {
                'event_type': event_type,
                'track_id': 0,
                'timestamp': timestamp,
                'in_zone': False,
                'helmet': False,
                'gloves': False,
                'boots': False,
                'safety_glasses': False,
                'helmet_color': None,
                'reba_score_a': 0,
                'reba_score_b': 0,
                'reba_total': 0,
                'confidence': 0.0,
                'bad_pose_reba': False,
                'bad_pose_mac': False,
                'mac_score_b': 0,
                'mac_score_c': 0,
                'mac_score_d': 0,
                'mac_total': 0,
                'video_file': None,
                'people_in_zone': None,
            }
        data.update(extra)
        return data
    
    def _enqueue_event(self, event_type, person, timestamp, video_file=None, log_msg="", **extra):
        """
        Encola un evento para escritura asíncrona en la BBDD.
        
        Args:
            event_type: Tipo de evento
            person: Objeto Person
            timestamp: Timestamp ISO
            video_file: Archivo de video (opcional)
            log_msg: Mensaje descriptivo para el log de consola
            **extra: Campos adicionales
        """
        try:
            data = self._build_event_data(event_type, person, timestamp, video_file, **extra)
            event_queue.put(data)
            if log_msg:
                print(log_msg)
        except Exception as e:
            print(f"🔴 [WorkZoneLogger] Error al encolar {event_type}: {e}")
    
    @staticmethod
    def _insert_event(cursor, data):
        """Inserta cualquier evento en la tabla person_events."""
        cursor.execute("""
            INSERT OR IGNORE INTO person_events (
                track_id,
                event_type,
                timestamp,
                in_zone,
                people_in_zone,
                bad_pose_reba,
                bad_pose_mac,
                reba_score_a,
                reba_score_b,
                reba_total,
                mac_score_b,
                mac_score_c,
                mac_score_d,
                mac_total,
                confidence,
                helmet,
                gloves,
                boots,
                safety_glasses,
                helmet_color,
                video_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['track_id'],
            data['event_type'],
            data['timestamp'],
            data['in_zone'],
            data['people_in_zone'],
            data['bad_pose_reba'],
            data['bad_pose_mac'],
            data['reba_score_a'],
            data['reba_score_b'],
            data['reba_total'],
            data['mac_score_b'],
            data['mac_score_c'],
            data['mac_score_d'],
            data['mac_total'],
            data['confidence'],
            data['helmet'],
            data['gloves'],
            data['boots'],
            data['safety_glasses'],
            data['helmet_color'],
            data['video_file'],
        ))
    
    # ──────────────────────────────────────────────────────────────
    # Worker de base de datos (hilo separado)
    # ──────────────────────────────────────────────────────────────
        
    def database_worker(self, db_file_path):
        """
        Worker que se ejecuta en un hilo separado.
        Escucha la cola y escribe en la BBDD.
        """
        conn = None
        try:
            conn = sqlite3.connect(db_file_path)
            cursor = conn.cursor()
            
            # Crear tabla person_events con índices
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS person_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Identificación
                    track_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    
                    -- Zona de Trabajo
                    in_zone BOOLEAN NOT NULL DEFAULT 0,
                    people_in_zone INTEGER DEFAULT NULL,
                    
                    -- EPP
                    helmet BOOLEAN DEFAULT 0,
                    gloves BOOLEAN DEFAULT 0,
                    boots BOOLEAN DEFAULT 0,
                    safety_glasses BOOLEAN DEFAULT 0,
                    helmet_color TEXT,
                    
                    -- REBA Scores
                    reba_score_a INTEGER DEFAULT 0,
                    reba_score_b INTEGER DEFAULT 0,
                    reba_total INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.0,
                    bad_pose_reba BOOLEAN DEFAULT 0,
                    bad_pose_mac BOOLEAN DEFAULT 0,
                    
                    -- Metadata
                    video_file TEXT,
                    
                    -- MAC Scores
                    mac_score_b INTEGER DEFAULT 0,
                    mac_score_c INTEGER DEFAULT 0,
                    mac_score_d INTEGER DEFAULT 0,
                    mac_total INTEGER DEFAULT 0,
                    
                    -- Evitar duplicados
                    UNIQUE(track_id, event_type, timestamp)
                )
            """)
            
            # Crear índices para optimizar consultas en Power BI
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON person_events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_id ON person_events(track_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON person_events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_in_zone ON person_events(in_zone)")
            
            conn.commit()
            print(f"🟢 [WorkZoneLogger] Hilo worker conectado a BBDD: {db_file_path}")
            
            while True:
                try:
                    # Bloquea hasta que haya un item en la cola
                    data = event_queue.get()
                    
                    # 'None' es la señal de parada
                    if data is None:
                        print("🟢 [WorkZoneLogger] Señal de parada recibida. Terminando hilo worker.")
                        break
                    
                    self._insert_event(cursor, data)
                    conn.commit()
                    
                except sqlite3.Error as e:
                    print(f"🔴 [WorkZoneLogger] Error de SQLite: {e}")
                except Exception as e:
                    print(f"🔴 [WorkZoneLogger] Error en hilo worker: {e}")
        
        finally:
            if conn:
                conn.close()
                print("🟢 [WorkZoneLogger] Conexión a BBDD cerrada.")
    
    # ──────────────────────────────────────────────────────────────
    # Inicio/Parada del logger
    # ──────────────────────────────────────────────────────────────
    
    def start_logger(self, output_dir="logs"):
        """
        Inicia el hilo worker de la base de datos.
        Debe llamarse una vez al inicio del programa.
        """
        global worker_thread, db_path

        # Si ya existe un hilo corriendo, no iniciar otro encima
        if worker_thread and worker_thread.is_alive():
            print("🟡 [Logger] El hilo ya estaba corriendo.")
            return db_path
        
        # Asegurarse de que la carpeta de logs exista
        os.makedirs(output_dir, exist_ok=True)
        
        # Nombre del archivo de base de datos
        db_path = os.path.join(output_dir, "person_events.db")

        # Limpiar la cola de ejecuciones anteriores por si acaso
        while not event_queue.empty():
            try: event_queue.get_nowait()
            except queue.Empty: break
        
        # Iniciar el hilo worker
        worker_thread = threading.Thread(
            target=self.database_worker,
            args=(db_path,),
            daemon=True
        )
        worker_thread.start()
        
        print(f"🟢 [WorkZoneLogger] Hilo worker iniciado. Guardando en: {db_path}")
        return db_path
    
    def stop_logger(self):
        """
        Envía la señal de parada al hilo worker y espera a que termine.
        Debe llamarse al final del programa para un cierre limpio.
        """
        global worker_thread

        if not worker_thread or not worker_thread.is_alive():
            print("🟡 [Logger] El hilo worker no estaba corriendo.")
            return
        
        print("🟡 [Logger] Enviando señal de parada al worker...")

        # Enviar señal de parada
        event_queue.put(None)

        # Intentar esperar al hilo (con reintentos)
        max_retries = 3
        for i in range(max_retries):
            worker_thread.join(timeout=5.0)
            if not worker_thread.is_alive():
                print("🟢 [Logger] Hilo worker detenido correctamente.")
                break
            print(f"🟡 [Logger] Esperando a que el worker termine (intento {i+1}/{max_retries})...")

        if worker_thread.is_alive():
            print("🔴 [Logger] ALERTA: El hilo worker NO terminó. Posible bloqueo de archivo.")
        else:
            # Limpiar referencia global
            worker_thread = None
            # Doble check de GC
            # gc.collect()
    
    # ──────────────────────────────────────────────────────────────
    # Lógica de eventos: Zona de Trabajo (Snapshot Periódico)
    # ──────────────────────────────────────────────────────────────
    
    def log_zone_snapshot(self, people_in_zone_count, timestamp):
        """
        Registra un snapshot periódico de personas en zona.
        Se llama en cada frame pero solo registra cada zone_snapshot_interval segundos (tiempo real).
        
        Args:
            people_in_zone_count: Cantidad de personas actualmente en zona
            timestamp: Timestamp ISO actual
        """
        now = datetime.fromisoformat(timestamp)
        
        # Inicializar en el primer llamado
        if self.last_zone_snapshot_time is None:
            self.last_zone_snapshot_time = now
            return
        
        # Verificar si pasó el intervalo (basado en tiempo real, no frames)
        elapsed = (now - self.last_zone_snapshot_time).total_seconds()
        
        if elapsed >= self.zone_snapshot_interval:
            self._enqueue_event(
                'zone_snapshot', None, timestamp,
                log_msg=f"📊 [WorkZoneLogger] Snapshot zona: {people_in_zone_count} personas en zona",
                people_in_zone=people_in_zone_count
            )
            self.last_zone_snapshot_time = now
    
    # ──────────────────────────────────────────────────────────────
    # Lógica de eventos: REBA Score Alto
    # ──────────────────────────────────────────────────────────────
    
    def log_reba_event(self, person, timestamp, video_file=None):
        """
        Registra un evento de REBA score alto (>= threshold).
        Solo registra el evento de inicio después de pasar la histéresis.
        
        Args:
            person: Objeto Person con todos los atributos
            timestamp: Timestamp actual (ISO format)
            video_file: Archivo de video asociado (opcional)
        """
        track_id = person.track_id
        self._enqueue_event(
            'reba_high', person, timestamp, video_file,
            log_msg=f"⚠️ [WorkZoneLogger] REBA ALTO: track_id={track_id}, score={person.reba_total}"
        )
    
    
    # ──────────────────────────────────────────────────────────────
    # Lógica de eventos: MAC Score (Lifting)
    # ──────────────────────────────────────────────────────────────
    
    def log_mac_event(self, person, timestamp, video_file=None):
        """
        Registra un evento de MAC score alto (>= threshold).
        La lógica de umbral ya fue evaluada en _handle_risks.
        
        Args:
            person: Objeto Person con todos los atributos
            timestamp: Timestamp actual (ISO format)
            video_file: Archivo de video asociado (opcional)
        """
        track_id = person.track_id
        self._enqueue_event(
            'mac_high', person, timestamp, video_file,
            log_msg=f"⚠️ [WorkZoneLogger] MAC ALTO: track_id={track_id}, score={person.mac_total}"
        )

    # ──────────────────────────────────────────────────────────────
    # Lógica de eventos: EPP No-Compliant
    # ──────────────────────────────────────────────────────────────
    
    def log_epp_event(self, person, missing_epps_str, timestamp, video_file=None):
        """
        Registra un evento de incumplimiento de EPP en la BBDD.
        La lógica de confirmación/histéresis ya fue evaluada por el EPPMonitor.
        
        Args:
            person: Objeto Person con todos los atributos
            missing_epps_str: Cadena de texto con los EPPs faltantes separados por guiones
            timestamp: Timestamp actual (ISO format)
            video_file: Archivo de video asociado
        """
        track_id = person.track_id
        
        self._enqueue_event(
            'epp_noncompliant', person, timestamp, video_file,
            log_msg=f"⚠️ [WorkZoneLogger] EPP FALTANTE: track_id={track_id}, faltante=[{missing_epps_str}]"
        )
