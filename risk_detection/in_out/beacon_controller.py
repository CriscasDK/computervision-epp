# risk_detection/in_out/beacon_controller.py
import socket
import queue
import threading
import time

class BeaconController:
    """
    Controlador de baliza (alarma) asíncrono y no bloqueante.
    
    Se ejecuta en un hilo separado para manejar la comunicación TCP con la baliza
    sin bloquear el hilo principal de inferencia de CV.
    Utiliza una cola para recibir "pings" de riesgo y una lógica de
    cooldown para evitar el parpadeo de la alarma.
    """
    def __init__(self, cfg):
        self.ip = cfg.BEACON_IP
        self.port = cfg.BEACON_PORT
        self.timeout = cfg.BEACON_CONNECTION_TIMEOUT
        self.cooldown_sec = cfg.BEACON_COOLDOWN_SEC
        
        #self.cmd_activate = cfg.BEACON_CMD_ACTIVATE
        self.cmd_activate1 = bytes([254, 109, 1])
        self.cmd_activate2 = bytes([254, 111, 1])

        # self.cmd_deactivate = cfg.BEACON_CMD_DEACTIVATE
        self.cmd_deactivate1 = bytes([254, 101, 1])
        self.cmd_deactivate2 = bytes([254, 103, 1])
        
        self.queue = queue.Queue()
        self.thread = None
        self.running = False
        
        self.alarm_is_on = False
        self.sock = None
        
        print(f"🟢 [Beacon] Controlador inicializado. IP: {self.ip}:{self.port}")

    def _connect(self):
        """Intenta (re)establecer la conexión con la baliza."""
        if self.sock:
            self.sock.close()
        
        try:
            print(f"🟡 [Beacon] Conectando a {self.ip}:{self.port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.ip, self.port))
            print("🟢 [Beacon] Conexión exitosa.")
            return True
        except (socket.error, socket.timeout) as e:
            print(f"🔴 [Beacon] Error de conexión: {e}")
            self.sock = None
            return False

    def _send_command(self, command1, command2):
        """
        Envía un comando a la baliza.
        Maneja reintentos de conexión si es necesario.
        """
        if not self.sock:
            if not self._connect():
                return False # Falla al conectar, no se puede enviar comando
        
        try:
            self.sock.sendall(command1)
            #self.sock.sendall(command2)
            return True
        except (socket.error, socket.timeout) as e:
            print(f"🔴 [Beacon] Error al enviar comando: {e}. Reintentando conexión...")
            # La conexión falló, intentar reconectar una vez
            if self._connect():
                try:
                    self.sock.sendall(command1) # Reintentar envío
                    self.sock.sendall(command2) # Reintentar envío
                    return True
                except (socket.error, socket.timeout) as e2:
                    print(f"🔴 [Beacon] Error en el reintento de envío: {e2}")
            
            self.sock = None
            return False

    def _run_worker(self):
        """
        Lógica principal del hilo worker.
        Espera eventos en la cola y gestiona el estado de la alarma.
        """
        print(f"🟢 [Beacon] Hilo worker iniciado. Cooldown: {self.cooldown_sec}s")
        
        while self.running:
            try:
                # Esperar un "True" de riesgo activo.
                # Si no llega nada en 'cooldown_sec' segundos, saltará una excepción.
                self.queue.get(timeout=self.cooldown_sec)
                
                # --- Caso A: Riesgo Detectado ---

                # SI LLEGA UNA ORDEN "True" (el 'try' tiene éxito)
                #   (El 'main' gritó ¡RIESGO!)
                # Se recibió un ping, vaciar la cola por si hay pings acumulados
                while not self.queue.empty():
                    self.queue.get_nowait()
                
                if not self.alarm_is_on:
                    print("🟡 [Beacon] Riesgo detectado. Enviando comando de ACTIVACIÓN.")
                    #if self._send_command(self.cmd_activate):
                    if self._send_command(self.cmd_activate1, self.cmd_activate2):
                        self.alarm_is_on = True
                
            except queue.Empty:
                # --- Caso B: Sin Riesgo (Timeout) ---
                # No llegó un True, asi que entró en el except porque el 'try' falla con 'Empty'
                # Quiere decir que pasaron 'cooldown_sec' segundos sin recibir un "True"
                # Pasó el tiempo de cooldown sin pings.
                if self.alarm_is_on:
                    print("🟢 [Beacon] Cooldown finalizado. Enviando comando de DESACTIVACIÓN.")
                    if self._send_command(self.cmd_deactivate1, self.cmd_deactivate2):
                        self.alarm_is_on = False
        
        # --- Bucle terminado (self.running = False) ---
        print("🟡 [Beacon] Deteniendo hilo worker...")
        if self.alarm_is_on:
            print("🟡 [Beacon] Apagando alarma por cierre de sistema.")
            self._send_command(self.cmd_deactivate1, self.cmd_deactivate2)
            # self._send_command(self.cmd_deactivate2)
            self.alarm_is_on = False
            
        if self.sock:
            self.sock.close()
        print("🟢 [Beacon] Hilo worker detenido limpiamente.")

    def start_controller(self):
        """Inicia el hilo worker de la baliza."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_worker, daemon=True)
            self.thread.start()

    def trigger_alarm(self):
        """
_        Llamado desde el hilo principal (CV).
        Pone un 'ping' en la cola. Es no bloqueante.
_        """
        if self.running:
            self.queue.put(True)

    def stop_controller(self):
        """Detiene el hilo worker y espera a que termine."""
        if self.running:
            print("🟡 [Beacon] Enviando señal de parada al controlador...")
            self.running = False
            self.queue.put(None) # Poner algo en la cola para desbloquear .get()
            if self.thread:
                self.thread.join(timeout=self.timeout + 1.0)
                if self.thread.is_alive():
                    print("🔴 [Beacon] El hilo worker no terminó a tiempo.")
                else:
                    print("🟢 [Beacon] Controlador detenido limpiamente.")

