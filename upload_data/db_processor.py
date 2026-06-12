import os
import sqlite3
import pandas as pd
import json
import sys
import logging
from config_uploader import ConfigUploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()
cfg = ConfigUploader()

def find_db():
    """Encuentra el archivo .db."""
    db_name = f"registros_riesgos.db"
    db_path = os.path.join(cfg.LOG_DIR, db_name)
    
    if not os.path.exists(db_path):
        logger.warning(f"No se encontró el archivo de log: {db_path}")
        return None
    return db_path

# def find_db_moved():
#     """Encuentra el archivo .db copiado (el que se subirá al storage)."""
#     db_name_upload = f"registros_riesgos_upload.db"
#     db_path_upload = os.path.join(cfg.LOG_DIR, db_name_upload)
    
#     if not os.path.exists(db_path_upload):
#         logger.warning(f"No se encontró el archivo de log: {db_path_upload}")
#         return None
#     return db_path_upload

def process_reba_events(df):

    df_reba = df[df["event_type"] == "reba_high"]

    # Ordenar por escena y luego por tiempo
    df_reba = df_reba.sort_values(by=['track_id', 'event_type', 'timestamp'])
    # Encontrar los "saltos" de tiempo
    time_diff = df_reba.groupby(['track_id', 'event_type'])['timestamp'].diff()
    
    # Identificar el inicio de un nuevo incidente
    is_new_incident = (time_diff.isna()) | (time_diff > pd.Timedelta(seconds=cfg.GAP_THRESHOLD_SECONDS))
    # Crear un ID de incidente único (usando cumsum)
    incident_grouper = is_new_incident.cumsum()
    
    def aggregate_video_files(series):
        unique_files = series.dropna().unique().tolist()
        return unique_files if unique_files else [] # Devolver lista vacía en lugar de None

    # Agrupar por escena y ID de incidente, y agregar
    df_result_reba = df_reba.groupby(['track_id', 'event_type', incident_grouper.rename('incident_id')]).agg(
        t_start=('timestamp', 'min'),
        t_end=('timestamp', 'max'),
        reba_score_a_max=("reba_score_a", "max"),
        reba_score_b_max=("reba_score_b", "max"),
        reba_score_total_max=("reba_total", "max"),
        reba_confidence_max=("confidence", "max"),
        # Si se requiere, crear una lista de archivos de video para este incidente
        video_files=('video_file', aggregate_video_files)
    ).reset_index()

    # Calcular la duración
    df_result_reba['duration_seconds'] = (df_result_reba['t_end'] - df_result_reba['t_start']).dt.total_seconds()

    logger.info(f"Procesamiento de BBDD Eventos REBA completo. {len(df_reba)} registros consolidados en {len(df_result_reba)} incidentes posturales")

    return df_result_reba

def process_mac_events(df):

    df_mac = df[df["event_type"] == "mac_high"]

    # Ordenar por escena y luego por tiempo
    df_mac = df_mac.sort_values(by=['track_id', 'event_type', 'timestamp'])
    # Encontrar los "saltos" de tiempo
    time_diff = df_mac.groupby(['track_id', 'event_type'])['timestamp'].diff()
    
    # Identificar el inicio de un nuevo incidente
    is_new_incident = (time_diff.isna()) | (time_diff > pd.Timedelta(seconds=cfg.GAP_THRESHOLD_SECONDS))
    # Crear un ID de incidente único (usando cumsum)
    incident_grouper = is_new_incident.cumsum()
    
    def aggregate_video_files(series):
        unique_files = series.dropna().unique().tolist()
        return unique_files if unique_files else [] # Devolver lista vacía en lugar de None

    # Agrupar por escena y ID de incidente, y agregar
    df_result_mac = df_mac.groupby(['track_id', 'event_type', incident_grouper.rename('incident_id')]).agg(
        t_start=('timestamp', 'min'),
        t_end=('timestamp', 'max'),
        mac_score_b_max=("mac_score_b", "max"),
        mac_score_c_max=("mac_score_c", "max"),
        mac_score_d_max=("mac_score_d", "max"),
        mac_score_total_max=("mac_total", "max"),
        mac_confidence_max=("confidence", "max"),
        # Si se requiere, crear una lista de archivos de video para este incidente
        video_files=('video_file', aggregate_video_files)
    ).reset_index()

    # Calcular la duración
    df_result_mac['duration_seconds'] = (df_result_mac['t_end'] - df_result_mac['t_start']).dt.total_seconds()

    logger.info(f"Procesamiento de BBDD Eventos MAC completo. {len(df_mac)} registros consolidados en {len(df_result_mac)} incidentes MAC")

    return df_result_mac

def process_zone_events(df):

    df_zone = df[df["event_type"] == "zone_snapshot"]
    df_zone = df_zone.sort_values(by=['timestamp'])
    df_zone = df_zone[["event_type","timestamp", "people_in_zone"]]

    df_result = (
        df_zone
        # 1️. Crear la hora base con fecha incluida
        .assign(hour_floor=df_zone["timestamp"].dt.floor("h"))
        
        # 2️. Agrupar por esa hora real
        .groupby("hour_floor", as_index=False)
        .agg(
            event_type=("event_type", "first"),
            people_in_zone_prom=("people_in_zone", "mean"),
            people_in_zone_max=("people_in_zone", "max"),
            people_in_zone_mode=(
                "people_in_zone",
                lambda x: x.mode().iloc[0] if not x.mode().empty else None
            )
        )
    )

    # # 3️. Construir time_start y time_end con fecha correcta
    df_result["t_start"] = df_result["hour_floor"]
    df_result["t_end"] = df_result["hour_floor"] + pd.Timedelta(minutes=59, seconds=59)

    # # 4. Si quieres mantener también el número de hora
    df_result["hour"] = df_result["hour_floor"].dt.hour

    # # 5. Duración en segundos
    df_result["duration_seconds"] = (
        df_result["t_end"] - df_result["t_start"]
    ).dt.total_seconds()

    # # 56. Orden final de columnas
    df_result_zone = df_result[
        ["event_type", "t_start", "t_end", "duration_seconds", "people_in_zone_prom", "people_in_zone_max", "people_in_zone_mode"]
    ]
    logger.info(f"Procesamiento de BBDD Eventos Zone completo. {len(df_result_zone)} registros consolidados.")

    return df_result_zone

def process_epp_events(df):

    df_epp = df[df["event_type"] == "epp_noncompliant"]
    df_epp = df_epp.sort_values(by=['track_id', 'timestamp'])
    df_epp["t_start"], df_epp["t_end"] = df_epp["timestamp"], df_epp["timestamp"]

    df_epp["duration_seconds"] = (
        df_epp["t_end"] - df_epp["t_start"]
    ).dt.total_seconds()
    df_epp["video_file"] = df_epp["video_file"].map(lambda x: [x])
    df_epp = df_epp.rename(columns={"video_file": "video_files"})

    df_result_epp = df_epp[
        ["track_id", "event_type", "t_start", "t_end", "duration_seconds", "helmet_color", "helmet", "gloves", "boots",	"safety_glasses", "video_files"]
    ]
    logger.info(f"Procesamiento de BBDD Eventos EPP completo. {len(df_result_epp)} registros consolidados.")
    return df_result_epp

def process_risk_events(db_path):
    """
    Lee la BBDD de SQLite, la procesa con Pandas para agrupar
    eventos de riesgo basados en el tiempo.
    """
    logger.info(f"Procesando archivo: {db_path}")
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM person_events WHERE event_type IS NOT NULL", conn)
        
        if df.empty:
            logger.warning("No se encontraron eventos de riesgo activos en la BBDD.")
            return pd.DataFrame() # Devolver DataFrame vacío

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_reba_events = process_reba_events(df)
        df_mac_events = process_mac_events(df)
        df_zone_events = process_zone_events(df)
        df_epp_events = process_epp_events(df)

        summary_df = pd.concat(
            [df_reba_events, df_mac_events, df_zone_events, df_epp_events],
            ignore_index=True,
            sort=False
        ).drop(columns=['incident_id'])

        logger.info(f"Procesamiento de BBDD completo. {len(df)} registros consolidados en {len(summary_df)} incidentes.")
        
        return summary_df

    except Exception as e:
        logger.error(f"Error al procesar la BBDD con Pandas: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def enrich_summary_data(summary_df, video_url_map):
    """
    Toma el DataFrame de incidentes y lo enriquece con metadatos
    y las URLs de los videos finales.
    """
    enriched_df = summary_df # Continuar sin enriquecimiento
        
    # Enriquecer con URLs de Video
    if not video_url_map:
        logger.warning("No hay mapa de URLs de video. La columna 'video_url' estará vacía.")
        enriched_df['video_url'] = None
    else:
        logger.info("Mapeando URLs de video a los incidentes...")
        enriched_df['video_url'] = enriched_df['final_video_file'].map(video_url_map)
    
    # Limpiar columnas internas
    final_cols = [col for col in enriched_df.columns if col not in ['video_files', 'final_video_file']]
    enriched_df = enriched_df[final_cols]
    
    return enriched_df

def write_or_append_csv(df, ruta_csv):
    """
    Guarda un DataFrame en un archivo CSV.
    Si el archivo ya existe, agrega las filas sin escribir el encabezado de nuevo.
    """
    # Verifica si el archivo ya existe
    archivo_existe = os.path.isfile(ruta_csv)
    
    # Escribe o agrega según el caso
    df.to_csv(
        ruta_csv,
        mode='a' if archivo_existe else 'w',  # 'a' = append, 'w' = write
        header=not archivo_existe,            # Solo escribe encabezado si el archivo no existe
        date_format='%Y-%m-%dT%H:%M:%S.%f',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ Datos {'agregados' if archivo_existe else 'guardados'} correctamente en {ruta_csv}")

def write_or_append_dbsqlite(df, ruta_db, nombre_tabla):
    """
    Guarda un DataFrame en una base de datos SQLite.
    Si la tabla ya existe, agrega las filas nuevas sin borrar las anteriores.
    """

    # Crear conexión (si no existe la BD, se crea)
    conexion = sqlite3.connect(ruta_db)

    # Insertar datos
    df.to_sql(
        nombre_tabla,
        conexion,
        if_exists='append',  # 'append' agrega los datos, 'replace' sobrescribe
        index=False
    )

    conexion.close()
    print(f"✅ Datos guardados/agregados correctamente en la tabla '{nombre_tabla}' de {ruta_db}")