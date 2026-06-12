import os
import sys
import time
import logging
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings
from config_uploader import ConfigUploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()
cfg = ConfigUploader()

class AzureBlobHandler:
    """
    Maneja toda la conectividad y subidas a Azure Blob Storage.
    """
    def __init__(self, conn_str, container_name, account_name, account_url):
        if not all([conn_str, container_name, account_name, account_url]):
            logger.error("Faltan una o más credenciales de Azure (ConnectionString, ContainerName, AccountName, AccountURL).")
            raise ValueError("Faltan credenciales de Azure")
            
        self.conn_str = conn_str
        self.container_name = container_name
        self.account_name = account_name
        self.account_url = account_url.rstrip('/') # Asegurar que no tenga / al final
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            logger.info("✅ Conexión con Azure Blob Storage establecida.")
        except Exception as e:
            logger.error(f"Error al conectar con Azure (revisar connection string): {e}")
            raise

    def upload_file_and_get_sas_url(self, local_file_path, azure_file_path, max_retries=3, retry_delay=5):
        """
        Sube un archivo local a Azure y devuelve una URL SAS de lectura.
        Incluye lógica de reintentos automática.
        Devuelve None si fallan todos los intentos.
        """
        # Bucle de reintentos
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    logger.info(f"🔄 Reintentando subida ({attempt}/{max_retries}) para: {os.path.basename(local_file_path)}...")

                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=azure_file_path
                )
                if local_file_path.lower().endswith('.mp4'):
                    content_type = 'video/mp4'
                else:
                    content_type = 'application/octet-stream'

                content_settings = ContentSettings(content_type=content_type)
                
                logger.info(f"  Subiendo {local_file_path} a {azure_file_path}...")
                with open(local_file_path, "rb") as data:
                    blob_client.upload_blob(
                        data,
                        overwrite=True,
                        content_settings=content_settings
                        )
                
                logger.info(f"  ✅ Subida completada. Generando URL SAS...")

                sas_token = generate_blob_sas(
                    account_name=self.account_name,
                    container_name=self.container_name,
                    blob_name=azure_file_path,
                    account_key=self.blob_service_client.credential.account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.now(cfg.BOGOTA_TZ) + timedelta(days=cfg.SAS_EXPIRATION_DAYS)
                )
                
                file_url = f"{self.account_url}/{self.container_name}/{azure_file_path}?{sas_token}"
                logger.info(f"  ✅ URL SAS generada (válida por {cfg.SAS_EXPIRATION_DAYS} días).")

                # Si todo sale bien, retornamos la URL inmediatamente (rompe el bucle y la función)
                return file_url

            except Exception as e:
                logger.error(f"  ❌ Fallo al subir {local_file_path} (Intento {attempt}/{max_retries}): {e}")
                # Si no es el último intento, esperamos antes de reintentar
                if attempt < max_retries:
                    logger.info(f"  ⏳ Esperando {retry_delay} segundos antes del siguiente intento...")
                    time.sleep(retry_delay)
                else:
                    logger.error("  ❌ Se han agotado todos los intentos de subida.", exc_info=True)
        # Si salimos del bucle sin retornar nada, significa que falló todas las veces
        return None