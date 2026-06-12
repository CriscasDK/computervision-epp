# Changelog

## [Fecha: 2026-06-12]
- **chore**: Limpieza profunda de archivos obsoletos y respaldos
  - Se eliminaron `main_realtime_backup.py`, `scheduler_backup.py`, `fusion_utils.py` y `risk_metadata.json`.
  - Se eliminó el archivo `skills-lock.json` en desuso.
- **docs**: Actualización extensa del README y recursos gráficos
  - Se actualizó el `README.md` con nueva estructura y detalles.
  - Se reemplazó `docs/Arquitectura.png` por `docs/arquitectura.jpg`.
- **refactor**: Ajustes menores en configuración y motores de riesgo
  - Se actualizaron variables en `.env.example` y configuraciones en `docker-compose.yml`.
  - Se realizaron mejoras de consistencia en `base_scene.py`, motores de score y `db_processor.py`.

## [Fecha: 2026-04-21]
- **feat(data)**: Integrar procesamiento de eventos MAC y ajustar umbrales de detección
  - Se implementó `process_mac_events` en `db_processor.py` para analizar incidentes MAC de alto score.
  - Se ajustó el valor de `EPP_CONFIRM_THRESHOLD` y el margen de histéresis `pos_threshold` en `mac_score.py`.
  - Se corrigió el nombre de la columna `reba_confidence` a `confidence` y se incluyó `track_id` en el procesamiento de eventos EPP.

## [Fecha: 2026-04-20]
- **refactor(core)**: reorganizar módulos de guardado en SQLite y grabacion de videos, a demás de limpieza de código
  - Se eliminó el módulo obsoleto `db_logger.py`.
  - Se agregaron logs de depuración para puntajes de eventos en `risk_engine.py`.
  - Se ajustaron monitores de zona y cálculos de eventos (MAC, REBA).
## [Fecha: 2026-03-25]
- **feat(models)**: Añadir campos para rastrear alertas de EPP en el modelo `Person`
  - Incluir `epp_alert_triggered` y `missing_epps_str` en la clase `Person` para gestionar el estado de las alertas.
- **refactor(engine)**: Ajustar umbrales de histéresis para mayor estabilidad
  - Incrementar `pos_threshold` de 20 a 30 en `epp_monitor.py` y `reba_score_total.py`.
- **fix(epp)**: Optimizar la limpieza de tracks no conformes en `epp_monitor.py`.

## [Fecha: 2026-02-27]
- **docs(logger)**: restaurar comentarios de documentación en `WorkZoneLogger`
  - Se restauraron los comentarios explicativos y docstrings dentro del código de `work_zone_logger.py` para mejorar la legibilidad y mantenimiento del logger basado en eventos.

## [Fecha: 2026-02-27]
- **refactor(logger)**: cambiar registro de presencia a snapshots periódicos
  - Reemplazar registro por eventos de entrada/salida de zona con toma de instantáneas (snapshots) cada cierto intervalo de tiempo en `WorkZoneLogger`.
  - Añadir parámetro de configuración `ZONE_SNAPSHOT_INTERVAL` en `config.py`.
  - Remover función `check_disappeared_tracks` y simplificar lógica de rastreo en `main_realtime.py`.

## [Fecha: 2026-02-26]
- **feat(core)**: Mejoras de robustez en tracking y visualización en tiempo real
  - Implementación de periodo de gracia para tracks desaparecidos en `WorkZoneLogger`.
  - Integración de scores REBA en el HUD y HUD simplificado en `main_realtime.py`.
  - Refactorización de `HelmetColorTracker` para mejorar estabilidad y soporte de perfiles día/noche.
  - Optimización de limpieza de recursos y cierre de hilos.

## [Date: 2026-02-26]
- **feat(epp)**: Agregar propiedad epp_evaluable para determinar si una persona es evaluable para EPP
  - Agregar campo `epp_evaluable` en el modelo `Person` en `models.py`
  - Implementar lógica de evaluabilidad en `epp_monitor.py`
  - Ajustar `work_zone_logger.py` para verificar `epp_evaluable` antes de evaluar EPP

## [Date: 2026-02-26]
- **chore**: Add agent skills configuration and update risk detection modules
  - Add `.agents/` directory with conventional-commit, changelog-updater, find-skills, and skill-creator skills
  - Add `risk_detection/skills-lock.json` for skills dependency tracking
  - Update `risk_detection/config.py` with configuration improvements
  - Update `risk_detection/engine/epp_monitor.py`, `models.py`, and `reba_score_a.py`
  - Update `risk_detection/in_out/work_zone_logger.py`
  - Update `risk_detection/main_realtime.py`
