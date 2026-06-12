# in_out/helmet_color_logger.py

import os
import csv
from datetime import datetime


class HelmetColorLogger:
    """
    Logger simple a CSV por frame para color de casco asociado a track_id.
    """

    def __init__(self):
        self._fh = None
        self._writer = None
        self._path = None

    @property
    def path(self):
        return self._path

    def start_logger(self, output_dir, filename="helmet_colors.csv"):
        os.makedirs(output_dir, exist_ok=True)
        self._path = os.path.join(output_dir, filename)

        file_exists = os.path.exists(self._path)
        self._fh = open(self._path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)

        if not file_exists:
            self._writer.writerow([
                "timestamp_iso",
                "frame_idx",
                "track_id",
                "helmet_color",
                "helmet_score",
                "helmet_box_xyxy",
                "person_box_xyxy",
            ])
            self._fh.flush()

        return self._path

    def log(self, frame_idx, track_id, helmet_color, helmet_score, helmet_box=None, person_box=None, timestamp_iso=None):
        if self._writer is None:
            return

        if timestamp_iso is None:
            timestamp_iso = datetime.utcnow().isoformat()

        self._writer.writerow([
            timestamp_iso,
            int(frame_idx),
            int(track_id),
            str(helmet_color),
            float(helmet_score),
            "" if helmet_box is None else [float(x) for x in helmet_box],
            "" if person_box is None else [float(x) for x in person_box],
        ])
        self._fh.flush()

    def stop_logger(self):
        if self._fh:
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = None
        self._writer = None
