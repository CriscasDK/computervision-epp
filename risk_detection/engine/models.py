from dataclasses import dataclass, field
from typing import List

@dataclass
class Person:
    track_id: int
    epp: List[str] = field(default_factory=list)
    helmet_color: str = "unknown"
    is_valid_pose: bool = True
    is_confidence: bool = False
    in_zone: bool = False
    bad_pose: dict = field(default_factory=lambda: {"bad_pose_reba": False, "bad_pose_mac": False})
    epp_evaluable: bool = False  # True cuando las extremidades son visibles para evaluar EPP
    reba_score_a: int = 0
    reba_score_conf: float = 0.0 # Confianza en la estimación del REBA Score
    reba_score_b: int = 0
    reba_total: int = 0 
    epp_alert_triggered: bool = False
    missing_epps_str: str = ""
    # MAC (Manual Handling Assessment Charts)
    mac_lifting_detected: bool = False  # True cuando se detecta escenario de levantamiento
    mac_score_b: int = 0
    mac_score_c: int = 0
    mac_score_d: int = 0
    mac_total: int = 0

@dataclass
class Detection:
    people: List[Person] = field(default_factory=list)