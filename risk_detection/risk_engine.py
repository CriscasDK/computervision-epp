from engine.work_zone_monitor import WorkZoneMonitor
from engine.epp_monitor import EPPMonitor
from engine.helmet_color import HelmetColorTracker
from engine.reba_score_a import REBAEvaluatorScoreA
from engine.reba_score_b import REBAEvaluatorScoreB
from engine.reba_score_total import REBAEvaluatorScoreTotal
from engine.mac_score import MACEvaluator
from engine.mac_scene_detector import MACSceneDetector

class RiskEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_zone_monitor = WorkZoneMonitor(cfg)
        self.epp_monitor = EPPMonitor(cfg)
        self.helmet_tracker = HelmetColorTracker(cfg)
        self.reba_evaluator_a = REBAEvaluatorScoreA(cfg)
        self.reba_evaluator_b = REBAEvaluatorScoreB(cfg)
        self.reba_evaluator_total = REBAEvaluatorScoreTotal(cfg)
        self.mac_evaluator = MACEvaluator(cfg)
        # Tabla C para cálculo del REBA Score Total
        self.TABLE_C = self.cfg.TABLE_C
        # MAC Scene Detector (condicional)
        self.mac_scene_detector = MACSceneDetector(cfg) if self.cfg.MAC_ENABLED else None

    def process(self, fused_entities, frame=None, raw_detections_sv=None, frame_idx=0):
        all_scene_results = {}

        # print("-----------------")

        # FASE 1: Crear la estructura base (IDs, Zona)
        # Retorna un objeto Detection con una lista de Person (epp=[], helmet_color='unknown')
        work_zone_results, current_detection = self.work_zone_monitor.evaluate(
            fused_entities,
            frame
        )
        
        # FASE 2: Detectar inventario de EPP
        # Actualiza person.epp = ["helmet", "boots", ...]
        epp_detection_results = self.epp_monitor.evaluate(
            fused_entities,
            frame,
            detection_data=current_detection
        )

        # FASE 3: Clasificar y estabilizar color de casco
        # Actualiza person.helmet_color = "white" / "yellow" / ...
        helmet_color_results, helmet_state = self.helmet_tracker.evaluate(
            frame_bgr=frame,
            fused_entities=fused_entities,
            detection_data=epp_detection_results,
            frame_idx=frame_idx
        )

        # FASE 4: Evaluación REBA Score A
        # Actualiza person.reba_score_a y person.reba_score_a_conf
        if self.cfg.REBA_ENABLED:
            reba_score_a_results = self.reba_evaluator_a.evaluate(
                fused_entities=fused_entities,
                detection_data=helmet_color_results
            )
            
        # FASE 5: Evaluación REBA Score B
        # Actualiza person.reba_score_b y person.reba_score_b_conf
            reba_score_b_results = self.reba_evaluator_b.evaluate(
                fused_entities=fused_entities,
                detection_data=reba_score_a_results
            )
        # FASE 6: Calcular REBA Score Total usando Tabla C
        # Tabla C: [score_a][score_b] -> reba_total
            reba_score_total = self.reba_evaluator_total.evaluate(
                detection_data=reba_score_b_results
            )
        else:
            reba_score_total = helmet_color_results

        # FASE 7: Detectar escenario de levantamiento de carga (MAC)
        # Actualiza person.mac_lifting_detected = True/False
        if self.cfg.MAC_ENABLED:
            mac_results = self.mac_scene_detector.evaluate(
                fused_entities=fused_entities,
                detection_data=reba_score_total
            )
        else:
            mac_results = reba_score_total
        
        # FASE 7.1: Evaluación MAC Score (Calculo matemático)
        if self.cfg.MAC_ENABLED:
            final_risk_score = self.mac_evaluator.evaluate(
                fused_entities=fused_entities,
                frame=frame,
                detection_data=mac_results
            )
        else:
            final_risk_score = reba_score_total

        print("-"*50)
        for person in final_risk_score.people:
            print(f"Final Risk Score: {person}")

        all_scene_results[self.work_zone_monitor.name] = work_zone_results

        return all_scene_results, helmet_state, final_risk_score