"""ByteTracker — asociační tracker se 3 koly párování."""
from __future__ import annotations

from .base_tracker import BaseTracker
from .structures import Track, TrackState
from ..detection.structures import Detection
from ..utils.overlap import greedy_match, box_center_dist, iou


class ByteTracker:
    """
    3-kola greedy IOU matching:
       1. všechny tracky vs silné detekce   → ACTIVE
       2. nespárované tracky vs slabé detekce → LOW
       3. zbylé silné detekce vs ztracené tracky → reinit, případně nové tracky

    `tracker_factory` je callable bez argumentů vracející nový BaseTracker.
    """

    def __init__(self, tracker_factory, config: dict | None = None):
        self._factory = tracker_factory
        self.tracks: list[Track] = []
        cfg = config or {}
        self.high_thresh: float = float(cfg.get("high_thresh", 0.4))
        self.low_thresh: float = float(cfg.get("low_thresh", 0.08))
        self.max_lost_frames: int = int(cfg.get("max_lost_frames", 6))
        self.iou_threshold: float = float(cfg.get("iou_threshold", 0.08))
        self.reinit_dist_thresh: float = float(cfg.get("reinit_dist_thresh", 40))
        self._next_id: int = 0

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _make_track(self, det: Detection, frame_bgr) -> Track:
        tr = Track(
            id=self._new_id(),
            box=list(det.box),
            conf=det.confidence,
            state=TrackState.ACTIVE,
            landmarks=det.landmarks if det.landmarks else None,
        )
        tr.tracker = self._factory()
        tr.tracker.init(frame_bgr, det.box)
        return tr

    @staticmethod
    def _get_tracker_box(tr: Track) -> list:
        return tr.tracker.get_box()

    def _match_step(
        self,
        tracks_subset: list[Track],
        detections: list[Detection],
        iou_thresh: float,
        on_match,
    ) -> tuple[list, list]:
        """
        Greedy IOU matching. `on_match(track, det)` zavoláno pro každý pár.
        Vrací (pairs, unmatched_detection_indices).
        """
        det_tuples = [d.to_tuple() for d in detections]
        # Připravíme fiktivní "track-like" objekty s get_box pro greedy_match
        class _BoxAdapter:
            __slots__ = ("box",)
            def __init__(self, box): self.box = box
            def get_box(self): return self.box

        # Adaptéry jen pro IOU výpočet
        adapters = [_BoxAdapter(t.tracker.get_box()) for t in tracks_subset]
        pairs, _, unmatched_d = greedy_match(adapters, det_tuples, iou_thresh)
        for t_idx, d_idx in pairs:
            on_match(tracks_subset[t_idx], detections[d_idx])
        return pairs, unmatched_d

    def update(
        self,
        detections: list[Detection],
        frame_bgr,
        cam_dx: float = 0.0,
        cam_dy: float = 0.0,
    ) -> list[Track]:
        # 1. CMC — posuneme tracky PŘED KCF update
        if abs(cam_dx) > 0.1 or abs(cam_dy) > 0.1:
            for t in self.tracks:
                t.tracker.apply_camera_motion(cam_dx, cam_dy)
                t.box = t.tracker.get_box()

        # 2. KCF step pro všechny živé tracky
        for t in self.tracks:
            t.tracker.update(frame_bgr)
            if t.tracker.alive:
                t.box = t.tracker.get_box()
            # refresh metadata z trackeru
            t.kcf_ok = t.tracker.is_ok
            t.kcf_psr = t.tracker.psr
            t.kcf_template_score = t.tracker.template_score

        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        # Kolo 1: všechny tracky vs silné detekce
        def on_match_high(t: Track, d: Detection) -> None:
            self._confirm(t, d, frame_bgr, TrackState.ACTIVE)

        pairs1, unmatched_d1 = self._match_step(
            list(self.tracks), high_dets, self.iou_threshold, on_match_high
        )

        # Kolo 2: nespárované tracky (všechny, ne jen ty z kola 1) vs slabé detekce
        #        — v praxi: tracky které nebyly spárovány v kole 1
        matched_t_ids_1 = {id(self.tracks[i]) for i, _ in pairs1}
        remaining = [t for t in self.tracks if id(t) not in matched_t_ids_1]

        def on_match_low(t: Track, d: Detection) -> None:
            self._confirm(t, d, frame_bgr, TrackState.LOW)

        pairs2, _ = self._match_step(
            remaining, low_dets, self.iou_threshold, on_match_low
        )

        # Kolo 3: zbylé silné detekce vs ztracené tracky → reinit nebo nové tracky
        matched_d_1 = {d_idx for _, d_idx in pairs1}
        leftover = [high_dets[i] for i in range(len(high_dets)) if i not in matched_d_1]

        lost_tracks = [t for t in self.tracks if t.state == TrackState.LOST]
        pairs3: list = []
        if lost_tracks and leftover:
            def on_match_reinit(t: Track, d: Detection) -> None:
                self._confirm(t, d, frame_bgr, TrackState.ACTIVE)
            pairs3, unmatched_dl = self._match_step(
                lost_tracks, leftover, self.iou_threshold, on_match_reinit
            )
        else:
            unmatched_dl = list(range(len(leftover)))

        # Zcela nespárované silné detekce → nové tracky
        active_boxes = [t.box for t in self.tracks if t.state == TrackState.ACTIVE]
        for i in unmatched_dl:
            det = leftover[i]
            if any(iou(det.box, ab) > 0.3 for ab in active_boxes):
                continue
            self.tracks.append(self._make_track(det, frame_bgr))

        # Confidence scoring: matched tracks +0.15, unmatched -0.3
        used_track_ids = set()
        for t_idx, _ in pairs1:
            used_track_ids.add(id(self.tracks[t_idx]))
        for t_idx, _ in pairs2:
            used_track_ids.add(id(remaining[t_idx]))
        if lost_tracks and leftover:
            for t_idx, _ in pairs3:
                used_track_ids.add(id(lost_tracks[t_idx]))

        for t in self.tracks:
            if id(t) in used_track_ids:
                t.confidence_score = min(1.0, t.confidence_score + 0.15)
            else:
                t.confidence_score = max(0.0, t.confidence_score - 0.3)

        # Nespárované tracky z kola 2 → LOST
        matched_t_ids_2 = {id(remaining[i]) for i, _ in pairs2}
        for t in remaining:
            if id(t) not in matched_t_ids_2:
                t.lost_frames += 1
                t.state = TrackState.LOST

        self._deduplicate_tracks()

        # Úklid — confidence_score <= 0 s lost_frames > 2 = definitivně mrtvý
        self.tracks = [
            t for t in self.tracks
            if t.lost_frames <= self.max_lost_frames
            and (t.tracker.alive or t.lost_frames == 0)
            and not (t.confidence_score <= 0.0 and t.lost_frames > 2)
        ]

        return list(self.tracks)

    # ── Pomocné ──────────────────────────────────────────

    def _confirm(self, t: Track, d: Detection, frame_bgr, new_state: TrackState) -> None:
        """LPM potvrdilo track — reinicializuj nebo jen posuň KCF střed."""
        dist = box_center_dist(t.box, d.box)
        if (not t.tracker.is_ok) or (dist > self.reinit_dist_thresh):
            t.tracker.init(frame_bgr, d.box)
        else:
            bx = (d.x1 + d.x2) / 2
            by = (d.y1 + d.y2) / 2
            t.tracker._cx = bx
            t.tracker._cy = by
            t.tracker._ok = True
            t.tracker.refresh_template(frame_bgr)
        # Running average: 30% nová detekce, 70% stávající box (hladší tracking)
        alpha = 0.3
        t.box = [
            alpha * d.x1 + (1 - alpha) * t.box[0],
            alpha * d.y1 + (1 - alpha) * t.box[1],
            alpha * d.x2 + (1 - alpha) * t.box[2],
            alpha * d.y2 + (1 - alpha) * t.box[3],
        ]
        t.conf = d.confidence
        if d.landmarks:
            t.landmarks = d.landmarks
        t.lost_frames = 0
        t.state = new_state

    def _deduplicate_tracks(self) -> None:
        active = [t for t in self.tracks if t.state == TrackState.ACTIVE]
        to_kill = set()
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if iou(active[i].box, active[j].box) > 0.3:
                    if active[i].conf < active[j].conf:
                        to_kill.add(id(active[i]))
                    else:
                        to_kill.add(id(active[j]))
        self.tracks = [t for t in self.tracks if id(t) not in to_kill]
