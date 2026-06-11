"""Backward fill — purely linear box interpolation, no CMC.

Postup:
  1. Merge duplicitních ID (embedding + IoU fallback)
  2. Sběr detection framů pro každé ID, přeskočit < 3
  3. Čistě lineární interpolace boxu mezi detekcema (žádná CMC, žádný landmarks)
  4. Finální cleanup duplicit
"""
from __future__ import annotations

from .overlap import iou
from .face_embedder import embed, distance
from .frame_history import FrameSnapshot, TrackFrame


class BackwardFiller:
    def __init__(self, video_width: int, video_height: int,
                 max_backfill_gap: int = 30,
                 min_detections: int = 3):
        self.video_width = video_width
        self.video_height = video_height
        self.max_backfill_gap = max_backfill_gap
        self.min_detections = min_detections

    # ── ID merge (stejný jako dřív, funguje) ─────────────────

    def _merge_duplicate_person_ids(
        self, snapshots: list[FrameSnapshot]
    ) -> None:
        id_merge: dict[int, int] = {}

        id_embeds: dict[int, list[float]] = {}
        for snap in snapshots:
            for tr in snap.tracks:
                if tr.id in id_embeds:
                    continue
                if not tr.landmarks or len(tr.landmarks) < 5:
                    continue
                feat = embed(tr.landmarks)
                if feat:
                    id_embeds[tr.id] = feat

        emb_ids = sorted(id_embeds.keys())
        for i in range(len(emb_ids)):
            for j in range(i + 1, len(emb_ids)):
                ai, aj = emb_ids[i], emb_ids[j]
                d = distance(id_embeds[ai], id_embeds[aj])
                if d < 0.3:
                    kill, keep = max(ai, aj), min(ai, aj)
                    if kill not in id_merge:
                        id_merge[kill] = keep

        for snap in snapshots:
            if len(snap.tracks) < 2:
                continue
            for i in range(len(snap.tracks)):
                for j in range(i + 1, len(snap.tracks)):
                    a, b = snap.tracks[i], snap.tracks[j]
                    if a.id == b.id:
                        continue
                    ka = id_merge.get(a.id, a.id)
                    kb = id_merge.get(b.id, b.id)
                    if ka == kb:
                        continue
                    if iou(a.box, b.box) > 0.3:
                        kill = max(ka, kb)
                        keep = min(ka, kb)
                        if kill not in id_merge:
                            id_merge[kill] = keep

        if not id_merge:
            return

        changed = True
        while changed:
            changed = False
            for kill, keep in list(id_merge.items()):
                if keep in id_merge:
                    id_merge[kill] = id_merge[keep]
                    changed = True

        for snap in snapshots:
            for tr in snap.tracks:
                if tr.id in id_merge:
                    tr.id = id_merge[tr.id]

        for snap in snapshots:
            if len(snap.tracks) < 2:
                continue
            merged: list[TrackFrame] = []
            for tr in snap.tracks:
                found = False
                for ex in merged:
                    if ex.id == tr.id and iou(tr.box, ex.box) > 0.3:
                        if tr.conf > ex.conf:
                            ex.box = list(tr.box)
                            ex.conf = tr.conf
                            ex.landmarks = tr.landmarks
                        found = True
                        break
                if not found:
                    merged.append(tr)
            snap.tracks = merged

    # ── Pomůcky ──────────────────────────────────────────────

    def _clip_box(self, box: list[float]) -> list[float] | None:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        if x2 < 0 or y2 < 0 or x1 > self.video_width or y1 > self.video_height:
            return None
        clipped = [
            max(0.0, x1), max(0.0, y1),
            min(float(self.video_width), x2),
            min(float(self.video_height), y2),
        ]
        if clipped[2] - clipped[0] <= 0 or clipped[3] - clipped[1] <= 0:
            return None
        return clipped

    def _should_skip(
        self, snap: FrameSnapshot, track_id: int, box: list[float]
    ) -> bool:
        """True = nepridávat — místo už je pokryté detekcí nebo jiným trackem"""
        for existing in snap.tracks:
            if existing.id == track_id:
                if existing.source == "detection":
                    return True
            elif iou(box, existing.box) > 0.3:
                return True
        return False

    # ── Lineární interpolace gapu (žádná CMC, žádné landmarks) ──

    def _fill_gap(
        self,
        snapshots: list[FrameSnapshot],
        track_id: int,
        f_a: int, tr_a: TrackFrame,
        f_b: int, tr_b: TrackFrame,
    ) -> tuple[int, set[int]]:
        """Lineární interpolace boxu mezi f_a a f_b. Žádná CMC."""
        cnt = 0
        modified: set[int] = set()

        si_a = next(j for j, s in enumerate(snapshots) if s.frame_idx == f_a)
        si_b = next(j for j, s in enumerate(snapshots) if s.frame_idx == f_b)

        for si in range(si_a + 1, si_b):
            snap = snapshots[si]
            t = (si - si_a) / (si_b - si_a)  # 0..1

            box = [
                (1 - t) * tr_a.box[0] + t * tr_b.box[0],
                (1 - t) * tr_a.box[1] + t * tr_b.box[1],
                (1 - t) * tr_a.box[2] + t * tr_b.box[2],
                (1 - t) * tr_a.box[3] + t * tr_b.box[3],
            ]

            clipped = self._clip_box(box)
            if clipped is None:
                continue
            if self._should_skip(snap, track_id, clipped):
                continue

            conf = tr_a.conf * (1 - t) + tr_b.conf * t
            snap.tracks.append(TrackFrame(
                id=track_id,
                box=clipped,
                conf=conf,
                state=tr_a.state,
                source="backfilled",
                landmarks=None,
            ))
            cnt += 1
            modified.add(snap.frame_idx)

        return cnt, modified

    # ── Main ─────────────────────────────────────────────────

    def fill(self, snapshots: list[FrameSnapshot]) -> tuple[int, set[int]]:
        self._merge_duplicate_person_ids(snapshots)

        n = len(snapshots)
        if n < 2:
            return 0, set()

        # Sběr detection framů pro každé ID
        id_frames: dict[int, list[tuple[int, TrackFrame]]] = {}
        for snap in snapshots:
            for tr in snap.tracks:
                if tr.source == "detection":
                    id_frames.setdefault(tr.id, []).append((snap.frame_idx, tr))

        if not id_frames:
            return 0, set()

        # Jen tracky s dostatkem detekcí
        id_frames = {tid: f for tid, f in id_frames.items() if len(f) >= self.min_detections}
        if not id_frames:
            return 0, set()

        backfilled_count = 0
        modified_frames: set[int] = set()

        for track_id, frames in id_frames.items():
            frames.sort(key=lambda x: x[0])

            # Lineární interpolace mezi každou dvojicí detection framů
            for i in range(len(frames) - 1):
                f0, tr0 = frames[i]
                f1, tr1 = frames[i + 1]
                gap = f1 - f0
                if gap <= 1:
                    continue
                if gap > self.max_backfill_gap:
                    continue

                cnt, mod = self._fill_gap(
                    snapshots, track_id,
                    f0, tr0, f1, tr1,
                )
                backfilled_count += cnt
                modified_frames |= mod

            # Žádný backward fill — jen mezi detekcema

        # Finální cleanup duplicit
        for snap in snapshots:
            if len(snap.tracks) < 2:
                continue
            clean: list[TrackFrame] = []
            for tr in snap.tracks:
                merged = False
                for ex in clean:
                    if iou(tr.box, ex.box) > 0.3:
                        if tr.conf > ex.conf:
                            ex.box = list(tr.box)
                            ex.conf = tr.conf
                        merged = True
                        break
                if not merged:
                    clean.append(tr)
            snap.tracks = clean

        return backfilled_count, modified_frames
