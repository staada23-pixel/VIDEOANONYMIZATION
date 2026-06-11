"""
Vygeneruje 5s demo video s pohybujici se osobou.

Spusteni:
    python tests/make_demo_video.py
Vystup:
    tests/demo.mp4
"""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    W, H = 640, 480
    fps = 20
    n = fps * 5  # 5 sekund

    out = os.path.join(HERE, "demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, fps, (W, H))

    rng = np.random.default_rng(42)
    # Staticky "svet" — podlaha, zdi
    floor_color = (60, 50, 40)
    wall_color = (90, 80, 70)

    for t in range(n):
        frame = np.full((H, W, 3), wall_color[0], dtype=np.uint8)
        # Podlaha
        cv2.rectangle(frame, (0, H - 100), (W, H), floor_color, -1)
        # Trochu textury
        noise = rng.integers(0, 15, (H, W), dtype=np.uint8)
        frame = np.clip(frame.astype(int) + noise[..., None], 0, 255).astype(np.uint8)

        # Osoba — pohyb po sinusoidach
        cx = W // 2 + 200 * np.sin(2 * np.pi * t / (fps * 2))
        cy = H // 2 + 50 * np.cos(2 * np.pi * t / fps)
        cx, cy = int(cx), int(cy)

        # Hlava
        head_r = 18
        cv2.circle(frame, (cx, cy - 60), head_r, (220, 210, 200), -1)
        # Tělo
        cv2.rectangle(frame, (cx - 20, cy - 40), (cx + 20, cy + 50), (180, 50, 50), -1)
        # Nohy
        cv2.rectangle(frame, (cx - 18, cy + 50), (cx - 4, cy + 110), (50, 50, 150), -1)
        cv2.rectangle(frame, (cx + 4, cy + 50), (cx + 18, cy + 110), (50, 50, 150), -1)
        # Ruce
        cv2.rectangle(frame, (cx - 35, cy - 35), (cx - 20, cy + 30), (200, 180, 150), -1)
        cv2.rectangle(frame, (cx + 20, cy - 35), (cx + 35, cy + 30), (200, 180, 150), -1)

        # Popisky
        cv2.putText(frame, f"Demo frame {t+1}/{n}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"Demo video: {out}  ({os.path.getsize(out)/1024:.0f} KB)")
    print(f"Delka: {n/fps:.1f}s, {W}x{H}@{fps}fps")


if __name__ == "__main__":
    main()
