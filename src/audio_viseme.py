# ==============================
# File: src/audio_viseme.py
# ==============================
import numpy as np


class VisemeMapper:
    """Map raw audio int16 chunks → smooth [0..1] mouth openness values.
    This uses RMS energy + simple attack/release smoothing suitable for VTuber A/I/U/E/O blendshapes.
    """

    def __init__(self, attack=0.08, release=0.15, gain=1.8, floor=0.02):
        self.attack = attack
        self.release = release
        self.gain = gain
        self.floor = floor
        self._env = 0.0

    def step(self, pcm_i16: np.ndarray) -> float:
        if pcm_i16.size == 0:
            return max(0.0, self._env * (1.0 - self.release))
        rms = np.sqrt(np.mean((pcm_i16.astype(np.float32) / 32768.0) ** 2))
        target = max(0.0, min(1.0, self.gain * max(0.0, rms - self.floor) / (1.0 - self.floor)))
        if target > self._env:
            # attack
            self._env = self._env + (target - self._env) * self.attack
        else:
            # release
            self._env = self._env + (target - self._env) * self.release
        return float(self._env)

    def to_vowels(self, openness: float) -> dict:
        """Optional: split openness into simplistic A/I/U/E/O weights.
        Very rough heuristic so the mouth shapes vary a bit.
        """
        a = openness
        e = max(0.0, (openness - 0.3) * 1.2)
        i = max(0.0, (0.7 - abs(openness - 0.7)) * 1.3)
        o = max(0.0, (openness - 0.5) * 0.8)
        u = max(0.0, (0.4 - openness) * 1.0)
        # Normalize to [0..1]
        def clamp(x): return max(0.0, min(1.0, x))
        return {"aa": clamp(a), "ee": clamp(e), "ih": clamp(i), "oh": clamp(o), "ou": clamp(u)}
