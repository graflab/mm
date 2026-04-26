"""Synthesize a 2-bar industrial drum loop at 124 BPM.

Mood: cold, mechanical, heavy.

Layers
------
* Kick   : deep, distorted sine-wave kick on every quarter note.
* Snare  : sharp, metallic crack on beats 2 and 4.
* Hats   : shuffling 16th-note pattern with varying velocity (swing).
* Hiss   : subtle white-noise burst every 8th note.

Output: ``industrial_loop.wav`` (16-bit PCM, 44.1 kHz, mono).

Run::

    python industrial_loop.py
"""

import numpy as np
from scipy.io import wavfile

# --------------------------------------------------------------------------- #
# Global timing / format
# --------------------------------------------------------------------------- #
SR = 44_100               # sample rate (Hz)
BPM = 124
BEATS_PER_BAR = 4
BARS = 2
SEC_PER_BEAT = 60.0 / BPM
TOTAL_BEATS = BEATS_PER_BAR * BARS
TOTAL_SECONDS = TOTAL_BEATS * SEC_PER_BEAT
TOTAL_SAMPLES = int(np.ceil(TOTAL_SECONDS * SR))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _t(n_samples: int) -> np.ndarray:
    """Return a time vector of ``n_samples`` samples in seconds."""
    return np.arange(n_samples) / SR


def _exp_env(n_samples: int, tau: float) -> np.ndarray:
    """Exponential decay envelope with time constant ``tau`` (seconds)."""
    return np.exp(-_t(n_samples) / tau)


def _mix(buffer: np.ndarray, sound: np.ndarray, start_sample: int, gain: float = 1.0) -> None:
    """Add ``sound`` into ``buffer`` at ``start_sample`` (clipped to length)."""
    end = start_sample + len(sound)
    if start_sample >= len(buffer):
        return
    if end > len(buffer):
        sound = sound[: len(buffer) - start_sample]
        end = len(buffer)
    buffer[start_sample:end] += gain * sound


# --------------------------------------------------------------------------- #
# Drum voices
# --------------------------------------------------------------------------- #
def kick(duration: float = 0.55) -> np.ndarray:
    """Deep, distorted sine kick.

    A sine oscillator whose frequency drops exponentially from ~120 Hz down to
    ~40 Hz, shaped by a fast amplitude envelope and pushed through a ``tanh``
    waveshaper for grit / weight.
    """
    n = int(duration * SR)
    t = _t(n)

    # Pitch envelope: 120 Hz -> 40 Hz, fast drop.
    f0, f1 = 120.0, 40.0
    pitch_tau = 0.06
    inst_freq = f1 + (f0 - f1) * np.exp(-t / pitch_tau)
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / SR

    # Click transient (very short high sine) for the attack snap.
    click = np.sin(2.0 * np.pi * 1800.0 * t) * np.exp(-t / 0.004)

    body = np.sin(phase)

    # Amp envelope: short attack, ~0.4 s decay.
    amp = np.exp(-t / 0.18)
    amp[: int(0.003 * SR)] = np.linspace(0.0, 1.0, int(0.003 * SR))  # 3 ms attack

    sig = body * amp + 0.25 * click * np.exp(-t / 0.02)

    # Distortion: tanh waveshaping for "industrial" weight.
    sig = np.tanh(2.8 * sig)

    # Final fade-out tail to avoid click at end.
    fade = np.ones(n)
    fade_len = min(int(0.01 * SR), n)
    fade[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
    return sig * fade * 0.95


def snare(rng: np.random.Generator, duration: float = 0.28) -> np.ndarray:
    """Sharp, metallic snare crack.

    A bright noise burst (high-passed by subtraction of a smoothed copy)
    layered with several inharmonic high sine partials to give a metallic,
    bell-like "crack".
    """
    n = int(duration * SR)
    t = _t(n)

    # Bright noise (approx. high-passed by subtracting moving average).
    noise = rng.uniform(-1.0, 1.0, n)
    smooth = np.convolve(noise, np.ones(8) / 8, mode="same")
    bright_noise = noise - smooth

    # Metallic inharmonic partials.
    partial_freqs = [1850.0, 2700.0, 3550.0, 4900.0, 6300.0]
    metal = np.zeros(n)
    for f in partial_freqs:
        metal += np.sin(2.0 * np.pi * f * t + rng.uniform(0.0, 2.0 * np.pi))
    metal /= len(partial_freqs)

    # Envelopes: very fast attack, short decay; metal shorter than noise.
    env_noise = np.exp(-t / 0.07)
    env_metal = np.exp(-t / 0.04)

    sig = 0.7 * bright_noise * env_noise + 0.6 * metal * env_metal

    # Attack ramp (1 ms) to avoid DC click.
    atk = min(int(0.001 * SR), n)
    sig[:atk] *= np.linspace(0.0, 1.0, atk)

    # Mild saturation for "crack".
    sig = np.tanh(1.6 * sig)

    return sig * 0.9


def hihat(rng: np.random.Generator, duration: float = 0.08, velocity: float = 1.0) -> np.ndarray:
    """Shuffling hi-hat: short bright noise burst, scaled by velocity."""
    n = int(duration * SR)
    t = _t(n)

    noise = rng.uniform(-1.0, 1.0, n)
    # Brighten via differentiation (high-pass-ish).
    bright = np.diff(noise, prepend=noise[0])

    env = np.exp(-t / 0.02)
    sig = bright * env

    atk = min(int(0.0005 * SR), n)
    sig[:atk] *= np.linspace(0.0, 1.0, atk)

    return sig * velocity * 0.5


def hiss(rng: np.random.Generator, duration: float = 0.05) -> np.ndarray:
    """Subtle white-noise hiss burst (8th-note layer)."""
    n = int(duration * SR)
    t = _t(n)
    noise = rng.uniform(-1.0, 1.0, n)
    env = np.exp(-t / 0.018)

    atk = min(int(0.001 * SR), n)
    env[:atk] *= np.linspace(0.0, 1.0, atk)
    # Soft tail fade.
    fade = min(int(0.005 * SR), n)
    env[-fade:] *= np.linspace(1.0, 0.0, fade)

    return noise * env * 0.18


# --------------------------------------------------------------------------- #
# Sequencing
# --------------------------------------------------------------------------- #
def build_loop() -> np.ndarray:
    # Single deterministic RNG used for all stochastic voices and humanization.
    # Seed 49181 (0xC01D) chosen for reproducibility.
    rng = np.random.default_rng(49181)

    out = np.zeros(TOTAL_SAMPLES, dtype=np.float64)

    sec_per_16th = SEC_PER_BEAT / 4.0
    sec_per_8th = SEC_PER_BEAT / 2.0

    # ----- Kick: every quarter note (8 hits over 2 bars) ------------------ #
    k = kick()
    for beat in range(TOTAL_BEATS):
        start = int(beat * SEC_PER_BEAT * SR)
        _mix(out, k, start, gain=1.0)

    # ----- Snare: beats 2 and 4 of each bar (4 hits) ---------------------- #
    s = snare(rng)
    for bar in range(BARS):
        for beat_in_bar in (1, 3):  # 2 and 4 (0-indexed)
            beat = bar * BEATS_PER_BAR + beat_in_bar
            start = int(beat * SEC_PER_BEAT * SR)
            _mix(out, s, start, gain=0.85)

    # ----- Hi-hats: shuffled 16ths with velocity variation ---------------- #
    # Swing: delay every off-16th (the 2nd and 4th 16th of each beat) by
    # ``swing`` * 16th-note duration.
    swing = 0.18
    # Velocity pattern across the four 16ths within a beat: strong-weak-mid-weak.
    base_vel = np.array([1.00, 0.55, 0.78, 0.50])

    total_16ths = TOTAL_BEATS * 4
    for i in range(total_16ths):
        pos_in_beat = i % 4
        # Apply swing offset to the 2nd and 4th 16th of each beat.
        swing_offset = swing * sec_per_16th if pos_in_beat in (1, 3) else 0.0
        t_sec = i * sec_per_16th + swing_offset

        # Velocity: base pattern + small random humanization.
        vel = base_vel[pos_in_beat] * float(rng.uniform(0.88, 1.05))
        # Occasional accent every 8 sixteenths for mechanical feel.
        if i % 8 == 0:
            vel *= 1.15

        h = hihat(rng, velocity=vel)
        start = int(t_sec * SR)
        _mix(out, h, start, gain=0.7)

    # ----- Noise hiss: every 8th note ------------------------------------- #
    total_8ths = TOTAL_BEATS * 2
    for i in range(total_8ths):
        t_sec = i * sec_per_8th
        start = int(t_sec * SR)
        _mix(out, hiss(rng), start, gain=1.0)

    return out


# --------------------------------------------------------------------------- #
# Mastering / write
# --------------------------------------------------------------------------- #
def normalize(sig: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = np.max(np.abs(sig))
    if m < 1e-9:
        return sig
    return sig * (peak / m)


def main() -> None:
    loop = build_loop()
    # Soft bus saturation for cohesion.
    loop = np.tanh(1.05 * loop)
    loop = normalize(loop, peak=0.95)

    pcm = np.int16(loop * 32767)
    wavfile.write("industrial_loop.wav", SR, pcm)
    print(
        f"Wrote industrial_loop.wav  "
        f"({TOTAL_SECONDS:.3f} s, {SR} Hz, {len(pcm)} samples, {BPM} BPM, {BARS} bars)"
    )


if __name__ == "__main__":
    main()
