"""Basic amplitude and silence measurements for validated WAV audio."""

from dataclasses import dataclass

import numpy as np

from speech_analyzer.processing.models import AudioData
from speech_analyzer.processing.preprocessing import convert_to_mono


@dataclass
class AudioFeatures:
    duration_seconds: float
    rms: float
    pause_count: int
    pause_seconds: float


def measure_audio(audio: AudioData) -> AudioFeatures:
    """Measure mono RMS and quiet runs of at least 200 ms in 20 ms windows."""
    samples = convert_to_mono(audio.samples)
    window = audio.sample_rate // 50
    minimum_pause = audio.sample_rate // 5
    pause_count = quiet_samples = pause_samples = 0
    for start in range(0, samples.size, window):
        segment = samples[start : start + window]
        if np.sqrt(np.mean(segment**2)) < 0.01:
            quiet_samples += segment.size
        else:
            if quiet_samples >= minimum_pause:
                pause_count += 1
                pause_samples += quiet_samples
            quiet_samples = 0

    if quiet_samples >= minimum_pause:
        pause_count += 1
        pause_samples += quiet_samples
    return AudioFeatures(
        duration_seconds=samples.size / audio.sample_rate,
        rms=float(np.sqrt(np.mean(samples**2))),
        pause_count=pause_count,
        pause_seconds=pause_samples / audio.sample_rate,
    )
