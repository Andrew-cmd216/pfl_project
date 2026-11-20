
from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from speech_analyzer.processing.audio_loader import AudioData


TARGET_SAMPLE_RATE = 22_050


@dataclass
class ProcessedAudio:
    samples: np.ndarray
    sample_rate: int


def convert_to_mono(samples: np.ndarray) -> np.ndarray:
    """
    Convert a mono or stereo signal to mono.
    """

    if samples.ndim != 2:
        raise ValueError("Audio samples must be a 2D array.")

    if samples.shape[1] == 1:
        return samples[:, 0]

    if samples.shape[1] == 2:
        return np.mean(samples, axis=1)

    raise ValueError("Only mono and stereo audio are supported.")


def resample_audio(
    samples: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    Change the sample rate of an audio signal.
    """

    if original_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive.")

    if original_sample_rate == target_sample_rate:
        return samples

    divisor = gcd(original_sample_rate, target_sample_rate)

    up = target_sample_rate // divisor
    down = original_sample_rate // divisor

    return resample_poly(samples, up, down).astype(np.float32)


def preprocess_audio(
    audio: AudioData,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> ProcessedAudio:
    """
    Convert audio to mono and resample it.
    """

    mono_samples = convert_to_mono(audio.samples)

    processed_samples = resample_audio(
        mono_samples,
        original_sample_rate=audio.sample_rate,
        target_sample_rate=target_sample_rate,
    )

    return ProcessedAudio(
        samples=processed_samples,
        sample_rate=target_sample_rate,
    )
