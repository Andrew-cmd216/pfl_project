
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import soundfile as sf


MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

MIN_SAMPLE_RATE = 16_000
MAX_SAMPLE_RATE = 48_000

MIN_DURATION = 0.5
MAX_DURATION = 120.0

SUPPORTED_SUBTYPES = {
    "PCM_16",
    "PCM_24",
    "PCM_32",
}


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    channels: int
    duration: float
    subtype: str


def load_audio(file_bytes: bytes) -> AudioData:
    """
    Load and validate a WAV audio file.

    Returns audio samples and basic metadata.
    """

    if not file_bytes:
        raise ValueError("Audio file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("Audio file exceeds the maximum size of 25 MB.")

    try:
        audio_info = sf.info(BytesIO(file_bytes))

        if audio_info.format != "WAV":
            raise ValueError("Only WAV files are supported.")

        if audio_info.subtype not in SUPPORTED_SUBTYPES:
            raise ValueError("Unsupported audio encoding.")

        if audio_info.channels not in (1, 2):
            raise ValueError("Only mono and stereo audio are supported.")

        if not MIN_SAMPLE_RATE <= audio_info.samplerate <= MAX_SAMPLE_RATE:
            raise ValueError(
                "Sample rate must be between 16 kHz and 48 kHz."
            )

        duration = audio_info.duration

        if not MIN_DURATION <= duration <= MAX_DURATION:
            raise ValueError(
                "Audio duration must be between 0.5 and 120 seconds."
            )

        samples, sample_rate = sf.read(
            BytesIO(file_bytes),
            dtype="float32",
            always_2d=True,
        )

    except (RuntimeError, OSError) as error:
        raise ValueError("Unable to read the audio file.") from error

    if samples.size == 0:
        raise ValueError("Audio file contains no samples.")

    if not np.isfinite(samples).all():
        raise ValueError("Audio contains invalid numerical values.")

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=audio_info.channels,
        duration=duration,
        subtype=audio_info.subtype,
    )
