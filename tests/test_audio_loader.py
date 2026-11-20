
import io

import numpy as np
import soundfile as sf

from speech_analyzer.processing.audio_loader import load_audio


def create_test_wav() -> bytes:
    sample_rate = 16_000
    duration = 1.0

    samples = np.zeros(sample_rate, dtype=np.float32)

    buffer = io.BytesIO()

    sf.write(
        buffer,
        samples,
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )

    return buffer.getvalue()


def test_load_audio():
    audio_bytes = create_test_wav()

    audio = load_audio(audio_bytes)

    assert audio.sample_rate == 16_000
    assert audio.channels == 1
    assert audio.duration == 1.0
    assert audio.samples.shape == (16_000, 1)
    assert audio.subtype == "PCM_16"


def test_empty_file():
    try:
        load_audio(b"")
    except ValueError as error:
        assert "empty" in str(error).lower()
    else:
        assert False, "Expected ValueError"
