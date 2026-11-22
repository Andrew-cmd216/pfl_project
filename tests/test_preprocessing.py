import numpy as np

from speech_analyzer.processing.audio_loader import AudioData
from speech_analyzer.processing.preprocessing import (
    convert_to_mono,
    preprocess_audio,
    resample_audio,
)


def test_convert_mono():
    samples = np.array(
        [
            [0.2, 0.4],
            [0.4, 0.6],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )

    result = convert_to_mono(samples)

    expected = np.array([0.3, 0.5, 0.7])

    np.testing.assert_allclose(result, expected)


def test_mono_signal_remains_unchanged():
    samples = np.array(
        [[0.2], [0.4], [0.6]],
        dtype=np.float32,
    )

    result = convert_to_mono(samples)

    np.testing.assert_array_equal(result, samples[:, 0])


def test_resample_audio():
    samples = np.zeros(16_000, dtype=np.float32)

    result = resample_audio(
        samples,
        original_sample_rate=16_000,
        target_sample_rate=22_050,
    )

    assert len(result) == 22_050


def test_preprocess_audio():
    audio = AudioData(
        samples=np.zeros((16_000, 2), dtype=np.float32),
        sample_rate=16_000,
        channels=2,
        duration=1.0,
        subtype="PCM_16",
    )

    result = preprocess_audio(audio)

    assert result.sample_rate == 22_050
    assert result.samples.ndim == 1
    assert len(result.samples) == 22_050
