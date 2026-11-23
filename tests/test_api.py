from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from speech_analyzer.main import app

client = TestClient(app)


@pytest.mark.parametrize(
    ("tone_frames", "silence_frames", "rms", "pause_count", "pause_seconds"),
    [
        (8000, 8000, 0.5 / np.sqrt(2), 1, 0.5),
        (4800, 3200, 0.5 * np.sqrt(0.6), 1, 0.2),
        (5120, 2880, 0.4, 0, 0.0),
    ],
)
def test_analyze_wav(tone_frames, silence_frames, rms, pause_count, pause_seconds):
    samples = np.concatenate((np.full(tone_frames, 0.5), np.zeros(silence_frames)))
    buffer = BytesIO()
    sf.write(buffer, samples, 16_000, format="WAV", subtype="PCM_16")

    response = client.post("/analyze", files={"file": ("sample.wav", buffer.getvalue())})

    assert response.status_code == 200
    assert response.json() == {
        "duration_seconds": (tone_frames + silence_frames) / 16_000,
        "rms": pytest.approx(rms),
        "pause_count": pause_count,
        "pause_seconds": pause_seconds,
    }


def test_analyze_invalid_wav():
    response = client.post("/analyze", files={"file": ("bad.wav", b"invalid audio")})

    assert response.status_code == 422
    assert response.json()["detail"] == "Unable to read the audio file."


def test_analyze_oversized_wav(monkeypatch):
    monkeypatch.setattr("speech_analyzer.main.MAX_FILE_SIZE", 4)
    response = client.post("/analyze", files={"file": ("big.wav", b"12345")})

    assert response.status_code == 413


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
