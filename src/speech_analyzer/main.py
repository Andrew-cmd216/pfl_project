import logging

from fastapi import FastAPI, HTTPException, UploadFile

from speech_analyzer.processing.audio_loader import load_audio
from speech_analyzer.processing.constants import MAX_FILE_SIZE
from speech_analyzer.processing.features import AudioFeatures, measure_audio

logger = logging.getLogger("uvicorn.error").getChild(__name__)

app = FastAPI(
    title="Speech Analyzer",
    description=("REST API for automated extraction and analysis of acoustic speech features."),
    version="0.1.0",
)


@app.get("/", tags=["General"])
async def root() -> dict[str, str]:
    """
    Return basic information about the application.
    """
    return {
        "name": "Speech Analyzer",
        "version": "0.1.0",
        "description": "API for acoustic speech analysis",
        "documentation": "/docs",
    }


@app.get("/health", tags=["General"])
async def health_check() -> dict[str, str]:
    """
    Check whether the API is running.
    """
    return {
        "status": "ok",
    }


@app.post("/analyze", tags=["Analysis"])
def analyze(file: UploadFile) -> AudioFeatures:
    """Validate a WAV upload and return acoustic measurements."""
    data = file.file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        logger.warning("Rejected oversized WAV upload")
        raise HTTPException(status_code=413, detail="WAV exceeds 25 MiB.")
    try:
        audio = load_audio(data)
    except ValueError as error:
        logger.warning("Rejected invalid WAV upload")
        raise HTTPException(status_code=422, detail=str(error)) from error

    features = measure_audio(audio)
    logger.info("WAV analysis completed")
    return features
