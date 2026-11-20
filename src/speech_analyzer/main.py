from fastapi import FastAPI

app = FastAPI(
    title="Speech Analyzer",
    description=(
        "REST API for automated extraction and analysis "
        "of acoustic speech features."
    ),
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
