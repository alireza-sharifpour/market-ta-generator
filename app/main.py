from fastapi import FastAPI

app = FastAPI(
    title="Market TA Generator",
    description="A service that generates technical analysis for cryptocurrency pairs using LBank data and LLM",
    version="0.1.0",
)


@app.get("/")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "ok", "service": "Market TA Generator"}
