"""针灸真实世界因果推断与个体化穴位组合推荐"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.protocol import router as protocol_router
from backend.api.analysis import router as analysis_router

app = FastAPI(
    title="TCM-Acupuncture-Causal",
    version="0.2.0",
    description="针灸真实世界因果推断与个体化穴位组合推荐 — Causal inference, uplift modeling, and association rule mining",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(protocol_router)
app.include_router(analysis_router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "TCM-Acupuncture-Causal",
        "version": "0.2.0",
        "routers": ["/protocol", "/analysis"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)
