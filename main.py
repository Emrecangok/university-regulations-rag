from fastapi import FastAPI
from routers import ask


app = FastAPI(title = "GaziMind API")


app.include_router(ask.router)


@app.get("/healt")
def healt():
    return {"status":"ok"}


