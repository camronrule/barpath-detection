from fastapi import FastAPI
from routers import yolo
import os
from uvicorn import Server, Config
from starlette.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(yolo.router)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    server = Server(Config(app, host="0.0.0.0", port=port, lifespan="on"))
    server.run()
