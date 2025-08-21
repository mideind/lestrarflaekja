from typing import Union

from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve raw HTML content from frontend/
app.mount("/", StaticFiles(directory="frontend"), name="frontend")


@app.post("/analyze")
def analyze_text(textInput: str = Form(...)):
    # Here you would add your text analysis logic
    # return {"analyzed_text": textInput}
    return "<p>Analyzed text: {}</p>".format(textInput)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)