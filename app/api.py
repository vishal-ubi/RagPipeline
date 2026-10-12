from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import generate_answer


app = FastAPI()


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: AskRequest):
    return {
        "question": request.question,
        **generate_answer(request.question),
    }
