from pydantic import BaseModel


class AskRequest(BaseModel):

    question: str


class Source(BaseModel):

    paper: str

    page: int

    chunk_id: int


class AskResponse(BaseModel):

    question: str

    answer: str

    sources: list[Source]