from pydantic import BaseModel, Field

class IncorrectParaphraseGeneration(BaseModel):
    incorrect_paraphrase: str
    criteria_number: int = Field(..., description="The criteria number followed to generate the incorrect paraphrase")
