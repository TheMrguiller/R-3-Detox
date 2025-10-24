from pydantic import BaseModel, Field

class ParaphraseGeneration(BaseModel):
    reasoning: str = Field(..., title="The reasoning process generated")
    paraphrase: str = Field(..., title="The final paraphrase generated")

