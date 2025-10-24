from pydantic import BaseModel, Field

class ReasoningCleaning(BaseModel):
    reasoning: str = Field(description="The completely clean, extracted reasoning in English.")