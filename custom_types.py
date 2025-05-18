from pydantic import BaseModel

class UserPreferences(BaseModel):
    answer_length: str = "medium"  # can be short, medium, or long
    expertise_level: str = "intermediate"  # can be naive, intermediate, or expert 