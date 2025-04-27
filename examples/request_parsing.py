"""
Add custom pydantic models to auto-parse request input and output.

The example here takes a request with an image and a text input and return a boolean response.

Usage:  uvicorn examples.request_parsing:app --reload
"""

import random

from pydantic import BaseModel

from fast_serve import create_app


# Input/request format.
class Request(BaseModel):
    image: str
    prompt: str


# Output/response format.
class Response(BaseModel):
    prediction: bool


# Your prediction function.
def binary_guess(data: Request):
    return {"prediction": random.randint(0, 1) == 1}


app = create_app(binary_guess, response_model=Response)
