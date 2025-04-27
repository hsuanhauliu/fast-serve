"""
Super minimal code to serve your model.

Usage:  uvicorn examples.minimal:app --reload
"""

import random

from fast_serve import create_app


# The function you want to serve.
def binary_guess(data):
    return {"output": random.randint(0, 1) == 1}


app = create_app(binary_guess)
