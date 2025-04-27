from typing import Any, Callable

from fastapi import FastAPI


def create_app(predict_func: Callable, response_model=Any, endpoint_name="/predict", methods=["POST"]):
    app = FastAPI()
    app.add_api_route(endpoint_name, response_model=response_model, endpoint=predict_func, methods=methods)
    return app
