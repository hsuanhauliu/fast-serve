# Fast Serve

A lightweight library to quickly create a FastAPI application for serving machine learning models.

This simple library is built on top of [FastAPI](https://fastapi.tiangolo.com/), and is easily customizable. It supports standard HTTP endpoints, WebSockets, and both synchronous and asynchronous prediction functions.

```python
# in main.py
# returns a FastAPI app
app = create_app(
        predict_func=inference_func,
        response_model=ModelOutput,
        http_endpoint="/predict",
        websocket_endpoint="/ws/predict",
    )
```

Run your application using an ASGI server like `uvicorn`.

```bash
uvicorn main:app --reload
```

This will start an app that routes HTTP requests to `/predict` and Websocket requests to `/ws/predict`.

Example:

```
http://127.0.0.1/predict
ws://127.0.0.1/ws/predict
```

## Dev

[Poetry](https://python-poetry.org/docs/basic-usage/) is used to manage this Python project.

To run the `examples/`, do:

```bash
poetry install              # install all dependencies
poetry run uvicorn examples.minimal:app --reload   # run example
```

### Tests

Run:

```bash
poetry install
poetry run pytest
```
