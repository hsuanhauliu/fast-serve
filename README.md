# Fast Serve

One-line code to spin up a HTTP inference server for your ML models.

In your Python code:

```python
# return a FastAPI app
app = create_app(binary_guess, response_model=Response)
```

To start the server:

```bash
uvicorn examples.request_parsing:app --reload
```

This simple library is built on top of [FastAPI](https://fastapi.tiangolo.com/), and is easily customizable.

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
