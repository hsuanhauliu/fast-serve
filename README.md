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
poetry env use python3.9    # create a virtual environment
eval $(poetry env activate) # activate virtual environment
poetry install              # install all dependencies
uvicorn examples.minimal:app --reload   # run example
```

Then, <http://127.0.0.1:8000/docs> should be accessible.

### Tests

Run:

```bash
poetry install
poetry run pytest
```
