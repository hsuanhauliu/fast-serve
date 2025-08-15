import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from fastapi import FastAPI
from pydantic import BaseModel

# Get a logger for this module
logger = logging.getLogger(__name__)


@dataclass
class FastAPIConfig:
    """
    A simple configuration structure for the FastAPI application.
    """
    enable_docs: bool = True


def create_app(
    predict_func: Callable[..., Any] | Callable[..., Coroutine[Any, Any, Any]],
    response_model: Any = None,
    http_endpoint: str | None = "/predict",
    websocket_endpoint: str | None = None,
    http_methods: list[str] = ["POST"],
    config: FastAPIConfig | None = None,
) -> FastAPI:
    """
    Creates a FastAPI application for machine learning model inference.

    This function sets up HTTP and WebSocket endpoints for your prediction function.
    It supports both synchronous and asynchronous prediction functions.

    Args:
        predict_func: The function that takes input and returns a prediction.
                      The first argument of this function should be type-hinted
                      with a Pydantic model for WebSocket support.
        response_model: The Pydantic model for the HTTP response body.
        http_endpoint: The path for the HTTP endpoint. If None, no HTTP endpoint is created.
        websocket_endpoint: The path for the WebSocket endpoint. If None, no WebSocket endpoint is created.
                            Requires the `websockets` library to be installed.
        http_methods: A list of allowed HTTP methods for the HTTP endpoint.
        config: An optional configuration object for the FastAPI app itself.

    Returns:
        A FastAPI application instance.
    """
    if http_endpoint is None and websocket_endpoint is None:
        raise ValueError(
            "At least one endpoint (http_endpoint or websocket_endpoint) must be provided."
        )

    # Use default config if none is provided
    if config is None:
        config = FastAPIConfig()

    # Instantiate the FastAPI app, enabling/disabling docs based on config
    app = FastAPI(
        docs_url="/docs" if config.enable_docs else None,
        redoc_url="/redoc" if config.enable_docs else None,
    )

    # Register the HTTP endpoint if one is provided
    if http_endpoint:
        app.add_api_route(
            http_endpoint,
            endpoint=predict_func,
            response_model=response_model,
            methods=http_methods,
        )

    # Register the WebSocket endpoint if one is provided
    if websocket_endpoint:
        try:
            from fastapi import WebSocket, WebSocketDisconnect
        except ImportError:
            raise ImportError(
                "The 'websockets' library is required for WebSocket support. "
                "Please install it with: pip install websockets"
            )

        input_model = None
        try:
            sig = inspect.signature(predict_func)
            if sig.parameters:
                first_param_key = next(iter(sig.parameters))
                first_param = sig.parameters[first_param_key]
                if (
                    first_param.annotation is not inspect.Parameter.empty
                    and issubclass(first_param.annotation, BaseModel)
                ):
                    input_model = first_param.annotation
        except (ValueError, TypeError):
            pass

        async def websocket_endpoint_func(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_json()

                    input_data = data
                    if input_model:
                        try:
                            if hasattr(input_model, 'model_validate'):
                                input_data = input_model.model_validate(data)
                            else:
                                input_data = input_model.parse_obj(data)
                        except Exception as validation_error:
                            await websocket.send_json({"error": f"Invalid input: {validation_error}"})
                            continue

                    if asyncio.iscoroutinefunction(predict_func):
                        prediction = await predict_func(input_data)
                    else:
                        prediction = predict_func(input_data)

                    output_data = prediction
                    if isinstance(prediction, BaseModel):
                        if hasattr(prediction, 'model_dump'):
                            output_data = prediction.model_dump()
                        else:
                            output_data = prediction.dict()

                    await websocket.send_json(output_data)
            except WebSocketDisconnect:
                logger.info("Client disconnected from WebSocket.")
            except Exception as e:
                logger.error(f"An error occurred in the WebSocket: {e}", exc_info=True)
                await websocket.send_json({"error": str(e)})

        app.add_api_websocket_route(websocket_endpoint, websocket_endpoint_func)

    return app


# --- Example Usage (remains the same) ---
if __name__ == "__main__":
    import uvicorn

    # Configure basic logging for the example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    class ModelInput(BaseModel):
        text: str

    class ModelOutput(BaseModel):
        prediction: str

    async def async_predict(data: ModelInput) -> ModelOutput:
        logger.info(f"Received async request with text: '{data.text}'")
        await asyncio.sleep(1)
        return ModelOutput(prediction=f"Async prediction for: {data.text}")

    # Example of disabling the documentation endpoints
    api_config = FastAPIConfig(enable_docs=False)

    app = create_app(
        predict_func=async_predict,
        response_model=ModelOutput,
        http_endpoint="/predict",
        websocket_endpoint="/ws",
        config=api_config,
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
