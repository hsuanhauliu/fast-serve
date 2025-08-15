import asyncio
import inspect
from typing import Any, Callable, Coroutine

from fastapi import FastAPI
from pydantic import BaseModel


def create_app(
    predict_func: Callable[..., Any] | Callable[..., Coroutine[Any, Any, Any]],
    response_model: Any = None,
    http_endpoint: str | None = "/predict",
    websocket_endpoint: str | None = "/ws",
    http_methods: list[str] = ["POST"],
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

    Returns:
        A FastAPI application instance.
    """
    app = FastAPI()

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
        from fastapi import WebSocket, WebSocketDisconnect

        # --- FIX: Inspect the predict_func to find the input Pydantic model ---
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
            # Ignore if signature can't be determined or annotation isn't a class
            pass
        # --- END FIX ---

        async def websocket_endpoint_func(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_json()

                    # --- FIX: Parse the incoming dict into the Pydantic model ---
                    input_data = data
                    if input_model:
                        try:
                            # model_validate is for Pydantic v2, parse_obj is for v1
                            if hasattr(input_model, 'model_validate'):
                                input_data = input_model.model_validate(data)
                            else:
                                input_data = input_model.parse_obj(data)
                        except Exception as validation_error:
                            await websocket.send_json({"error": f"Invalid input: {validation_error}"})
                            continue
                    # --- END FIX ---

                    if asyncio.iscoroutinefunction(predict_func):
                        prediction = await predict_func(input_data)
                    else:
                        prediction = predict_func(input_data)

                    # --- FIX: Convert Pydantic output model to dict for JSON ---
                    output_data = prediction
                    if isinstance(prediction, BaseModel):
                        # model_dump is for Pydantic v2, dict is for v1
                        if hasattr(prediction, 'model_dump'):
                            output_data = prediction.model_dump()
                        else:
                            output_data = prediction.dict()
                    # --- END FIX ---

                    await websocket.send_json(output_data)
            except WebSocketDisconnect:
                print("Client disconnected")
            except Exception as e:
                print(f"An error occurred: {e}")
                await websocket.send_json({"error": str(e)})

        app.add_api_websocket_route(websocket_endpoint, websocket_endpoint_func)

    return app


# --- Example Usage (remains the same) ---
if __name__ == "__main__":
    import uvicorn

    class ModelInput(BaseModel):
        text: str

    class ModelOutput(BaseModel):
        prediction: str

    async def async_predict(data: ModelInput) -> ModelOutput:
        print(f"Received async request: {data.text}")
        await asyncio.sleep(1)
        return ModelOutput(prediction=f"Async prediction for: {data.text}")

    app = create_app(
        predict_func=async_predict,
        response_model=ModelOutput,
        http_endpoint="/predict",
        websocket_endpoint="/ws",
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
