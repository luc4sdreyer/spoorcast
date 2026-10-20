"""FastAPI application. Run with ``spoorcast serve`` or
``uvicorn --factory spoorcast.serving.app:create_app``."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status

from spoorcast import __version__
from spoorcast.config import Settings
from spoorcast.serving.logging import RequestIdMiddleware, configure_logging, request_id_var
from spoorcast.serving.predict import UnknownStationError, predict
from spoorcast.serving.providers import (
    FeatureLookup,
    FixedWeatherProvider,
    HistoricalWeatherProvider,
    InferenceLogger,
    JsonLinesInferenceLogger,
    LoadedModel,
    LocalModelProvider,
    ModelProvider,
    NotReadyError,
    OpenMeteoForecastProvider,
    SnapshotFeatureLookup,
    WeatherProvider,
)
from spoorcast.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ReadyResponse,
)

logger = logging.getLogger("spoorcast.api")


def weather_provider_from_settings(settings: Settings) -> WeatherProvider:
    if settings.weather_provider == "historical":
        return HistoricalWeatherProvider(settings.weather_raw_dir)
    if settings.weather_provider == "fixed":
        return FixedWeatherProvider()
    return OpenMeteoForecastProvider()


def create_app(
    settings: Settings | None = None,
    *,
    model_provider: ModelProvider | None = None,
    feature_lookup: FeatureLookup | None = None,
    weather_provider: WeatherProvider | None = None,
    inference_logger: InferenceLogger | None = None,
) -> FastAPI:
    """Build the app. Each dependency defaults to its local implementation from settings."""
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # A failed load leaves the app running but not ready (/ready reports it).
        for name, dependency in (
            ("model", app.state.model_provider),
            ("feature snapshot", app.state.feature_lookup),
        ):
            try:
                dependency.load()
            except Exception:
                logger.exception("Could not load %s", name)
        yield

    app = FastAPI(
        title="Spoorcast",
        version=__version__,
        description=(
            "Probability that a Dutch passenger train departs at least the configured number "
            "of minutes late. Train data: Rijden de Treinen (CC BY 4.0). "
            "Weather data by Open-Meteo.com (CC BY 4.0)."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_provider = model_provider or LocalModelProvider(settings.model_dir)
    app.state.feature_lookup = feature_lookup or SnapshotFeatureLookup(
        settings.resolved_snapshot_dir
    )
    app.state.weather_provider = weather_provider or weather_provider_from_settings(settings)
    app.state.inference_logger = inference_logger or JsonLinesInferenceLogger(
        settings.inference_log_path
    )

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
        return response

    # Added last so it runs first: everything below (including the access log) sees the id.
    app.add_middleware(RequestIdMiddleware)
    _add_routes(app)
    return app


# --- Dependencies --------------------------------------------------------------------------------


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_model_provider(request: Request) -> ModelProvider:
    provider: ModelProvider = request.app.state.model_provider
    return provider


def get_feature_lookup(request: Request) -> FeatureLookup:
    lookup: FeatureLookup = request.app.state.feature_lookup
    return lookup


def get_weather_provider(request: Request) -> WeatherProvider:
    provider: WeatherProvider = request.app.state.weather_provider
    return provider


def get_inference_logger(request: Request) -> InferenceLogger:
    inference_logger: InferenceLogger = request.app.state.inference_logger
    return inference_logger


def get_loaded_model(
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    feature_lookup: Annotated[FeatureLookup, Depends(get_feature_lookup)],
) -> LoadedModel:
    if not (provider.is_ready() and feature_lookup.is_ready()):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model or features not loaded")
    return provider.get()


SettingsDep = Annotated[Settings, Depends(get_settings)]
ModelDep = Annotated[LoadedModel, Depends(get_loaded_model)]
FeatureLookupDep = Annotated[FeatureLookup, Depends(get_feature_lookup)]
WeatherDep = Annotated[WeatherProvider, Depends(get_weather_provider)]
InferenceLoggerDep = Annotated[InferenceLogger, Depends(get_inference_logger)]
DebugFlag = Annotated[bool, Query(description="Include the model input features")]


# --- Routes --------------------------------------------------------------------------------------


def _add_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Liveness: the process is up."""
        return HealthResponse(status="ok")

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["health"],
        responses={503: {"model": ReadyResponse, "description": "Not ready"}},
    )
    def ready(
        response: Response,
        provider: Annotated[ModelProvider, Depends(get_model_provider)],
        feature_lookup: FeatureLookupDep,
    ) -> ReadyResponse:
        """Readiness: the model and the feature snapshot are loaded."""
        body = ReadyResponse(
            ready=provider.is_ready() and feature_lookup.is_ready(),
            model_loaded=provider.is_ready(),
            features_loaded=feature_lookup.is_ready(),
        )
        if not body.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return body

    @app.get("/v1/model", response_model=ModelInfoResponse, tags=["model"])
    def model_info(
        model: ModelDep,
        settings: SettingsDep,
        feature_lookup: FeatureLookupDep,
        weather: WeatherDep,
    ) -> ModelInfoResponse:
        """Metadata of the loaded model (feature list, training window, metrics)."""
        return ModelInfoResponse(
            model_version=model.version,
            decision_threshold=settings.decision_threshold,
            weather_provider=weather.name,
            feature_snapshot=feature_lookup.info(),
            metadata=model.model.metadata,
        )

    def _predict(
        requests: list[PredictionRequest],
        model: LoadedModel,
        settings: Settings,
        feature_lookup: FeatureLookup,
        weather: WeatherProvider,
        inference_logger: InferenceLogger,
        debug: bool,
    ) -> list[PredictionResponse]:
        try:
            return predict(
                requests,
                model=model,
                feature_lookup=feature_lookup,
                weather_provider=weather,
                inference_logger=inference_logger,
                decision_threshold=settings.decision_threshold,
                request_id=request_id_var.get(),
                include_features=debug,
            )
        except UnknownStationError as exc:
            detail: Any = {"message": str(exc), "unknown_station_codes": exc.codes}
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail) from exc
        except NotReadyError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    @app.post(
        "/v1/predictions",
        response_model=PredictionResponse,
        response_model_exclude_none=True,
        tags=["predictions"],
    )
    def predict_one(
        body: PredictionRequest,
        model: ModelDep,
        settings: SettingsDep,
        feature_lookup: FeatureLookupDep,
        weather: WeatherDep,
        inference_logger: InferenceLoggerDep,
        debug: DebugFlag = False,
    ) -> PredictionResponse:
        """Probability that one departure is late."""
        return _predict([body], model, settings, feature_lookup, weather, inference_logger, debug)[
            0
        ]

    @app.post(
        "/v1/predictions/batch",
        response_model=BatchPredictionResponse,
        response_model_exclude_none=True,
        tags=["predictions"],
    )
    def predict_batch(
        body: BatchPredictionRequest,
        model: ModelDep,
        settings: SettingsDep,
        feature_lookup: FeatureLookupDep,
        weather: WeatherDep,
        inference_logger: InferenceLoggerDep,
        debug: DebugFlag = False,
    ) -> BatchPredictionResponse:
        """Probabilities for up to 500 departures, in request order."""
        predictions = _predict(
            body.instances, model, settings, feature_lookup, weather, inference_logger, debug
        )
        return BatchPredictionResponse(predictions=predictions)
