"""Request and response models of the prediction API."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_BATCH_SIZE = 500

_EXAMPLE = {
    "station_code": "UT",
    "scheduled_departure": "2026-08-14T08:12:00+02:00",
    "service_type": "Intercity",
    "company": "NS",
    "service_start_time": "2026-08-14T07:35:00+02:00",
}


class PredictionRequest(BaseModel):
    """One scheduled departure of one service from one station."""

    model_config = ConfigDict(json_schema_extra={"examples": [_EXAMPLE]})

    station_code: str = Field(
        min_length=1, max_length=10, description="Station code, e.g. UT for Utrecht Centraal"
    )
    scheduled_departure: AwareDatetime = Field(
        description="Planned departure time, RFC 3339 with a UTC offset"
    )
    service_type: str = Field(min_length=1, max_length=50, examples=["Intercity", "Sprinter"])
    company: str = Field(min_length=1, max_length=50, examples=["NS", "Arriva"])
    service_start_time: AwareDatetime | None = Field(
        default=None,
        description="Planned first departure of the service (its origin). Improves accuracy.",
    )

    @field_validator("station_code")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _start_before_departure(self) -> PredictionRequest:
        start = self.service_start_time
        if start is not None and start > self.scheduled_departure:
            raise ValueError("service_start_time must not be after scheduled_departure")
        return self


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"instances": [_EXAMPLE]}]})

    instances: list[PredictionRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class PredictionResponse(BaseModel):
    prediction_id: UUID
    probability_late: float = Field(ge=0.0, le=1.0)
    late: bool = Field(description="probability_late >= the configured decision threshold")
    model_version: str
    features_used: dict[str, Any] | None = Field(
        default=None, description="Model input features; only returned with ?debug=true"
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    features_loaded: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    decision_threshold: float
    weather_provider: str
    feature_snapshot: dict[str, Any]
    metadata: dict[str, Any]
