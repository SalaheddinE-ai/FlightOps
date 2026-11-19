from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH = 1000


class MachineObservation(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {
            "type": "M", "air_temperature": 298.1, "process_temperature": 308.6,
            "rotational_speed": 1551, "torque": 42.8, "tool_wear": 120}},
        extra="forbid",
    )
    type: Literal["L", "M", "H"] = Field(description="Product quality variant")
    air_temperature: float = Field(gt=0, description="Air temperature [K]")
    process_temperature: float = Field(gt=0, description="Process temperature [K]")
    rotational_speed: float = Field(gt=0, description="Rotational speed [rpm]")
    torque: float = Field(gt=0, description="Torque [Nm]")
    tool_wear: float = Field(ge=0, description="Tool wear [min]")


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class Explanation(BaseModel):
    base_value: float
    unit: str
    contributions: list[FeatureContribution]


class PredictionResponse(BaseModel):
    request_id: str
    prediction: int
    failure_probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    model_version: str
    threshold: float
    timestamp: str
    explanation: Explanation | None = None


class BatchRequest(BaseModel):
    observations: list[MachineObservation] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    count: int
    model_version: str
    predictions: list[PredictionResponse]


class ErrorResponse(BaseModel):
    error: str
    detail: str | list | dict | None = None
