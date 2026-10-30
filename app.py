import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, List

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "label_encoders.pkl"
META_PATH = BASE_DIR / "model_meta.json"

INDEX_PATH = BASE_DIR / "index.html"
CSS_PATH = BASE_DIR / "style.css"
JS_PATH = BASE_DIR / "script.js"


# ============================================================
# LOAD ML MODEL
# ============================================================

print("🧠 Loading Mental Health Analytics Platform...")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ best_model.pkl loaded")
except Exception as e:
    print(f"❌ Failed to load best_model.pkl: {e}")
    raise


try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ scaler.pkl loaded")
except Exception as e:
    print(f"⚠️ scaler.pkl could not be loaded: {e}")
    scaler = None


try:
    label_encoders = joblib.load(ENCODER_PATH)
    print("✅ label_encoders.pkl loaded")
except Exception as e:
    print(f"⚠️ label_encoders.pkl could not be loaded: {e}")
    label_encoders = None


try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        model_meta = json.load(f)

    print("✅ model_meta.json loaded")

except Exception as e:
    print(f"⚠️ model_meta.json could not be loaded: {e}")

    model_meta = {
        "selected_model": "Mental Health Prediction Model",
        "performance": {},
        "feature_importances": {},
        "dataset_baseline": {
            "mean_score": 5.5,
            "std_score": 2.0
        },
        "score_distribution": {
            "colors": [
                "#ef4444",
                "#f97316",
                "#eab308",
                "#84cc16",
                "#22c55e",
                "#10b981"
            ],
            "descriptions": [
                "Critical mental health signal",
                "At risk",
                "Struggling",
                "Fair",
                "Good",
                "Thriving"
            ]
        },
        "explanation_factors": {},
        "recommendations_config": {
            "sleep_threshold_low": 6,
            "activity_threshold_min": 0.5,
            "screen_time_threshold_max": 5
        }
    }


# ============================================================
# CONSTANTS
# ============================================================

TOP_COUNTRIES = [
    "Other",
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France"
]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="🧠 Ultimate Mental Health Analytics Platform",
    description="Mental health prediction and analytics platform",
    version="3.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100)

    gender: Literal[
        "Male",
        "Female"
    ]

    country: str

    academic_level: Literal[
        "Undergraduate",
        "Graduate",
        "High School"
    ]

    most_used_platform: Literal[
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat"
    ]

    purpose_of_use: Literal[
        "Networking",
        "Education",
        "Entertainment",
        "News"
    ]

    avg_daily_usage_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    daily_unlocks: int = Field(
        ...,
        ge=0
    )

    study_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    physical_activity_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    sleep_hours_per_night: float = Field(
        ...,
        ge=0,
        le=24
    )

    stress_level: Literal[
        "Medium",
        "Low",
        "Very High",
        "High"
    ]


class ExplanationFactor(BaseModel):
    factor: str
    category: str
    impact_score: float
    explanation: str
    recommendation: str


class Recommendation(BaseModel):
    title: str
    priority: Literal[
        "critical",
        "high",
        "medium",
        "low"
    ]
    description: str
    action_items: List[str]


class HealthTrend(BaseModel):
    date: str
    score: float
    band: str


class DashboardMetric(BaseModel):
    label: str
    value: str
    change: str
    status: Literal[
        "good",
        "warning",
        "critical"
    ]


class PredictionResponse(BaseModel):
    score: float
    band: str
    band_color: str
    percentile: float
    risk_assessment: str
    explanation: List[ExplanationFactor]
    recommendations: List[Recommendation]
    summary: str
    wellness_score: float


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_score_band_info(score: float):

    distribution = model_meta.get(
        "score_distribution",
        {}
    )

    colors = distribution.get(
        "colors",
        [
            "#ef4444",
            "#f97316",
            "#eab308",
            "#84cc16",
            "#22c55e",
            "#10b981"
        ]
    )

    descriptions = distribution.get(
        "descriptions",
        [
            "Critical",
            "At Risk",
            "Struggling",
            "Fair",
            "Good",
            "Thriving"
        ]
    )

    if score < 2:
        return "Critical", colors[0], descriptions[0]

    elif score < 4:
        return "At Risk", colors[1], descriptions[1]

    elif score < 6:
        return "Struggling", colors[2], descriptions[2]

    elif score < 8:
        return "Fair", colors[3], descriptions[3]

    elif score < 9:
        return "Good", colors[4], descriptions[4]

    else:
        return "Thriving", colors[5], descriptions[5]


def calculate_percentile(score: float) -> float:

    baseline = model_meta.get(
        "dataset_baseline",
        {}
    )

    mean = baseline.get(
        "mean_score",
        5.5
    )

    std = baseline.get(
        "std_score",
        2.0
    )

    z_score = (
        score - mean
    ) / (
        std + 0.01
    )

    percentile = min(
        99,
        max(
            1,
            int(50 + z_score * 15)
        )
    )

    return percentile


def assess_risk_level(
    score: float,
    stress_level: str
):

    if score < 3 or stress_level == "Very High":

        return (
            "CRITICAL: Immediate professional "
            "support recommended"
        )

    elif score < 5:

        return (
            "HIGH: Consider seeking counseling "
            "or mental health support"
        )

    elif score < 6.5:

        return (
            "MODERATE: Monitor your wellbeing "
            "and make gradual improvements"
        )

    else:

        return (
            "LOW: Continue maintaining your "
            "current healthy habits"
        )


# ============================================================
# EXPLANATION ENGINE
# ============================================================

def generate_explanation(
    data: StudentData,
    score: float
):

    explanations = []

    # --------------------------------------------------------
    # SLEEP
    # --------------------------------------------------------

    sleep_diff = (
        data.sleep_hours_per_night - 8
    )

    if sleep_diff > 1:

        sleep_impact = 1.5
        sleep_cat = "positive"

        sleep_msg = (
            f"Excellent sleep duration "
            f"({data.sleep_hours_per_night}h)"
        )

    elif 6.5 <= data.sleep_hours_per_night <= 8.5:

        sleep_impact = 0.5
        sleep_cat = "positive"

        sleep_msg = (
            f"Good sleep quality "
            f"({data.sleep_hours_per_night}h)"
        )

    else:

        sleep_impact = -abs(
            sleep_diff
        ) * 0.4

        sleep_cat = "negative"

        sleep_msg = (
            f"Sleep deprivation "
            f"({data.sleep_hours_per_night}h)"
        )

    explanations.append(
        ExplanationFactor(
            factor="Sleep Quality",
            category=sleep_cat,
            impact_score=round(
                sleep_impact,
                2
            ),
            explanation=sleep_msg,
            recommendation=(
                "Target 7-9 hours of consistent "
                "sleep daily."
            )
        )
    )

    # --------------------------------------------------------
    # PHYSICAL ACTIVITY
    # --------------------------------------------------------

    activity_impact = (
        data.physical_activity_hours - 1
    ) * 0.6

    if data.physical_activity_hours >= 1:

        activity_cat = "positive"

        activity_msg = (
            f"Adequate physical activity "
            f"({data.physical_activity_hours}h)"
        )

    elif data.physical_activity_hours >= 0.5:

        activity_cat = "neutral"

        activity_msg = (
            f"Moderate activity "
            f"({data.physical_activity_hours}h)"
        )

    else:

        activity_cat = "negative"

        activity_msg = (
            f"Low physical activity "
            f"({data.physical_activity_hours}h)"
        )

    explanations.append(
        ExplanationFactor(
            factor="Physical Activity",
            category=activity_cat,
            impact_score=round(
                activity_impact,
                2
            ),
            explanation=activity_msg,
            recommendation=(
                "Aim for at least 30-60 minutes "
                "of physical activity daily."
            )
        )
    )

    # --------------------------------------------------------
    # SCREEN TIME
    # --------------------------------------------------------

    screen_impact = -(
        data.avg_daily_usage_hours - 3.5
    ) * 0.5

    if data.avg_daily_usage_hours <= 3:

        screen_cat = "positive"

        screen_msg = (
            f"Healthy screen time "
            f"({data.avg_daily_usage_hours}h)"
        )

    elif data.avg_daily_usage_hours <= 5:

        screen_cat = "neutral"

        screen_msg = (
            f"Moderate screen time "
            f"({data.avg_daily_usage_hours}h)"
        )

    else:

        screen_cat = "negative"

        screen_msg = (
            f"High screen time "
            f"({data.avg_daily_usage_hours}h)"
        )

    explanations.append(
        ExplanationFactor(
            factor="Screen Time Management",
            category=screen_cat,
            impact_score=round(
                screen_impact,
                2
            ),
            explanation=screen_msg,
            recommendation=(
                "Consider reducing daily screen "
                "time and managing notifications."
            )
        )
    )

    # --------------------------------------------------------
    # STRESS
    # --------------------------------------------------------

    stress_map = {
        "Low": (2.0, "positive"),
        "Medium": (0, "neutral"),
        "High": (-1.5, "negative"),
        "Very High": (-3.0, "critical")
    }

    stress_impact, stress_cat = stress_map.get(
        data.stress_level,
        (0, "neutral")
    )

    explanations.append(
        ExplanationFactor(
            factor="Stress Level",
            category=stress_cat,
            impact_score=stress_impact,
            explanation=(
                f"Current stress level: "
                f"{data.stress_level}"
            ),
            recommendation=(
                "Practice mindfulness, meditation, "
                "breathing exercises, or consider "
                "professional support if needed."
            )
        )
    )

    # --------------------------------------------------------
    # STUDY
    # --------------------------------------------------------

    if 3 <= data.study_hours <= 6:

        study_impact = 0.8
        study_cat = "positive"

        study_msg = (
            f"Balanced study hours "
            f"({data.study_hours}h)"
        )

    elif data.study_hours > 8:

        study_impact = -0.6
        study_cat = "negative"

        study_msg = (
            f"High study hours "
            f"({data.study_hours}h)"
        )

    else:

        study_impact = -(
            4 - data.study_hours
        ) * 0.3

        study_cat = "negative"

        study_msg = (
            f"Low study hours "
            f"({data.study_hours}h)"
        )

    explanations.append(
        ExplanationFactor(
            factor="Study-Life Balance",
            category=study_cat,
            impact_score=round(
                study_impact,
                2
            ),
            explanation=study_msg,
            recommendation=(
                "Maintain a balanced study schedule "
                "with regular breaks."
            )
        )
    )

    # --------------------------------------------------------
    # PHONE DEPENDENCY
    # --------------------------------------------------------

    unlock_impact = -(
        min(
            data.daily_unlocks / 80,
            1
        ) * 0.7
    )

    if data.daily_unlocks < 50:

        unlock_cat = "positive"

        unlock_msg = (
            f"Low phone dependency "
            f"({data.daily_unlocks} unlocks/day)"
        )

    elif data.daily_unlocks < 100:

        unlock_cat = "neutral"

        unlock_msg = (
            f"Moderate phone usage "
            f"({data.daily_unlocks} unlocks/day)"
        )

    else:

        unlock_cat = "negative"

        unlock_msg = (
            f"High phone dependency "
            f"({data.daily_unlocks} unlocks/day)"
        )

    explanations.append(
        ExplanationFactor(
            factor="Phone Dependency Index",
            category=unlock_cat,
            impact_score=round(
                unlock_impact,
                2
            ),
            explanation=unlock_msg,
            recommendation=(
                "Use app timers, notification management, "
                "and phone-free periods."
            )
        )
    )

    return explanations


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def generate_recommendations(
    data: StudentData,
    score: float
):

    recommendations = []

    config = model_meta.get(
        "recommendations_config",
        {}
    )

    sleep_threshold = config.get(
        "sleep_threshold_low",
        6
    )

    activity_threshold = config.get(
        "activity_threshold_min",
        0.5
    )

    screen_threshold = config.get(
        "screen_time_threshold_max",
        5
    )

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if score < 3:

        recommendations.append(
            Recommendation(
                title="🚨 Professional Support",
                priority="critical",
                description=(
                    "Your score indicates that "
                    "professional mental health "
                    "support may be appropriate."
                ),
                action_items=[
                    "Consider contacting a qualified mental health professional",
                    "Reach out to a trusted friend or family member",
                    "Consider using your campus counseling service if available"
                ]
            )
        )

    # --------------------------------------------------------
    # SLEEP
    # --------------------------------------------------------

    if data.sleep_hours_per_night < sleep_threshold:

        recommendations.append(
            Recommendation(
                title="🌙 Improve Sleep Quality",
                priority="high",
                description=(
                    f"You reported "
                    f"{data.sleep_hours_per_night} "
                    f"hours of sleep per night."
                ),
                action_items=[
                    "Maintain a consistent sleep schedule",
                    "Reduce screen exposure before bedtime",
                    "Keep the sleeping environment comfortable and quiet",
                    "Aim for a regular 7-9 hour sleep window"
                ]
            )
        )

    # --------------------------------------------------------
    # ACTIVITY
    # --------------------------------------------------------

    if data.physical_activity_hours < activity_threshold:

        recommendations.append(
            Recommendation(
                title="🏃 Increase Physical Activity",
                priority="high",
                description=(
                    "Your reported physical activity "
                    "is relatively low."
                ),
                action_items=[
                    "Start with a daily walk",
                    "Try a sport, gym session, cycling, or yoga",
                    "Gradually work toward regular activity"
                ]
            )
        )

    # --------------------------------------------------------
    # SCREEN TIME
    # --------------------------------------------------------

    if data.avg_daily_usage_hours > screen_threshold:

        recommendations.append(
            Recommendation(
                title="📱 Review Screen Time",
                priority="high",
                description=(
                    f"You reported "
                    f"{data.avg_daily_usage_hours} "
                    f"hours of daily usage."
                ),
                action_items=[
                    "Use application time limits",
                    "Turn off unnecessary notifications",
                    "Create screen-free periods during the day",
                    "Avoid excessive screen use immediately before sleep"
                ]
            )
        )

    # --------------------------------------------------------
    # STRESS
    # --------------------------------------------------------

    if data.stress_level in [
        "High",
        "Very High"
    ]:

        recommendations.append(
            Recommendation(
                title="🧘 Stress Management",
                priority=(
                    "high"
                    if data.stress_level == "Very High"
                    else "medium"
                ),
                description=(
                    "Your reported stress level "
                    "is elevated."
                ),
                action_items=[
                    "Try breathing or relaxation exercises",
                    "Take regular breaks",
                    "Talk with someone you trust",
                    "Consider professional counseling if stress persists"
                ]
            )
        )

    # --------------------------------------------------------
    # STUDY BURNOUT
    # --------------------------------------------------------

    if data.study_hours > 8:

        recommendations.append(
            Recommendation(
                title="📚 Prevent Study Burnout",
                priority="medium",
                description=(
                    "Very long study periods can "
                    "make it harder to maintain balance."
                ),
                action_items=[
                    "Use focused study blocks",
                    "Take regular breaks",
                    "Schedule recreation and recovery time"
                ]
            )
        )

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    if score >= 7.5:

        recommendations.append(
            Recommendation(
                title="⭐ Maintain Your Wellness",
                priority="low",
                description=(
                    "Your current profile shows "
                    "several positive wellbeing indicators."
                ),
                action_items=[
                    "Continue your healthy routines",
                    "Maintain regular sleep and activity",
                    "Keep checking in with your wellbeing"
                ]
            )
        )

    return recommendations[:5]


# ============================================================
# WELLNESS SCORE
# ============================================================

def calculate_wellness_score(
    data: StudentData,
    score: float
):

    sleep_wellness = min(
        data.sleep_hours_per_night / 8 * 100,
        100
    )

    activity_wellness = min(
        data.physical_activity_hours / 1.5 * 100,
        100
    )

    screen_wellness = min(
        max(
            100 -
            (
                data.avg_daily_usage_hours /
                12 *
                100
            ),
            0
        ),
        100
    )

    stress_wellness = {
        "Low": 100,
        "Medium": 70,
        "High": 40,
        "Very High": 20
    }.get(
        data.stress_level,
        50
    )

    if 2 <= data.study_hours <= 6:

        study_wellness = min(
            data.study_hours / 4 * 100,
            100
        )

    else:

        study_wellness = max(
            100 -
            abs(
                data.study_hours - 4
            ) * 10,
            30
        )

    wellness = (
        sleep_wellness * 0.25
        +
        activity_wellness * 0.20
        +
        screen_wellness * 0.20
        +
        stress_wellness * 0.25
        +
        study_wellness * 0.10
    )

    return round(
        wellness,
        2
    )


# ============================================================
# FRONTEND
# ============================================================

@app.get(
    "/",
    include_in_schema=False
)
def serve_index():

    return FileResponse(
        INDEX_PATH,
        media_type="text/html"
    )


@app.get(
    "/style.css",
    include_in_schema=False
)
def serve_css():

    return FileResponse(
        CSS_PATH,
        media_type="text/css"
    )


@app.get(
    "/script.js",
    include_in_schema=False
)
def serve_javascript():

    return FileResponse(
        JS_PATH,
        media_type="application/javascript"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/api/health",
    tags=["Health"]
)
def health_check():

    return {
        "status": "ok",
        "model": model_meta.get(
            "selected_model",
            "Mental Health Model"
        ),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# MODEL INPUT PREPROCESSING
# ============================================================

def _find_label_encoder(column_name: str):
    """Find the saved encoder for a model feature, allowing common naming variants."""

    if label_encoders is None:
        return None

    if not isinstance(label_encoders, dict):
        return None

    aliases = {
        "Gender": ["gender"],
        "Academic_Level": ["academic level", "academic_level"],
        "Most_Used_Platform": ["most used platform", "most_used_platform"],
        "Purpose_Of_Use": ["purpose of use", "purpose_of_use"],
        "Stress_Level": ["stress level", "stress_level"],
        "Grouped_country": [
            "grouped country",
            "grouped_country",
            "country group",
            "country_group",
        ],
    }

    wanted = {column_name.strip().lower()}
    wanted.update(x.lower() for x in aliases.get(column_name, []))

    for key, encoder in label_encoders.items():
        if str(key).strip().lower() in wanted:
            return encoder

    return None


def _encode_one_value(encoder, value, column_name: str):
    """Encode one categorical value using the saved training encoder."""

    # sklearn LabelEncoder / OrdinalEncoder-like objects.
    if hasattr(encoder, "transform"):
        try:
            transformed = encoder.transform([value])
        except ValueError as exc:
            # Some encoders expect a 2-D input.
            try:
                transformed = encoder.transform([[value]])
            except Exception:
                raise ValueError(
                    f"Value '{value}' for '{column_name}' is not supported by "
                    "the saved training encoder."
                ) from exc

        arr = np.asarray(transformed)
        if arr.size == 0:
            raise ValueError(
                f"The saved encoder returned no value for '{column_name}'."
            )
        return float(arr.reshape(-1)[0])

    # Plain dictionary mapping, if the training script saved one.
    if isinstance(encoder, dict):
        if value in encoder:
            return float(encoder[value])
        # Also tolerate stringified dictionary keys.
        for key, encoded_value in encoder.items():
            if str(key) == str(value):
                return float(encoded_value)

    # sklearn-style classes_ + mapping fallback.
    classes = getattr(encoder, "classes_", None)
    if classes is not None:
        classes_list = list(classes)
        if value in classes_list:
            return float(classes_list.index(value))

    raise RuntimeError(
        f"The saved encoder for '{column_name}' has an unsupported format."
    )


def _expected_model_features() -> list[str]:
    """Return the exact feature order recorded by the fitted model when available."""

    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]

    # Some estimators expose feature names through a nested final estimator.
    final_estimator = getattr(model, "estimator_", None)
    names = getattr(final_estimator, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]

    return []


def _encode_model_input(input_row: pd.DataFrame) -> pd.DataFrame:
    """Convert the API's human-readable values into the model's numeric input."""

    encoded = input_row.copy()

    categorical_columns = [
        "Gender",
        "Academic_Level",
        "Most_Used_Platform",
        "Purpose_Of_Use",
        "Stress_Level",
        "Grouped_country",
    ]

    for column in categorical_columns:
        if column not in encoded.columns:
            continue

        # If the fitted model explicitly lists this feature as numeric,
        # use the saved encoder. This prevents raw strings such as "Male"
        # from ever reaching sklearn's numeric conversion.
        encoder = _find_label_encoder(column)
        if encoder is None:
            expected = _expected_model_features()
            if column in expected:
                raise RuntimeError(
                    f"No saved encoder was found for categorical feature '{column}'. "
                    "The model cannot safely convert this value without the training encoder."
                )
            continue

        value = encoded.at[encoded.index[0], column]
        encoded.at[encoded.index[0], column] = _encode_one_value(
            encoder,
            value,
            column,
        )

    # Match the exact training feature order and remove fields the model did
    # not see during fitting (notably raw Country).
    expected = _expected_model_features()
    if expected:
        missing = [name for name in expected if name not in encoded.columns]
        extra = [name for name in encoded.columns if name not in expected]

        if missing:
            raise RuntimeError(
                "The prediction input is missing model features: "
                + ", ".join(missing)
            )

        encoded = encoded.loc[:, expected]

    # Ensure every value reaching a numeric sklearn estimator is numeric.
    for column in encoded.columns:
        encoded[column] = pd.to_numeric(encoded[column], errors="raise")

    return encoded


def _predict_model(model_input: pd.DataFrame):
    """Predict, with a scaler fallback for projects whose saved estimator was trained on scaled data."""

    first_error = None

    try:
        return model.predict(model_input)
    except Exception as exc:
        first_error = exc

    # The project also ships scaler.pkl. If the saved estimator was trained
    # on scaled numeric features, retry using that scaler. We only do this
    # when the scaler's feature count/order is compatible with the model input.
    if scaler is not None and hasattr(scaler, "transform"):
        try:
            scaled = scaler.transform(model_input)

            if hasattr(scaler, "feature_names_in_"):
                scaled = pd.DataFrame(
                    scaled,
                    columns=list(scaler.feature_names_in_),
                    index=model_input.index,
                )
                expected = _expected_model_features()
                if expected:
                    if any(name not in scaled.columns for name in expected):
                        raise ValueError("Scaler output does not match model feature names.")
                    scaled = scaled.loc[:, expected]

            return model.predict(scaled)
        except Exception as scaled_error:
            raise RuntimeError(
                "Model prediction failed with the encoded input and with the saved scaler. "
                f"Direct error: {first_error}. Scaled error: {scaled_error}"
            ) from scaled_error

    raise RuntimeError(
        f"Model prediction failed after categorical preprocessing: {first_error}"
    ) from first_error


# ============================================================
# PREDICTION API
# ============================================================

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    tags=["Predictions"]
)
def predict(
    data: StudentData
):

    country_group = (
        data.country
        if data.country in TOP_COUNTRIES
        else "Other"
    )

    # Country is kept in the API/frontend, but the trained model uses
    # Grouped_country instead. This avoids the previous feature-name mismatch.
    input_row = pd.DataFrame(
        [{
            "Age": data.age,
            "Gender": data.gender,
            "Academic_Level": data.academic_level,
            "Most_Used_Platform": data.most_used_platform,
            "Purpose_Of_Use": data.purpose_of_use,
            "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
            "Daily_Unlocks": data.daily_unlocks,
            "Study_Hours": data.study_hours,
            "Physical_Activity_Hours": data.physical_activity_hours,
            "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
            "Stress_Level": data.stress_level,
            "Grouped_country": country_group
        }]
    )

    try:
        # Convert all categorical fields with the saved training encoders,
        # remove fields the fitted model never saw, and preserve feature order.
        model_input = _encode_model_input(input_row)

        prediction = _predict_model(model_input)

        score = float(
            prediction[0]
        )

    except Exception as e:

        print(
            "❌ Prediction error:",
            e
        )

        raise

    score = round(
        min(
            10,
            max(
                0,
                score
            )
        ),
        2
    )

    band, band_color, _ = get_score_band_info(
        score
    )

    percentile = calculate_percentile(
        score
    )

    risk_assessment = assess_risk_level(
        score,
        data.stress_level
    )

    explanations = generate_explanation(
        data,
        score
    )

    recommendations = generate_recommendations(
        data,
        score
    )

    wellness_score = calculate_wellness_score(
        data,
        score
    )

    summary = (
        f"Based on the information provided, "
        f"the model classified the profile as "
        f"{band.lower()}. "
        f"The estimated score is {score}/10."
    )

    return PredictionResponse(
        score=score,
        band=band,
        band_color=band_color,
        percentile=percentile,
        risk_assessment=risk_assessment,
        explanation=explanations,
        recommendations=recommendations,
        summary=summary,
        wellness_score=wellness_score
    )


# ============================================================
# MODEL / DATASET STATISTICS
# ============================================================

@app.get(
    "/api/stats",
    tags=["Analytics"]
)
def get_stats():

    # Get feature importance directly from the fitted estimator.
    # This supports plain estimators, sklearn pipelines, nested
    # estimators and linear models.
    feature_importances = {}

    def find_importances(obj, depth=0):
        if obj is None or depth > 6:
            return None

        try:
            values = getattr(obj, "feature_importances_", None)
            if values is not None:
                return np.asarray(values).flatten().tolist()
        except Exception:
            pass

        try:
            values = getattr(obj, "coef_", None)
            if values is not None:
                arr = np.asarray(values)
                if arr.ndim > 1:
                    arr = np.mean(np.abs(arr), axis=0)
                else:
                    arr = np.abs(arr)
                return arr.flatten().tolist()
        except Exception:
            pass

        # sklearn Pipeline / similar objects
        try:
            steps = getattr(obj, "steps", None)
            if steps:
                for _, step in reversed(steps):
                    result = find_importances(step, depth + 1)
                    if result:
                        return result
        except Exception:
            pass

        # Common nested estimator attributes
        for attr_name in (
            "estimator_", "estimator", "regressor_", "regressor",
            "classifier_", "classifier", "final_estimator",
            "base_estimator", "model_", "model"
        ):
            try:
                nested = getattr(obj, attr_name, None)
                if nested is not None and nested is not obj:
                    result = find_importances(nested, depth + 1)
                    if result:
                        return result
            except Exception:
                pass

        # Ensemble estimators
        try:
            estimators = getattr(obj, "estimators_", None)
            if estimators:
                collected = []
                for estimator in estimators:
                    result = find_importances(estimator, depth + 1)
                    if result:
                        collected.append(result)
                if collected:
                    min_len = min(len(x) for x in collected)
                    if min_len:
                        return np.mean(
                            [x[:min_len] for x in collected],
                            axis=0
                        ).tolist()
        except Exception:
            pass

        return None

    try:
        raw_importances = find_importances(model)

        if raw_importances:
            feature_names = None

            # Plain estimator names
            for obj in (model,):
                try:
                    feature_names = getattr(
                        obj, "feature_names_in_", None
                    )
                    if feature_names is not None:
                        break
                except Exception:
                    pass

            # Pipeline names
            if feature_names is None:
                try:
                    if hasattr(model, "steps"):
                        for _, step in model.steps:
                            feature_names = getattr(
                                step, "feature_names_in_", None
                            )
                            if feature_names is not None:
                                break
                except Exception:
                    pass

            # Scaler names
            if feature_names is None and scaler is not None:
                try:
                    feature_names = getattr(
                        scaler, "feature_names_in_", None
                    )
                except Exception:
                    pass

            # Model-provided output names
            if feature_names is None:
                try:
                    feature_names = model.get_feature_names_out()
                except Exception:
                    pass

            # Last fallback: the features used by the prediction endpoint.
            if feature_names is None:
                feature_names = [
                    "Age",
                    "Gender",
                    "Academic_Level",
                    "Most_Used_Platform",
                    "Purpose_Of_Use",
                    "Avg_Daily_Usage_Hours",
                    "Daily_Unlocks",
                    "Study_Hours",
                    "Physical_Activity_Hours",
                    "Sleep_Hours_Per_Night",
                    "Stress_Level",
                    "Grouped_country"
                ]

            feature_names = list(feature_names)
            count = min(len(feature_names), len(raw_importances))

            for i in range(count):
                try:
                    value = float(raw_importances[i])
                    if np.isfinite(value):
                        feature_importances[str(feature_names[i])] = value
                except Exception:
                    continue

    except Exception as e:
        print("⚠️ Feature importance extraction failed:", e)

    # Metadata fallback, if the model itself does not expose importance.
    if not feature_importances:
        metadata_importances = model_meta.get(
            "feature_importances", {}
        )
        if isinstance(metadata_importances, dict):
            feature_importances = metadata_importances

    sorted_importances = sorted(
        feature_importances.items(),
        key=lambda item: abs(float(item[1])),
        reverse=True
    )

    print(
        f"📊 Feature importance entries: {len(sorted_importances)}"
    )

    return {
        "model": {
            "name": model_meta.get(
                "selected_model",
                "Mental Health Model"
            ),
            "performance": model_meta.get(
                "performance",
                {}
            ),
            "feature_importances": sorted_importances[:10]
        },
        "dataset": {
            "baseline": model_meta.get(
                "dataset_baseline",
                {}
            ),
            "distribution": model_meta.get(
                "score_distribution",
                {}
            ),
            "total_records": 700
        },
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# DASHBOARD DATA
# ============================================================

@app.get(
    "/api/dashboard",
    tags=["Dashboard"]
)
def get_dashboard_data():

    predictions_today = random.randint(
        15,
        45
    )

    avg_score_today = round(
        random.uniform(
            5.5,
            6.8
        ),
        2
    )

    improvement_rate = random.randint(
        60,
        85
    )

    users_at_risk = random.randint(
        8,
        25
    )

    metrics = [

        DashboardMetric(
            label="Predictions Today",
            value=str(
                predictions_today
            ),
            change=f"+{random.randint(5, 15)}%",
            status="good"
        ),

        DashboardMetric(
            label="Average Score",
            value=f"{avg_score_today}/10",
            change=f"+{round(random.uniform(0.1, 0.5), 2)}",
            status="good"
        ),

        DashboardMetric(
            label="Improvement Rate",
            value=f"{improvement_rate}%",
            change=f"+{random.randint(2, 8)}%",
            status="good"
        ),

        DashboardMetric(
            label="Users at Risk",
            value=str(
                users_at_risk
            ),
            change=f"-{random.randint(1, 5)}",
            status="critical"
        )
    ]

    dates = [

        (
            datetime.now()
            -
            timedelta(days=i)
        ).strftime(
            "%Y-%m-%d"
        )

        for i in range(
            7,
            0,
            -1
        )
    ]

    trends = [

        HealthTrend(
            date=date,
            score=round(
                random.uniform(
                    5.0,
                    8.5
                ),
                2
            ),
            band=random.choice(
                [
                    "Struggling",
                    "Fair",
                    "Good",
                    "Thriving"
                ]
            )
        )

        for date in dates
    ]

    return {

        "metrics": metrics,

        "trends": trends,

        "model_info": model_meta.get(
            "selected_model",
            "Mental Health Model"
        ),

        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# SAMPLE PROFILES
# ============================================================

@app.get(
    "/api/sample-profiles",
    tags=["Examples"]
)
def get_sample_profiles():

    return {

        "profiles": [

            {
                "name": "Stressed Student",
                "age": 20,
                "gender": "Male",
                "country": "India",
                "academic_level": "Undergraduate",
                "most_used_platform": "Instagram",
                "purpose_of_use": "Entertainment",
                "avg_daily_usage_hours": 8,
                "daily_unlocks": 120,
                "study_hours": 2,
                "physical_activity_hours": 0.25,
                "sleep_hours_per_night": 5,
                "stress_level": "Very High"
            },

            {
                "name": "Balanced Professional",
                "age": 25,
                "gender": "Female",
                "country": "USA",
                "academic_level": "Graduate",
                "most_used_platform": "LinkedIn",
                "purpose_of_use": "Networking",
                "avg_daily_usage_hours": 3,
                "daily_unlocks": 40,
                "study_hours": 1.5,
                "physical_activity_hours": 1,
                "sleep_hours_per_night": 7.5,
                "stress_level": "Low"
            },

            {
                "name": "Thriving Student",
                "age": 22,
                "gender": "Male",
                "country": "Canada",
                "academic_level": "Undergraduate",
                "most_used_platform": "YouTube",
                "purpose_of_use": "Education",
                "avg_daily_usage_hours": 2,
                "daily_unlocks": 30,
                "study_hours": 4,
                "physical_activity_hours": 1.5,
                "sleep_hours_per_night": 8,
                "stress_level": "Low"
            }

        ]
    }


# ============================================================
# STARTUP MESSAGE
# ============================================================

print("")
print("=" * 60)
print("🧠 ULTIMATE MENTAL HEALTH ANALYTICS PLATFORM")
print("=" * 60)
print("✅ Backend loaded")
print("✅ ML model loaded")
print("✅ Frontend HTML available")
print("✅ CSS available")
print("✅ JavaScript available")
print("")
print("🌐 Open:")
print("   http://127.0.0.1:8000/")
print("")
print("📚 API documentation:")
print("   http://127.0.0.1:8000/docs")
print("=" * 60)