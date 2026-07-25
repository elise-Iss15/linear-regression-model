# salary prediction api using FASTAPI  for Linkedin job postings dataset


"""
AKAZI SCROLL — Salary Prediction API
FastAPI app that loads the best-performing model (DecisionTreeRegressor) trained
in summative/linear_regression/multivariate.ipynb and exposes prediction endpoints.
"""

from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load model artifacts once, at startup
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).parent.parent / "linear_regression"

model = joblib.load(ARTIFACT_DIR / "best_model.joblib")
scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
model_columns = joblib.load(ARTIFACT_DIR / "model_columns.joblib")

EXPERIENCE_ORDER = {
    "Not Specified": 0,
    "Internship": 1,
    "Entry level": 2,
    "Associate": 3,
    "Mid-Senior level": 4,
    "Director": 5,
    "Executive": 6,
}


# ---------------------------------------------------------------------------
# Enums — restrict input to values the model was actually trained on
# ---------------------------------------------------------------------------
class TitleCategory(str, Enum):
    administrative_office_support = "Administrative/Office Support"
    customer_service_support = "Customer Service/Support"
    data_analytics = "Data/Analytics"
    design = "Design"
    education = "Education"
    engineering = "Engineering"
    finance = "Finance"
    general_labor_hospitality = "General Labor/Hospitality"
    hr_recruiting = "HR/Recruiting"
    healthcare = "Healthcare"
    legal = "Legal"
    management_executive = "Management/Executive"
    marketing = "Marketing"
    other = "Other"
    sales = "Sales"
    senior_lead = "Senior/Lead"
    technical_trades = "Technical/Trades"


class State(str, Enum):
    AK = "AK"
    AL = "AL"
    AR = "AR"
    AZ = "AZ"
    CA = "CA"
    CO = "CO"
    CT = "CT"
    DC = "DC"
    DE = "DE"
    FL = "FL"
    GA = "GA"
    HI = "HI"
    IA = "IA"
    ID = "ID"
    IL = "IL"
    IN = "IN"
    international_other = "International/Other"
    KS = "KS"
    KY = "KY"
    LA = "LA"
    MA = "MA"
    MD = "MD"
    ME = "ME"
    MI = "MI"
    MN = "MN"
    MO = "MO"
    MS = "MS"
    MT = "MT"
    NC = "NC"
    ND = "ND"
    NE = "NE"
    NH = "NH"
    NJ = "NJ"
    NM = "NM"
    NV = "NV"
    NY = "NY"
    OH = "OH"
    OK = "OK"
    OR = "OR"
    PA = "PA"
    RI = "RI"
    SC = "SC"
    SD = "SD"
    TN = "TN"
    TX = "TX"
    us_metro_unspecified = "US - Metro Area (Unspecified State)"
    us_unspecified = "US - Unspecified"
    UT = "UT"
    VA = "VA"
    VT = "VT"
    WA = "WA"
    WI = "WI"
    WV = "WV"
    WY = "WY"


class WorkType(str, Enum):
    full_time = "Full-time"
    internship = "Internship"
    contract = "Contract"
    part_time = "Part-time"
    temporary = "Temporary"
    other = "Other"
    volunteer = "Volunteer"


class ExperienceLevel(str, Enum):
    not_specified = "Not Specified"
    internship = "Internship"
    entry_level = "Entry level"
    associate = "Associate"
    mid_senior_level = "Mid-Senior level"
    director = "Director"
    executive = "Executive"


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    title_category: TitleCategory
    state: State
    formatted_work_type: WorkType
    formatted_experience_level: ExperienceLevel
    views: int = Field(
        ..., ge=0, le=100_000, description="Number of times the posting was viewed"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "title_category": "Engineering",
                "state": "CA",
                "formatted_work_type": "Full-time",
                "formatted_experience_level": "Mid-Senior level",
                "views": 50,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_salary: float
    currency: str = "USD"


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AKAZI SCROLL Salary Prediction API",
    description="Predicts a fair salary estimate for a job posting, powering "
    "AKAZI SCROLL's salary-estimation feature.",
    version="1.0.0",
)

# CORS reasoning: we scope allowed origins explicitly rather than using "*",
# since a wildcard would let ANY website call this API and burn our compute /
# potentially scrape predictions. Only localhost (dev) and the deployed
# Flutter app's origin (added once known) are allowed. Only GET/POST are
# needed since this API has no other verbs. Credentials are not required
# (no cookies/auth session used), so allow_credentials stays False.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def encode_request(req: PredictionRequest) -> pd.DataFrame:
    """Turn simple human-readable input into the exact column vector the model expects."""
    row = {col: 0 for col in model_columns}

    row["views"] = req.views
    row["experience_level_encoded"] = EXPERIENCE_ORDER[
        req.formatted_experience_level.value
    ]

    # Set the one matching one-hot column to 1, if it exists as a column.
    # (If the submitted category was the "dropped" baseline during training,
    # no column exists for it — leaving everything else 0 is exactly correct.)
    for prefix, value in [
        ("title_category_", req.title_category.value),
        ("state_", req.state.value),
        ("formatted_work_type_", req.formatted_work_type.value),
    ]:
        col_name = f"{prefix}{value}"
        if col_name in row:
            row[col_name] = 1

    df_row = pd.DataFrame([row])[model_columns]

    numeric_cols = ["views", "experience_level_encoded"]
    df_row[numeric_cols] = scaler.transform(df_row[numeric_cols])

    return df_row


@app.get("/")
def root():
    return {
        "message": "AKAZI SCROLL Salary Prediction API. Visit /docs for Swagger UI."
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        encoded = encode_request(request)
        log_prediction = model.predict(encoded)[0]
        salary = float(np.expm1(log_prediction))
        return PredictionResponse(predicted_salary=round(salary, 2))
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
