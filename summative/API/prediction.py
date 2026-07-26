# salary prediction api using FASTAPI  for Linkedin job postings dataset

"""AKAZI SCROLL — Salary Prediction API"""

from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

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


class PredictionRequest(BaseModel):
    title_category: TitleCategory
    state: State
    formatted_work_type: WorkType
    formatted_experience_level: ExperienceLevel
    views: int = Field(..., ge=0, le=100_000)

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


class RetrainDataPoint(BaseModel):
    title_category: TitleCategory
    state: State
    formatted_work_type: WorkType
    formatted_experience_level: ExperienceLevel
    views: int = Field(..., ge=0, le=100_000)
    normalized_salary: float = Field(..., ge=15_000, le=400_000)


class RetrainRequest(BaseModel):
    new_data: list[RetrainDataPoint]


class RetrainResponse(BaseModel):
    message: str
    rows_used_for_retraining: int
    new_test_rmse: float
    new_test_r2: float


app = FastAPI(
    title="AKAZI SCROLL Salary Prediction API",
    description="Predicts a fair salary estimate for a job posting, powering AKAZI SCROLL's salary-estimation feature.",
    version="1.0.0",
)

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
    row = {col: 0 for col in model_columns}

    row["views"] = req.views
    row["experience_level_encoded"] = EXPERIENCE_ORDER[
        req.formatted_experience_level.value
    ]

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
    except (ValueError, KeyError, IndexError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest):
    global model, scaler

    if len(request.new_data) == 0:
        raise HTTPException(status_code=400, detail="No data provided for retraining")

    try:
        base_df = pd.read_csv(ARTIFACT_DIR / "cleaned_base_data.csv")

        new_rows = pd.DataFrame(
            [
                {
                    "title_category": d.title_category.value,
                    "state": d.state.value,
                    "formatted_work_type": d.formatted_work_type.value,
                    "formatted_experience_level": d.formatted_experience_level.value,
                    "views": d.views,
                    "normalized_salary": d.normalized_salary,
                }
                for d in request.new_data
            ]
        )

        combined = pd.concat([base_df, new_rows], ignore_index=True)

        combined["experience_level_encoded"] = combined[
            "formatted_experience_level"
        ].map(EXPERIENCE_ORDER)
        combined = pd.get_dummies(
            combined,
            columns=["title_category", "state", "formatted_work_type"],
            drop_first=True,
        )
        combined["log_salary"] = np.log1p(combined["normalized_salary"])
        combined = combined.drop(
            columns=["formatted_experience_level", "normalized_salary"]
        )

        X_new = combined.drop(columns=["log_salary"])
        y_new = combined["log_salary"]

        for col in model_columns:
            if col not in X_new.columns:
                X_new[col] = 0
        X_new = X_new[model_columns]

        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
            X_new, y_new, test_size=0.2, random_state=42
        )

        new_scaler = StandardScaler()
        numeric_cols = ["views", "experience_level_encoded"]
        X_train_new_scaled = X_train_new.copy()
        X_test_new_scaled = X_test_new.copy()
        X_train_new_scaled[numeric_cols] = new_scaler.fit_transform(
            X_train_new[numeric_cols]
        )
        X_test_new_scaled[numeric_cols] = new_scaler.transform(X_test_new[numeric_cols])

        new_model = DecisionTreeRegressor(max_depth=10, random_state=42)
        new_model.fit(X_train_new_scaled, y_train_new)

        test_pred = new_model.predict(X_test_new_scaled)
        test_rmse = float(np.sqrt(mean_squared_error(y_test_new, test_pred)))
        test_r2 = float(r2_score(y_test_new, test_pred))

        joblib.dump(new_model, ARTIFACT_DIR / "best_model.joblib")
        joblib.dump(new_scaler, ARTIFACT_DIR / "scaler.joblib")
        joblib.dump(model_columns, ARTIFACT_DIR / "model_columns.joblib")

        model = new_model
        scaler = new_scaler

        return RetrainResponse(
            message="Model retrained successfully",
            rows_used_for_retraining=len(combined),
            new_test_rmse=round(test_rmse, 4),
            new_test_r2=round(test_r2, 4),
        )

    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"Retraining failed: {e}")
