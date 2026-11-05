
import os
import warnings
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

app = FastAPI(
    title="E-Commerce AI Engine API",
    description=(
        "REST API for customer churn prediction, "
        "similar products, cart bundles and personalized recommendations."
    ),
    version="2.2.0",
)

MODELS_DIR = Path(
    os.getenv("MODELS_DIR", r"D:\ecomai\models")
)


# ============================================================
# 2. GLOBAL MODEL VARIABLES
# ============================================================

churn_model = None
engine1_sim = None
engine2_rules = None
engine3_svd = None

model_errors = {}


# ============================================================
# 3. MODEL LOADING
# ============================================================

def load_model(filename):

    path = MODELS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    return joblib.load(path)


def load_all_models():

    global churn_model
    global engine1_sim
    global engine2_rules
    global engine3_svd

    models = {
        "churn_model": "gradient_boosting_tuned.pkl",
        "engine1_sim": "engine1_item_similarity.pkl",
        "engine2_rules": "engine2_cart_rules.pkl",
        "engine3_svd": "engine3_personalized_svd.pkl",
    }

    for variable_name, filename in models.items():

        try:

            model = load_model(filename)

            globals()[variable_name] = model

            print(f"[SUCCESS] Loaded: {filename}")

        except Exception as error:

            model_errors[variable_name] = str(error)

            print(
                f"[ERROR] Could not load {filename}: {error}"
            )


load_all_models()


# ============================================================
# 4. MODEL VALIDATION
# ============================================================

def require_model(model, model_name):

    if model is None:

        detail = model_errors.get(
            model_name,
            "Model is not loaded."
        )

        raise HTTPException(
            status_code=503,
            detail=f"{model_name} unavailable: {detail}"
        )


# ============================================================
# 5. REQUEST SCHEMAS
# ============================================================

class ChurnInput(BaseModel):

    Frequency: float = Field(ge=0)
    Monetary: float = Field(ge=0)
    Tenure: float = Field(ge=0)
    AOV: float = Field(ge=0)
    MaxOrderValue: float = Field(ge=0)
    StdDevSpend: float = Field(ge=0)
    TotalUnits: float = Field(ge=0)
    AvgBasketSize: float = Field(ge=0)
    UniqueProducts: float = Field(ge=0)
    AvgDaysBetweenOrders: float = Field(ge=0)
    UnitsPerOrder: float = Field(ge=0)


class ItemRecInput(BaseModel):

    product_name: str = Field(min_length=1)

    top_n: int = Field(
        default=5,
        ge=1,
        le=50
    )


class UserRecInput(BaseModel):

    customer_id: int

    top_n: int = Field(
        default=5,
        ge=1,
        le=50
    )


# ============================================================
# 6. ROOT HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():

    return {
        "status": "online",
        "message": "E-Commerce AI REST API is running.",

        "models": {
            "churn_model": churn_model is not None,
            "item_similarity": engine1_sim is not None,
            "cart_rules": engine2_rules is not None,
            "personalized_svd": engine3_svd is not None,
        }
    }


# ============================================================
# 7. DETAILED HEALTH CHECK
# ============================================================

@app.get("/health")
def detailed_health_check():

    return {
        "status": "online",

        "models_directory": str(MODELS_DIR),

        "models_loaded": {
            "churn_model": churn_model is not None,
            "item_similarity": engine1_sim is not None,
            "cart_rules": engine2_rules is not None,
            "personalized_svd": engine3_svd is not None,
        },

        "model_errors": model_errors,

        "churn_expected_features": (
            list(churn_model.feature_names_in_)
            if churn_model is not None
            and hasattr(churn_model, "feature_names_in_")
            else None
        )
    }


# ============================================================
# 8. CUSTOMER CHURN PREDICTION
# ============================================================

@app.post("/predict/churn")
def predict_churn(data: ChurnInput):

    require_model(
        churn_model,
        "churn_model"
    )

    try:

        # Convert request into DataFrame
        input_df = pd.DataFrame([
            data.model_dump()
        ])

        # Obtain exact model training features
        if not hasattr(churn_model, "feature_names_in_"):

            raise ValueError(
                "The churn model does not expose feature_names_in_. "
                "Verify the model training feature columns."
            )

        expected_features = list(
            churn_model.feature_names_in_
        )

        # Validate required features
        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_df.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        # Match training feature order
        input_df = input_df[expected_features]

        # Generate prediction
        if hasattr(churn_model, "predict_proba"):

            probabilities = churn_model.predict_proba(
                input_df
            )[0]

            classes = list(
                churn_model.classes_
            )

            positive_index = classes.index(1)

            churn_probability = float(
                probabilities[positive_index]
            )

        else:

            prediction = int(
                churn_model.predict(input_df)[0]
            )

            churn_probability = float(prediction)

        # Keep probability between 0 and 1
        churn_probability = max(
            0.0,
            min(1.0, churn_probability)
        )

        # Risk classification
        if churn_probability >= 0.35:

            risk_level = "High"

        elif churn_probability >= 0.20:

            risk_level = "Medium"

        else:

            risk_level = "Low"

        return {

            "churn_prediction": int(
                churn_probability >= 0.35
            ),

            "churn_probability": round(
                churn_probability, 4
            ),

            "risk_level": risk_level,

            "model": "Gradient Boosting"
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Churn prediction error: {str(error)}"
        )


# ============================================================
# 9. SIMILAR PRODUCT RECOMMENDATIONS
# ============================================================

@app.post("/recommend/similar-items")
def recommend_similar(data: ItemRecInput):

    require_model(
        engine1_sim,
        "engine1_sim"
    )

    try:

        if data.product_name not in engine1_sim.columns:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Product '{data.product_name}' "
                    "was not found."
                )
            )

        scores = (

            engine1_sim[data.product_name]

            .sort_values(ascending=False)

            .drop(
                labels=[data.product_name],
                errors="ignore"
            )

            .head(data.top_n)
        )

        recommendations = [

            {
                "product_name": str(product),

                "similarity_score": round(
                    float(score), 4
                )
            }

            for product, score in scores.items()

        ]

        return {

            "product_name": data.product_name,

            "recommendations": recommendations,

            "total_recommendations": len(
                recommendations
            )
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Similar product recommendation error: {str(error)}"
            )
        )


# ============================================================
# 10. CART-BASED CROSS-SELL RECOMMENDATIONS
# ============================================================

@app.post("/recommend/cart-bundles")
def recommend_cart(data: ItemRecInput):

    require_model(
        engine2_rules,
        "engine2_rules"
    )

    try:

        required_columns = {

            "antecedents",
            "consequents",
            "confidence",
            "lift"

        }

        if not required_columns.issubset(
            engine2_rules.columns
        ):

            raise ValueError(
                "Cart rules are missing required columns."
            )

        matching = engine2_rules[

            engine2_rules["antecedents"].apply(

                lambda items: data.product_name in items

            )
        ]

        if matching.empty:

            return {

                "product_name": data.product_name,

                "message": "No cross-sell rules found.",

                "bundles": [],

                "total_bundles": 0
            }

        top_rules = (

            matching.sort_values(

                by=[
                    "confidence",
                    "lift"
                ],

                ascending=False

            )

            .head(data.top_n)
        )

        bundles = []

        for _, row in top_rules.iterrows():

            bundles.append({

                "recommended_items": [

                    str(item)

                    for item in row["consequents"]

                ],

                "confidence": round(
                    float(row["confidence"]), 4
                ),

                "lift": round(
                    float(row["lift"]), 4
                )
            })

        return {

            "product_name": data.product_name,

            "bundles": bundles,

            "total_bundles": len(bundles)
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Cart recommendation error: {str(error)}"
            )
        )


# ============================================================
# 11. PERSONALIZED SVD RECOMMENDATIONS
# ============================================================

@app.post("/recommend/user-feed")
def recommend_user(data: UserRecInput):

    require_model(
        engine3_svd,
        "engine3_svd"
    )

    try:

        matrix = engine3_svd["user_item_matrix"]

        user_factors = engine3_svd["user_factors"]

        item_factors = engine3_svd["item_factors"]

        # Validate customer
        if data.customer_id not in matrix.index:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Customer ID {data.customer_id} "
                    "was not found."
                )
            )

        # Get customer index
        user_index = matrix.index.get_loc(
            data.customer_id
        )

        # Calculate predicted preference scores
        predicted_ratings = (

            user_factors[user_index] @ item_factors

        )

        predictions = pd.Series(

            predicted_ratings,

            index=matrix.columns
        )

        # Exclude previously purchased products
        already_purchased = matrix.loc[
            data.customer_id
        ]

        predictions = predictions[
            already_purchased == 0
        ]

        # Get top recommendations
        top_predictions = (

            predictions

            .sort_values(ascending=False)

            .head(data.top_n)
        )

        recommendations = [

            {

                "product_name": str(product),

                "predicted_score": round(
                    float(score), 4
                )

            }

            for product, score in top_predictions.items()

        ]

        return {

            "customer_id": data.customer_id,

            "recommendations": recommendations,

            "total_recommendations": len(
                recommendations
            )
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Personalized recommendation error: {str(error)}"
            )
        )


# ============================================================
# 12. RUN API SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
