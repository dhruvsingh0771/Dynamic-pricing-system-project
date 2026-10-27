from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Literal
import joblib
import numpy as np
from fastapi import Query
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "events.db"
# --------------------------------------------------
# 2. Database setup
# --------------------------------------------------
def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection





def initialize_database():

    with get_connection() as connection:

        connection.execute("""

            CREATE TABLE IF NOT EXISTS events (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                event_type TEXT NOT NULL

                    CHECK (event_type IN ('view', 'cart', 'purchase')),

                product_id TEXT NOT NULL,

                user_id TEXT,

                session_id TEXT,

                event_timestamp TEXT NOT NULL,

                price REAL,

                quantity INTEGER NOT NULL DEFAULT 1,

                received_at TEXT NOT NULL

            )

        """)



        connection.execute("""

            CREATE INDEX IF NOT EXISTS idx_events_product

            ON events(product_id)

        """)



        connection.execute("""

            CREATE INDEX IF NOT EXISTS idx_events_timestamp

            ON events(event_timestamp)

        """)

        connection.execute("""

            CREATE TABLE IF NOT EXISTS pricing_decisions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                product_id TEXT NOT NULL,

                decision_timestamp TEXT NOT NULL,

                current_price REAL NOT NULL,

                recommended_price REAL NOT NULL,

                predicted_demand REAL NOT NULL,

                expected_revenue REAL NOT NULL

            )

        """)



        connection.execute("""

            CREATE INDEX IF NOT EXISTS idx_pricing_product_timestamp

            ON pricing_decisions(product_id, decision_timestamp)

        """)

        connection.execute("""

            CREATE TABLE IF NOT EXISTS active_prices (

                product_id TEXT PRIMARY KEY,

                active_price REAL NOT NULL CHECK (active_price > 0),

                updated_at TEXT NOT NULL

            )

        """)



@asynccontextmanager

async def lifespan(app: FastAPI):

    initialize_database()

    yield





# --------------------------------------------------

# 3. Create FastAPI application

# --------------------------------------------------



app = FastAPI(

    title="Real-Time Dynamic Pricing API",

    description="Backend for demand forecasting and dynamic pricing",

    version="1.0.0",

    lifespan=lifespan,

)





# --------------------------------------------------

# 4. Event request validation

# --------------------------------------------------



class EventCreate(BaseModel):

    event_type: Literal["view", "cart", "purchase"]

    product_id: str = Field(min_length=1, max_length=100)



    user_id: str | None = Field(default=None, max_length=100)

    session_id: str | None = Field(default=None, max_length=100)



    timestamp: datetime | None = None

    price: float | None = Field(default=None, gt=0)

    quantity: int = Field(default=1, ge=1, le=10000)





# --------------------------------------------------

# 5. Home and health endpoints

# --------------------------------------------------



@app.get("/")

def home():

    return {

        "message": "Real-Time Dynamic Pricing API is running",

        "status": "success",

    }





@app.get("/health")

def health_check():

    return {"status": "healthy"}





# --------------------------------------------------

# 6. Event ingestion endpoint

# --------------------------------------------------



@app.post("/api/events", status_code=status.HTTP_201_CREATED)

def create_event(event: EventCreate):

    event_timestamp = event.timestamp or datetime.now(timezone.utc)



    if event_timestamp.tzinfo is None:

        event_timestamp = event_timestamp.replace(tzinfo=timezone.utc)



    timestamp_text = event_timestamp.astimezone(

        timezone.utc

    ).isoformat()



    received_at = datetime.now(timezone.utc).isoformat()



    try:

        with get_connection() as connection:

            cursor = connection.execute(

                """

                INSERT INTO events (

                    event_type,

                    product_id,

                    user_id,

                    session_id,

                    event_timestamp,

                    price,

                    quantity,

                    received_at

                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?)

                """,

                (

                    event.event_type,

                    event.product_id.strip(),

                    event.user_id,

                    event.session_id,

                    timestamp_text,

                    event.price,

                    event.quantity,

                    received_at,

                ),

            )



            event_id = cursor.lastrowid



    except sqlite3.Error as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail="Could not save event to the database.",

        ) from exc



    return {

        "message": "Event accepted and stored",

        "event_id": event_id,

        "event_type": event.event_type,

        "product_id": event.product_id.strip(),

        "timestamp": timestamp_text,

    }





# --------------------------------------------------

# 7. Retrieve stored events

# --------------------------------------------------



@app.get("/api/events")

def list_events(limit: int = 50):

    if not 1 <= limit <= 500:

        raise HTTPException(

            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,

            detail="limit must be between 1 and 500",

        )



    with get_connection() as connection:

        rows = connection.execute(

            """

            SELECT *

            FROM events

            ORDER BY id DESC

            LIMIT ?

            """,

            (limit,),

        ).fetchall()



    return {

        "count": len(rows),

        "events": [dict(row) for row in rows],

    }



# --------------------------------------------------

# 8. Product data endpoints

# --------------------------------------------------



import pandas as pd

from functools import lru_cache



PRODUCT_DATA_PATH = (

    BASE_DIR.parent

    / "data"

    / "processed"

    / "final_hourly_15_products.csv"

)





@lru_cache(maxsize=1)

def load_product_data():

    if not PRODUCT_DATA_PATH.exists():

        raise HTTPException(

            status_code=500,

            detail="Historical product dataset was not found.",

        )



    df = pd.read_csv(PRODUCT_DATA_PATH)



    df["product_id"] = df["product_id"].astype(str)

    df["hour"] = pd.to_datetime(df["hour"], utc=True)



    return df





@app.get("/api/products")

def list_products():

    df = load_product_data()



    # Latest historical record for each product.

    latest = (

        df.sort_values("hour")

        .groupby("product_id", as_index=False)

        .tail(1)

        .sort_values("product_id")

    )



    products = []



    for _, row in latest.iterrows():

        products.append({

            "product_id": row["product_id"],

            "latest_historical_price": float(row["price"]),

            "latest_historical_views": int(row["views"]),

            "latest_historical_purchases": int(row["purchases"]),

            "last_recorded_hour": row["hour"].isoformat(),

        })



    return {

        "count": len(products),

        "products": products,

    }





@app.get("/api/products/{product_id}")

def get_product(product_id: str):

    df = load_product_data()



    product_rows = df[df["product_id"] == product_id]



    if product_rows.empty:

        raise HTTPException(

            status_code=404,

            detail=f"Product '{product_id}' was not found.",

        )



    latest = product_rows.sort_values("hour").iloc[-1]



    return {

        "product_id": product_id,

        "latest_historical_price": float(latest["price"]),

        "latest_historical_views": int(latest["views"]),

        "latest_historical_purchases": int(latest["purchases"]),

        "last_recorded_hour": latest["hour"].isoformat(),

        "historical_records": int(len(product_rows)),

        "historical_min_price": float(product_rows["price"].min()),

        "historical_max_price": float(product_rows["price"].max()),

    }

MODEL_PATH = BASE_DIR.parent / "models" / "demand_model_lagged.joblib"





@lru_cache(maxsize=1)

def load_model_artifact():

  if not MODEL_PATH.exists():

        raise HTTPException(

            status_code=500,

            detail="Trained demand model was not found."

        )

  return joblib.load(MODEL_PATH)





def get_previous_hour_views(product_id: str, forecast_time: datetime, historical_fallback: int) -> tuple[int, str]:
    """Return view-event volume from the hour before forecast_time, or historical fallback."""
    end_time = forecast_time.astimezone(timezone.utc)
    start_time = end_time.timestamp() - 3600
    start_text = datetime.fromtimestamp(start_time, timezone.utc).isoformat()
    end_text = end_time.isoformat()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS view_event_count, COALESCE(SUM(quantity), 0) AS view_total
            FROM events
            WHERE product_id = ?
              AND event_type = 'view'
              AND event_timestamp >= ?
              AND event_timestamp < ?
            """,
            (product_id, start_text, end_text),
        ).fetchone()

    if row is not None and int(row["view_event_count"]) > 0:
        return int(row["view_total"]), "live_events"

    return max(0, int(historical_fallback)), "historical_fallback"


@app.get("/api/products/{product_id}/forecast")
def forecast_demand(
    product_id: str,
    price: float | None = Query(default=None, gt=0),
    views: int | None = Query(default=None, ge=0),
    hour: datetime | None = None,
):
    df = load_product_data()
    product_rows = df[df["product_id"] == product_id]

    if product_rows.empty:
        raise HTTPException(status_code=404, detail="Product not found.")

    artifact = load_model_artifact()
    if product_id not in artifact["product_ids"]:
        raise HTTPException(
            status_code=404,
            detail="Product is not supported by the trained model.",
        )

    latest = product_rows.sort_values("hour").iloc[-1]
    price_used = float(price if price is not None else latest["price"])

    if hour is None:
        forecast_time = datetime.now(timezone.utc)
    elif hour.tzinfo is None:
        forecast_time = hour.replace(tzinfo=timezone.utc)
    else:
        forecast_time = hour.astimezone(timezone.utc)

    # The lagged model expects traffic from the previous hour, not views from
    # the hour being forecast. An explicit `views` parameter is retained for
    # compatibility and is interpreted as previous-hour views.
    if views is not None:
        previous_hour_views = int(views)
        traffic_source = "query_parameter"
    else:
        historical_value = latest.get("previous_hour_views", latest["views"])
        if pd.isna(historical_value):
            historical_value = latest["views"]
        previous_hour_views, traffic_source = get_previous_hour_views(
            product_id,
            forecast_time,
            int(historical_value),
        )

    log_price = float(np.log(price_used))
    row = {column: 0.0 for column in artifact["feature_columns"]}
    row["log_price"] = log_price
    traffic_feature = artifact.get("traffic_feature", "log_previous_hour_views")
    if traffic_feature not in row:
        raise HTTPException(
            status_code=500,
            detail=f"The loaded model expects an unsupported traffic feature: {traffic_feature}",
        )
    row[traffic_feature] = float(np.log1p(previous_hour_views))
    row["hour_sin"] = float(np.sin(2 * np.pi * forecast_time.hour / 24))
    row["hour_cos"] = float(np.cos(2 * np.pi * forecast_time.hour / 24))
    row["day_of_week"] = forecast_time.weekday()

    product_dummy = f"product_{product_id}"
    if product_dummy in row:
        row[product_dummy] = 1.0

    interaction = f"{product_dummy}_x_log_price"
    if interaction in row:
        row[interaction] = log_price

    X = pd.DataFrame([row], columns=artifact["feature_columns"])
    prediction = float(artifact["model"].predict(X)[0])

    return {
        "product_id": product_id,
        "predicted_hour": forecast_time.isoformat(),
        "price_used": price_used,
        "previous_hour_views_used": previous_hour_views,
        "traffic_source": traffic_source,
        "predicted_demand": round(max(0.0, prediction), 4),
        "model_alpha": artifact["alpha"],
        "model_version": artifact.get("model_version", "unknown"),
        "message": "Predicted demand represents estimated purchases for one hour. Traffic input is from the preceding hour.",
    }

@app.get("/api/products/{product_id}/pricing")
def recommend_price(
    product_id: str,
    price: float | None = Query(default=None, gt=0),
    views: int | None = Query(default=None, ge=0),
    hour: datetime | None = None,
):
    df = load_product_data()
    product_rows = df[df["product_id"] == product_id]

    if product_rows.empty:
        raise HTTPException(status_code=404, detail="Product not found.")

    artifact = load_model_artifact()
    if product_id not in artifact["product_ids"]:
        raise HTTPException(
            status_code=404,
            detail="Product is not supported by the trained model.",
        )

    latest = product_rows.sort_values("hour").iloc[-1]

    # An explicit price is useful for simulation. Otherwise use the accepted
    # active price, falling back to the latest historical price.
    with get_connection() as connection:
        active_record = connection.execute(
            """
            SELECT active_price, updated_at
            FROM active_prices
            WHERE product_id = ?
            """,
            (product_id,),
        ).fetchone()

    if price is not None:
        current_price = float(price)
    elif active_record is not None:
        current_price = float(active_record["active_price"])
    else:
        current_price = float(latest["price"])

    if hour is None:
        forecast_time = datetime.now(timezone.utc)
    elif hour.tzinfo is None:
        forecast_time = hour.replace(tzinfo=timezone.utc)
    else:
        forecast_time = hour.astimezone(timezone.utc)

    # Keep pricing compatible with the lagged-traffic model. Explicit `views`
    # means previous-hour views; otherwise use live events or historical fallback.
    if views is not None:
        previous_hour_views = int(views)
        traffic_source = "query_parameter"
    else:
        historical_value = latest.get("previous_hour_views", latest["views"])
        if pd.isna(historical_value):
            historical_value = latest["views"]
        previous_hour_views, traffic_source = get_previous_hour_views(
            product_id, forecast_time, int(historical_value)
        )

    feature_columns = artifact["feature_columns"]
    base_row = {column: 0.0 for column in feature_columns}
    base_row["log_price"] = float(np.log(current_price))
    traffic_feature = artifact.get("traffic_feature", "log_previous_hour_views")
    if traffic_feature not in base_row:
        raise HTTPException(
            status_code=500,
            detail=f"The loaded model expects an unsupported traffic feature: {traffic_feature}",
        )
    base_row[traffic_feature] = float(np.log1p(previous_hour_views))
    base_row["hour_sin"] = float(
        np.sin(2 * np.pi * forecast_time.hour / 24)
    )
    base_row["hour_cos"] = float(
        np.cos(2 * np.pi * forecast_time.hour / 24)
    )
    base_row["day_of_week"] = forecast_time.weekday()

    product_dummy = f"product_{product_id}"
    if product_dummy in base_row:
        base_row[product_dummy] = 1.0

    interaction = f"{product_dummy}_x_log_price"
    if interaction in base_row:
        base_row[interaction] = base_row["log_price"]

    # Load training-price bounds so recommendations remain in the range
    # represented by the training data.
    train_path = BASE_DIR.parent / "data" / "processed" / "train_demand.csv"
    if not train_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Training demand dataset was not found.",
        )

    train_df = pd.read_csv(train_path, usecols=["product_id", "price"])
    train_df["product_id"] = train_df["product_id"].astype(str)
    history = train_df.loc[
        train_df["product_id"] == product_id, "price"
    ].dropna()

    if history.empty:
        raise HTTPException(
            status_code=422,
            detail="No training price history for this product.",
        )

    historical_min = float(history.min())
    historical_max = float(history.max())

    # Retrieve decisions from the rolling 60-minute window. Timestamps are
    # stored in UTC ISO format by this API.
    now = datetime.now(timezone.utc)
    one_hour_ago = now.timestamp() - 3600
    cutoff_timestamp = datetime.fromtimestamp(
        one_hour_ago, timezone.utc
    ).isoformat()

    with get_connection() as connection:
        recent_decisions = connection.execute(
            """
            SELECT decision_timestamp, current_price, recommended_price
            FROM pricing_decisions
            WHERE product_id = ? AND decision_timestamp >= ?
            ORDER BY decision_timestamp ASC, id ASC
            """,
            (product_id, cutoff_timestamp),
        ).fetchall()

    # Anchor the entire rolling window to the current price recorded by its
    # earliest decision. This prevents a sequence of recommendations from
    # ratcheting more than 5% over the hour. With no recent decisions, the
    # first recommendation may use the normal +/-20% candidate range.
    if recent_decisions:
        anchor_price = float(recent_decisions[0]["current_price"])
        guardrail_min = anchor_price * 0.95
        guardrail_max = anchor_price * 1.05
    else:
        guardrail_min = historical_min
        guardrail_max = historical_max

    candidates = current_price * np.arange(0.80, 1.201, 0.05)
    candidates = candidates[
        (candidates >= historical_min)
        & (candidates <= historical_max)
        & (candidates >= guardrail_min)
        & (candidates <= guardrail_max)
    ]

    # If the normal candidate grid has no valid price, choose a safe value
    # from the intersection of historical bounds and the guardrail. Never
    # fall back to a price that violates either constraint.
    if len(candidates) == 0:
        safe_min = max(historical_min, guardrail_min)
        safe_max = min(historical_max, guardrail_max)
        if safe_min > safe_max:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No candidate price satisfies both the historical price "
                    "range and the rolling one-hour 5% guardrail."
                ),
            )
        safe_price = min(max(current_price, safe_min), safe_max)
        candidates = np.array([safe_price], dtype=float)

    candidate_features = pd.DataFrame(
        [base_row.copy() for _ in candidates],
        columns=feature_columns,
    )
    log_prices = np.log(candidates)
    candidate_features["log_price"] = log_prices
    if product_dummy in candidate_features.columns:
        candidate_features[product_dummy] = 1.0
    if interaction in candidate_features.columns:
        candidate_features[interaction] = log_prices

    predicted_demands = artifact["model"].predict(candidate_features)
    predicted_demands = np.maximum(predicted_demands, 0.0)
    expected_revenues = candidates * predicted_demands
    best_index = int(np.argmax(expected_revenues))

    recommended_price = float(candidates[best_index])
    predicted_demand = float(predicted_demands[best_index])
    expected_revenue = float(expected_revenues[best_index])
    decision_timestamp = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO pricing_decisions (
                product_id,
                decision_timestamp,
                current_price,
                recommended_price,
                predicted_demand,
                expected_revenue
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                product_id,
                decision_timestamp,
                current_price,
                recommended_price,
                predicted_demand,
                expected_revenue,
            ),
        )

    return {
        "product_id": product_id,
        "current_price": round(current_price, 2),
        "recommended_price": round(recommended_price, 2),
        "price_change_percent": round(
            (recommended_price / current_price - 1) * 100, 2
        ),
        "predicted_hourly_demand": round(predicted_demand, 4),
        "expected_hourly_revenue": round(expected_revenue, 2),
        "model_alpha": artifact["alpha"],
        "model_version": artifact.get("model_version", "unknown"),
        "previous_hour_views_used": previous_hour_views,
        "traffic_source": traffic_source,
        "note": (
            "Revenue is model-estimated, not guaranteed. When a pricing "
            "decision exists in the previous 60 minutes, recommendations are "
            "limited to within 5% of the price recorded at the start of that "
            "rolling window and remain within the historical training-price range."
        ),
    }

@app.get("/api/pricing-decisions")

def list_pricing_decisions(limit: int = 50):

    if not 1 <= limit <= 500:

        raise HTTPException(

            status_code=422,

            detail="limit must be between 1 and 500"

        )



    with get_connection() as connection:

        rows = connection.execute(

            """

            SELECT *

            FROM pricing_decisions

            ORDER BY id DESC

            LIMIT ?

            """,

            (limit,)

        ).fetchall()



    return {

        "count": len(rows),

        "decisions": [dict(row) for row in rows]

    }

@app.post("/api/products/{product_id}/accept-price")

def accept_recommended_price(

    product_id: str,

    accepted_price: float = Query(gt=0),

):

    df = load_product_data()



    if product_id not in df["product_id"].values:

        raise HTTPException(status_code=404, detail="Product not found.")



    artifact = load_model_artifact()



    if product_id not in artifact["product_ids"]:

        raise HTTPException(

            status_code=404,

            detail="Product is not supported by the trained model."

        )



    with get_connection() as connection:

        decision = connection.execute(

            """

            SELECT recommended_price

            FROM pricing_decisions

            WHERE product_id = ?

            ORDER BY decision_timestamp DESC, id DESC

            LIMIT 1

            """,

            (product_id,),

        ).fetchone()



        if decision is None:

            raise HTTPException(

                status_code=404,

                detail="No pricing recommendation exists for this product."

            )



        if not np.isclose(

            accepted_price,

            float(decision["recommended_price"]),

            rtol=0,

            atol=0.01,

        ):

            raise HTTPException(

                status_code=400,

                detail="Accepted price must match the latest recommendation."

            )



        updated_at = datetime.now(timezone.utc).isoformat()



        connection.execute(

            """

            INSERT INTO active_prices (product_id, active_price, updated_at)

            VALUES (?, ?, ?)

            ON CONFLICT(product_id) DO UPDATE SET

                active_price = excluded.active_price,

                updated_at = excluded.updated_at

            """,

            (product_id, accepted_price, updated_at),

        )



    return {

        "product_id": product_id,

        "active_price": accepted_price,

        "updated_at": updated_at,

        "message": "Recommended price accepted as the active price."

    }

@app.get("/api/products/{product_id}/active-price")

def get_active_price(product_id: str):

    df = load_product_data()



    if product_id not in df["product_id"].values:

        raise HTTPException(

            status_code=404,

            detail="Product not found."

        )



    with get_connection() as connection:

        row = connection.execute(

            """

            SELECT product_id, active_price, updated_at

            FROM active_prices

            WHERE product_id = ?

            """,

            (product_id,)

        ).fetchone()



    if row is None:

        raise HTTPException(

            status_code=404,

            detail="No active price has been accepted for this product."

        )



    return {

        "product_id": row["product_id"],

        "active_price": row["active_price"],

        "updated_at": row["updated_at"],

        "message": "Retrieved the currently accepted active price."

    }
