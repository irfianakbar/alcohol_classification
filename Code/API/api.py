from typing import Any, Dict
from engine.main import ModelHandler 
import uvicorn
from pydantic import BaseModel
from fastapi import BackgroundTasks, FastAPI
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()
model_funct = ModelHandler()

class BodyRequest(BaseModel):
    record_id: str
    model_input: Dict

class BodyResponse(BaseModel):
    record_id: str
    prediction_class: str


@app.post('/XGBoost/', response_model=BodyResponse)
async def xgboost_model(body: BodyRequest):
    data = body.dict()
   
    pred_result = model_funct.xgb_pred(data['model_input'])
    respond_out = {
        "record_id": data['record_id'],
        "prediction_class": pred_result
    }
    return respond_out


@app.post('/ANN/', response_model=BodyResponse)
async def ANN_model(body: BodyRequest):
    data = body.dict()
   
    pred_result = model_funct.ann_pred(data['model_input'])
    respond_out = {
        "record_id": data['record_id'],
        "prediction_class": pred_result
    }
    return respond_out


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, log_level="info", reload=True)
