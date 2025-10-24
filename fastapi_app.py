import json
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Dict, Optional
import joblib
import numpy as np
import pandas as pd

from models import (
	UserFeatures, PredictionRequest, PredictionResponse,
	HealthResponse, InfoResponse, ErrorResponse, UserPrediction
)	

# from models import (
#     UserFeatures, PredictionRequest, PredictionResponse, 
#     HealthResponse, InfoResponse, ErrorResponse,
# )
# from util import make_prediction
from sklearn.pipeline import Pipeline
import traceback
from contextlib import asynccontextmanager

SERVICE_NAME = "Employee Attrition Prediction Service"

@asynccontextmanager
async def lifespan(app: FastAPI):
	# Startup code
	app.state.start_time = time.time()
	try:
		# Load the model during startup
		# app.state.model: Pipeline = joblib.load("attrition_model_pipeline.joblib")
		# print("Model loaded successfully.")
		print ("Server started successfully.")
	except Exception as e:
		print(f"Error loading model: {e}")
		raise e
	yield
	# Shutdown code (if any)
	print("Server shutting down.")

app = FastAPI(
	title=SERVICE_NAME,
	lifespan=lifespan
)

@app.post("/predict", response_model=PredictionResponse, responses={500: {"model": ErrorResponse}})
async def predict(request: PredictionRequest):
	try:
		# Convert list of UserFeatures to DataFrame
		# features_list = [feature.model_dump() for feature in request.features]
		# input_df = pd.DataFrame(features_list)

		# No model yet, we simulate predictions
		predictions = [UserPrediction(user_id=feature.UserId, prediction="No Prediction Yet") for feature in request.features]
		# Provide required fields "message" and "recommendations" expected by PredictionResponse
		return PredictionResponse(
			status="success",
			message="Predictions generated successfully.",
			predictions=predictions,
			recommendations="No recommendation yet"  # added missing required parameter
		)
	except Exception as e:
		error_details = traceback.format_exc()
		raise HTTPException(
			status_code=500,
			detail=ErrorResponse(
				status="error",
				message="An error occurred during prediction.",
				error_details=error_details
			).model_dump()
		)

@app.get("/health", response_model=HealthResponse)
async def health_check():
	uptime = time.time() - app.state.start_time
	return HealthResponse(
		status="healthy",
		message="Service is up and running.",
		uptime=uptime,
	)

@app.get("/info", response_model=InfoResponse)
async def service_info():
	return InfoResponse(
		status="success",
		message="Service information retrieved successfully.",
		service_name=SERVICE_NAME,
		model_version="1.0.0",
		model_type="RandomForestClassifier"
	)