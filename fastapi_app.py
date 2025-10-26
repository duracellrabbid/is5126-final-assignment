import json
from random import random, choice
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

def convert_features_to_dataframe(user_features_list: list[UserFeatures]) -> pd.DataFrame:
	"""
	Convert a list of UserFeatures to a pandas DataFrame.
	"""
	features_list = [feature.model_dump() for feature in user_features_list]
	return pd.DataFrame(features_list)

def perform_prediction(input_df: pd.DataFrame) -> np.ndarray:
	"""
	Perform a mock prediction by returning a probability for the positive class for each row.
	Replace this with model.predict_proba(input_df)[:, 1] when a real model is loaded.
	"""
	# Use a reproducible random generator for mock probabilities
	# Provide an explicit seed to satisfy the requirement and ensure reproducibility.
	rng = np.random.default_rng(42)
	probs = rng.random(size=len(input_df))
	return probs

@app.post("/predict", response_model=PredictionResponse, responses={500: {"model": ErrorResponse}})
async def predict(request: PredictionRequest):
	try:
		# Convert list of UserFeatures to DataFrame
		input_df = convert_features_to_dataframe(request.features)

		# Get mock probabilities for each row
		probabilities = perform_prediction(input_df)

		# Build UserPrediction list: label = 1 if prob >= 0.5 else 0
		predictions = []
		for feature, prob in zip(request.features, probabilities):
			label = 1 if prob >= 0.5 else 0
			predictions.append(
				UserPrediction(
					user_id=feature.UserId,
					prediction=label,
					probability=float(round(float(prob), 4))
				)
			)

		return PredictionResponse(
			status="success",
			message="Predictions generated successfully.",
			predictions=predictions,
			recommendations="No recommendation yet"
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