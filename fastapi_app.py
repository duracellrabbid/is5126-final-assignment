import json
import os
from random import random, choice
import time
from fastapi import FastAPI, HTTPException
from typing import Any, Dict, List, Optional, final
import joblib
import numpy as np
import pandas as pd

from models import (
	RecommendationRequest, UserFeatures, PredictionRequest, PredictionResponse,
	HealthResponse, InfoResponse, ErrorResponse, UserPrediction
)	

from sklearn.pipeline import Pipeline
import traceback
from contextlib import asynccontextmanager
from dotenv import load_dotenv


def load_env_variables():
	load_dotenv()
	open_api_key = os.environ.get('OPENAI_API_KEY', '')
	print("Environment variables loaded.")

	return open_api_key

open_api_key = load_env_variables()
if open_api_key:
	print("OpenAI API key loaded successfully.")

from rag_and_llm.agents import debate
from rag_and_llm.messages import pretty_print_messages

SERVICE_NAME = "Employee Attrition Prediction Service"

def load_model() -> Optional[Pipeline]:
	"""
	Load the pre-trained model pipeline from a joblib file.
	"""
	try:
		print("Loading model pipeline...")
		model = joblib.load("final_model_pipeline.pkl")
	except Exception as e:
		print(f"Failed to load model pipeline: {e}")
		model = None
	return model

prediction_pipeline: Optional[Pipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
	# Startup code
	global prediction_pipeline
	app.state.start_time = time.time()
	try:
		# Load the model during startup

		prediction_pipeline = load_model()
		if prediction_pipeline is not None:
			print("Model loaded successfully.")

		print("Server started successfully.")
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
	df = pd.DataFrame(features_list)
	df = df.drop(columns=["UserId"])  # Drop UserId for prediction
	return df

def perform_mock_prediction(input_df: pd.DataFrame) -> np.ndarray:
	"""
	Perform a mock prediction by returning a probability for the positive class for each row.
	Replace this with model.predict_proba(input_df)[:, 1] when a real model is loaded.
	"""
	# Use a reproducible random generator for mock probabilities
	# Provide an explicit seed to satisfy the requirement and ensure reproducibility.
	rng = np.random.default_rng(42)
	probs = rng.random(size=len(input_df))


	
	return probs
def get_ai_messages(messages: Any, sender_name: Optional[str] = "director_agent"):
	"""Print all message contents in order, with sender name if available."""
	message_list = []
	for i, msg in enumerate(messages, start=1):
		sender = getattr(msg, "name", None)
		msg_type = type(msg).__name__
		content = getattr(msg, "content", "")

		print(f"\n[{i}] {msg_type}{' (' + sender + ')' if sender else ''}:")
		print(content or "(empty)")
		if sender == sender_name:
			message_list.append(content)
	return message_list

def make_recommendation(recommendation_request: RecommendationRequest) -> str:
	"""
	Generate recommendations based on user features and predictions.
	"""
	# Placeholder recommendation logic
	schema = RecommendationRequest.model_json_schema()
	# Here you would integrate with an LLM or other logic to generate recommendations
	prompt = f"""
You are a HR consultant specializing in employee retention. You are given a schema, user features, and their attrition predictions.

Based on the following features and predictions, provide recommendations to reduce attrition:
Schema: {schema}
Features: {recommendation_request.features}
Predictions: {recommendation_request.predictions}

Provide actionable recommendations for the company based on the above data. Justify the recommendations with reference to the features and predictions.

Do not provide recommendation for individual users; focus on overall strategies to improve retention.
"""
	message = { "messages": [
			{
				"role": "user",
				"content": prompt,
			}
		]
	}

	print("Starting debate among agents for recommendation...")
	for chunk in debate.stream(message):
		pretty_print_messages(chunk, last_message=True)

	final_message_history = chunk["director_agent"]["messages"]
	print("Final message history obtained from debate.\n\n")
	message_list = get_ai_messages(final_message_history)
	print("\n\nRecommendation generation completed.")
	return "\n".join(message_list)

@app.post("/predict", response_model=PredictionResponse, responses={500: {"model": ErrorResponse}})
async def predict(request: PredictionRequest):
	try:
		# Convert list of UserFeatures to DataFrame
		input_df = convert_features_to_dataframe(request.features)
		# Get mock probabilities for each row
		if prediction_pipeline is not None:

			probs = prediction_pipeline.predict_proba(input_df)[:, 1]
			preds = prediction_pipeline.predict(input_df)
			predictions = []
			for feature, prob, pred in zip(request.features, probs, preds):
				predictions.append(
					UserPrediction(
						user_id=feature.UserId,
						prediction=pred,
						probability=float(round(float(prob), 4))
					)
				)
			recommendation_request = RecommendationRequest(
				features=request.features,
				predictions=predictions
			)
			recommendation = "No recommendation yet"
			try:
				recommendation = make_recommendation(recommendation_request)
			except Exception as e:
				print(f"Error generating recommendations: {e}")
			return PredictionResponse(
				status="success",
				message="Predictions generated successfully.",
				predictions=predictions,
				recommendations=recommendation
			)
		else:
			raise Exception("Prediction pipeline is not loaded.")

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
		model_status="loaded" if prediction_pipeline is not None else "not loaded",
		uptime=uptime,
	)

@app.get("/info", response_model=InfoResponse)
async def service_info():
	return InfoResponse(
		status="success",
		message="Service information retrieved successfully.",
		service_name=SERVICE_NAME,
		model_version="1.0.0",
		model_type="Logistic Regression",
	)