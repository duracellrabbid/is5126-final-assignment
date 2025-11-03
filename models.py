from email import message
from pyexpat import features
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Literal


class UserFeatures(BaseModel):
	UserId: str = Field(..., description="Unique identifier for the user")
	Age: int = Field(..., description="Age of the employee")
	BusinessTravel: str = Field(..., description="Business travel frequency")
	DailyRate: int = Field(..., description="Daily rate of the employee")
	Department: str = Field(..., description="Department of the employee")
	DistanceFromHome: int = Field(..., description="Distance from home")
	Education: int = Field(..., description="Education level")
	EducationField: str = Field(..., description="Field of education")
	EmployeeNumber: int = Field(..., description="Unique identifier for the employee")
	EnvironmentSatisfaction: int = Field(..., description="Environment satisfaction rating")
	Gender : str = Field(..., description="Gender of the employee")
	HourlyRate: int = Field(..., description="Hourly rate of the employee")
	JobInvolvement: int = Field(..., description="Job involvement rating")
	JobLevel: int = Field(..., description="Job level")
	JobRole: str = Field(..., description="Job role")
	JobSatisfaction: int = Field(..., description="Job satisfaction rating")
	MaritalStatus: str = Field(..., description="Marital status")
	MonthlyIncome: int = Field(..., description="Monthly income")
	MonthlyRate: int = Field(..., description="Monthly rate")
	NumCompaniesWorked: int = Field(..., description="Number of companies worked")
	Over18: str = Field(..., description="Over 18 status")
	OverTime: str = Field(..., description="Overtime status")
	PercentSalaryHike: int = Field(..., description="Percentage salary hike")
	PerformanceRating: int = Field(..., description="Performance rating")
	RelationshipSatisfaction: int = Field(..., description="Relationship satisfaction")
	StandardHours: int = Field(..., description="Standard working hours")
	StockOptionLevel: int = Field(..., description="Stock option level")
	TotalWorkingYears: int = Field(..., description="Total working years")
	TrainingTimesLastYear: int = Field(..., description="Training times in the last year")
	WorkLifeBalance: int = Field(..., description="Work-life balance")
	YearsAtCompany: int = Field(..., description="Years at the company")
	YearsInCurrentRole: int = Field(..., description="Years in current role")
	YearsSinceLastPromotion: int = Field(..., description="Years since last promotion")
	YearsWithCurrManager: int = Field(..., description="Years with current manager")

class PredictionRequest(BaseModel):
	features: List[UserFeatures] = Field(..., description="List of user features for prediction")


class BaseResponse(BaseModel):
	status: str = Field(..., description="Status of the response")
	message: Optional[str] = Field(None, description="Optional message providing additional information")
class UserPrediction(BaseModel):
	user_id: str = Field(..., description="Unique identifier for the user")
	prediction: Literal[0, 1] = Field(..., description="Prediction result for the user (0 or 1)")
	probability: float = Field(..., description="Probability associated with the prediction")

class PredictionResponse(BaseResponse):
	predictions: List[UserPrediction] = Field(..., description="List of predictions for the input features")
	recommendations: Optional[str] = Field(None, description="Optional recommendations based on predictions")

class HealthResponse(BaseResponse):
	uptime: float = Field(..., description="Service uptime in seconds")
	status: str = Field(..., description="Health status of the service")	
	model_status: str = Field(..., description="Status of the model")
 
class InfoResponse(BaseResponse):
	service_name: str = Field(..., description="Name of the service")
	model_version: str = Field(..., description="Version of the deployed model")
	model_type: str = Field(..., description="Type of the deployed model")

class ErrorResponse(BaseResponse):
	error_details: Optional[str] = Field(None, description="Details about the error encountered")

class RecommendationRequest(BaseModel):
	features: List[UserFeatures] = Field(..., description="List of user features for generating recommendations")
	predictions: List[UserPrediction] = Field(..., description="List of user predictions corresponding to the features")