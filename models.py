from email import message
from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class UserFeatures(BaseModel):
	UserId: str = Field(..., description="Unique identifier for the user")
	Age: int = Field(..., description="Age of the employee")
	Attrition: Optional[str] = Field(None, description="Whether the employee left the company")
	BusinessTravel: str = Field(..., description="Business travel frequency")
	DailyRate: int = Field(..., description="Daily rate of the employee")
	Department: str = Field(..., description="Department of the employee")
	DistanceFromHome: int = Field(..., description="Distance from home")
	Education: int = Field(..., description="Education level")
	EducationField: str = Field(..., description="Field of education")
	EmployeeCount: int = Field(..., description="Number of employees in the company")
	EmployeeNumber: int = Field(..., description="Unique identifier for the employee")
	EnvironmentSatisfaction: int = Field(..., description="Environment satisfaction rating")
	JobInvolvement: int = Field(..., description="Job involvement rating")
	JobLevel: int = Field(..., description="Job level")
	JobRole: str = Field(..., description="Job role")
	MaritalStatus: str = Field(..., description="Marital status")
	MonthlyIncome: int = Field(..., description="Monthly income")
	NumCompaniesWorked: int = Field(..., description="Number of companies worked")
	OverTime: str = Field(..., description="Overtime status")
	PercentSalaryHike: int = Field(..., description="Percentage salary hike")
	PerformanceRating: int = Field(..., description="Performance rating")
	RelationshipSatisfaction: int = Field(..., description="Relationship satisfaction")
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
	prediction: str = Field(..., description="Prediction result for the user")

class PredictionResponse(BaseResponse):
	predictions: List[UserPrediction] = Field(..., description="List of predictions for the input features")
	recommendations: Optional[str] = Field(None, description="Optional recommendations based on predictions")

class HealthResponse(BaseResponse):
	uptime: float = Field(..., description="Service uptime in seconds")
	status: str = Field(..., description="Health status of the service")	
 
class InfoResponse(BaseResponse):
	service_name: str = Field(..., description="Name of the service")
	model_version: str = Field(..., description="Version of the deployed model")
	model_type: str = Field(..., description="Type of the deployed model")

class ErrorResponse(BaseResponse):
	error_details: Optional[str] = Field(None, description="Details about the error encountered")