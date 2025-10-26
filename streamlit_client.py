import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from models import UserFeatures, PredictionRequest, PredictionResponse, HealthResponse, InfoResponse, ErrorResponse
import json
from typing import Optional
import io

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:5001"

def validate_file_type(uploaded_file) -> tuple[bool, str]:
    """Validate if the uploaded file is CSV or Excel"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    valid_extensions = ['csv', 'xlsx', 'xls']
    
    if file_extension not in valid_extensions:
        return False, f"Invalid file type: .{file_extension}. Please upload a CSV or Excel file (.csv, .xlsx, .xls)"
    
    return True, "Valid file type"

def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load data from uploaded CSV or Excel file"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            return None
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def validate_dataframe_columns(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate that the dataframe has all required columns for UserFeatures"""
    required_columns = [
        'UserId', 'Age', 'BusinessTravel', 'DailyRate', 'Department', 
        'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
        'EmployeeNumber', 'EnvironmentSatisfaction', 'JobInvolvement', 
        'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome', 
        'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 
        'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
        'YearsWithCurrManager'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    return True, "All required columns present"

def convert_df_to_prediction_request(df: pd.DataFrame) -> PredictionRequest:
    """Convert DataFrame to PredictionRequest with list of UserFeatures"""
    features_list = []
    
    for _, row in df.iterrows():
        user_feature = UserFeatures(
            UserId=str(row['UserId']),
            Age=int(row['Age']),
            Attrition=str(row.get('Attrition', None)) if pd.notna(row.get('Attrition')) else None,
            BusinessTravel=str(row['BusinessTravel']),
            DailyRate=int(row['DailyRate']),
            Department=str(row['Department']),
            DistanceFromHome=int(row['DistanceFromHome']),
            Education=int(row['Education']),
            EducationField=str(row['EducationField']),
            EmployeeCount=int(row['EmployeeCount']),
            EmployeeNumber=int(row['EmployeeNumber']),
            EnvironmentSatisfaction=int(row['EnvironmentSatisfaction']),
            JobInvolvement=int(row['JobInvolvement']),
            JobLevel=int(row['JobLevel']),
            JobRole=str(row['JobRole']),
            MaritalStatus=str(row['MaritalStatus']),
            MonthlyIncome=int(row['MonthlyIncome']),
            NumCompaniesWorked=int(row['NumCompaniesWorked']),
            OverTime=str(row['OverTime']),
            PercentSalaryHike=int(row['PercentSalaryHike']),
            PerformanceRating=int(row['PerformanceRating']),
            RelationshipSatisfaction=int(row['RelationshipSatisfaction']),
            WorkLifeBalance=int(row['WorkLifeBalance']),
            YearsAtCompany=int(row['YearsAtCompany']),
            YearsInCurrentRole=int(row['YearsInCurrentRole']),
            YearsSinceLastPromotion=int(row['YearsSinceLastPromotion']),
            YearsWithCurrManager=int(row['YearsWithCurrManager'])
        )
        features_list.append(user_feature)
    
    return PredictionRequest(features=features_list)

def call_prediction_api(prediction_request: PredictionRequest) -> Optional[PredictionResponse]:
    """Call the FastAPI prediction endpoint"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=prediction_request.model_dump(),
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return PredictionResponse(**response.json())
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to API at {API_BASE_URL}. Make sure the FastAPI server is running.")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def check_service_health():
    """Check if the API service is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = HealthResponse(**response.json())
            return True, health_data
        return False, None
    except Exception:
        return False, None

def main():
    st.title("👥 Employee Attrition Prediction Client")
    st.markdown("---")
    
    # Sidebar for API configuration and health check
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API URL configuration
        global API_BASE_URL
        API_BASE_URL = st.text_input("API Base URL", value=API_BASE_URL)
        
        st.markdown("---")
        st.header("🏥 Service Health")
        
        if st.button("Check Health"):
            is_healthy, health_data = check_service_health()
            if is_healthy and health_data:
                st.success("✅ Service is healthy")
                st.metric("Uptime", f"{health_data.uptime:.2f}s")
            else:
                st.error("❌ Service is unavailable")
        
        st.markdown("---")
        st.header("📋 Service Info")
        if st.button("Get Info"):
            try:
                response = requests.get(f"{API_BASE_URL}/info")
                if response.status_code == 200:
                    info = InfoResponse(**response.json())
                    st.write(f"**Service:** {info.service_name}")
                    st.write(f"**Model Version:** {info.model_version}")
                    st.write(f"**Model Type:** {info.model_type}")
            except Exception as e:
                st.error(f"Could not fetch service info: {str(e)}")
    
    # Main content area
    st.header("📁 Upload Employee Data")
    st.write("Upload a CSV or Excel file containing employee data for attrition prediction.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with employee data"
    )
    
    if uploaded_file is not None:
        # Validate file type
        is_valid, message = validate_file_type(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {message}")
            st.stop()
        
        st.success(f"✅ {message}")
        
        # Load data
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
        
        if df is None:
            st.error("Failed to load data from the uploaded file.")
            st.stop()
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("File Type", uploaded_file.name.split('.')[-1].upper())
        
        # Validate columns
        is_valid_cols, col_message = validate_dataframe_columns(df)
        
        if not is_valid_cols:
            st.error(f"❌ {col_message}")
            st.write("**Uploaded columns:**")
            st.write(list(df.columns))
            st.stop()
        
        st.success(f"✅ {col_message}")
        
        # Display data preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10), width='stretch')
        
        # Show data statistics
        with st.expander("📈 View Data Statistics"):
            st.write(df.describe())
        
        # Predict button
        st.markdown("---")

        if st.button("🔮 Generate Predictions", type="primary", width='stretch'):
            try:
                # Convert to PredictionRequest
                with st.spinner("Preparing prediction request..."):
                    prediction_request = convert_df_to_prediction_request(df)
                
                st.info(f"Sending {len(prediction_request.features)} employee records for prediction...")
                
                # Call API
                with st.spinner("Calling prediction API..."):
                    prediction_response = call_prediction_api(prediction_request)
                
                if prediction_response:
                    st.success("✅ Predictions received successfully!")
                    
                    # Display predictions
                    st.subheader("🎯 Prediction Results")
                    
                    # Convert predictions to DataFrame
                    predictions_data = [
                        {
                            "User ID": pred.user_id,
                            "Prediction": pred.prediction,
                            "Probability": pred.probability
                        }
                        for pred in prediction_response.predictions
                    ]
                    predictions_df = pd.DataFrame(predictions_data)

                    st.dataframe(predictions_df, width='stretch')

                    # Show recommendations if available
                    if prediction_response.recommendations:
                        st.subheader("💡 Recommendations")
                        st.info(prediction_response.recommendations)
                    
                    # Download predictions
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a CSV or Excel file to get started.")
        
        with st.expander("📖 Required Columns"):
            required_cols = [
                'UserId', 'Age', 'BusinessTravel', 'DailyRate', 'Department', 
                'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
                'EmployeeNumber', 'EnvironmentSatisfaction', 'JobInvolvement', 
                'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome', 
                'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 
                'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance',
                'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                'YearsWithCurrManager'
            ]
            st.write("Your file must contain the following columns:")
            st.write(required_cols)

if __name__ == "__main__":
    main()
