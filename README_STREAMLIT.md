# Employee Attrition Prediction - Streamlit Client

A user-friendly web interface for predicting employee attrition using the FastAPI backend service.

## Features

✨ **File Upload Support**
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- Automatic file type validation
- Error handling for invalid file formats

📊 **Data Validation**
- Validates all required columns are present
- Shows data preview and statistics
- Displays file information (rows, columns, type)

🔮 **Predictions**
- Batch prediction for multiple employees
- Real-time API communication
- Download predictions as CSV
- View recommendations (if available)

🏥 **Service Monitoring**
- Health check functionality
- Service information display
- Uptime monitoring

## Installation

1. Install required dependencies:
```bash
pip install streamlit requests pandas openpyxl
```

## Running the Application

1. **Start the FastAPI backend** (in one terminal):
```bash
uvicorn fastapi_app:app --reload --port 8000
```

2. **Start the Streamlit client** (in another terminal):
```bash
streamlit run streamlit_client.py
```

3. Open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## How to Use

### 1. Upload a File
- Click on "Browse files" button
- Select a CSV or Excel file containing employee data
- The app will validate the file type automatically

### 2. Review Your Data
- View the data preview (first 10 rows)
- Check data statistics in the expandable section
- Verify all required columns are present

### 3. Generate Predictions
- Click the "🔮 Generate Predictions" button
- Wait for the API to process your data
- View the predictions in a table format

### 4. Download Results
- Click "📥 Download Predictions as CSV" to save results
- The file will be named with a timestamp for easy tracking

## Required Columns

Your CSV/Excel file must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| UserId | string | Unique identifier for the employee |
| Age | integer | Employee age |
| BusinessTravel | string | Travel frequency (e.g., Travel_Rarely) |
| DailyRate | integer | Daily rate |
| Department | string | Department name |
| DistanceFromHome | integer | Distance from home in miles/km |
| Education | integer | Education level (1-5) |
| EducationField | string | Field of education |
| EmployeeCount | integer | Employee count |
| EmployeeNumber | integer | Unique employee number |
| EnvironmentSatisfaction | integer | Satisfaction level (1-4) |
| JobInvolvement | integer | Involvement level (1-4) |
| JobLevel | integer | Job level (1-5) |
| JobRole | string | Current job role |
| MaritalStatus | string | Marital status |
| MonthlyIncome | integer | Monthly income |
| NumCompaniesWorked | integer | Number of companies worked |
| OverTime | string | Overtime (Yes/No) |
| PercentSalaryHike | integer | Salary hike percentage |
| PerformanceRating | integer | Performance rating (1-4) |
| RelationshipSatisfaction | integer | Satisfaction level (1-4) |
| WorkLifeBalance | integer | Balance rating (1-4) |
| YearsAtCompany | integer | Years at company |
| YearsInCurrentRole | integer | Years in current role |
| YearsSinceLastPromotion | integer | Years since last promotion |
| YearsWithCurrManager | integer | Years with current manager |

**Optional:**
- Attrition (string): Yes/No - can be included for comparison

## Sample Data

A sample CSV file (`sample_employee_data.csv`) is provided in the project directory. You can use this to test the application.

## Configuration

### API URL
You can change the API base URL in the sidebar:
- Default: `http://localhost:8000`
- Useful for connecting to remote servers or different ports

## Troubleshooting

### "Connection Error: Could not connect to API"
- Ensure the FastAPI server is running on the specified URL
- Check the API URL in the sidebar configuration
- Verify your firewall settings

### "Missing required columns" error
- Check that your file contains all required columns (see table above)
- Column names are case-sensitive and must match exactly
- View the list of uploaded columns in the error message

### "Invalid file type" error
- Only CSV (.csv) and Excel (.xlsx, .xls) files are supported
- Check the file extension
- Ensure the file is not corrupted

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  Streamlit  │ ──────> │   FastAPI    │ ──────> │   Model     │
│   Client    │ <────── │   Backend    │ <────── │  Pipeline   │
└─────────────┘         └──────────────┘         └─────────────┘
     Web UI              RESTful API           ML Predictions
```

## Error Handling

The application handles various error scenarios:
- ✅ Invalid file formats
- ✅ Missing columns
- ✅ API connection issues
- ✅ Data type mismatches
- ✅ Service unavailability

All errors are displayed with clear, actionable messages.

## Features in Sidebar

### Service Health Check
- Click "Check Health" to verify API is running
- Shows service uptime in seconds
- Displays status (healthy/unavailable)

### Service Information
- Click "Get Info" to view:
  - Service name
  - Model version
  - Model type

## Tips for Best Results

1. **Data Quality**: Ensure your data is clean and complete
2. **Column Names**: Match column names exactly (case-sensitive)
3. **Data Types**: Ensure numeric columns contain numbers
4. **File Size**: For large files, consider batching into smaller files
5. **Testing**: Use the sample data file to verify setup

## Future Enhancements

Potential features for future versions:
- Data preprocessing options
- Visualization of predictions
- Bulk file upload
- Export to Excel
- Historical prediction tracking
- User authentication
