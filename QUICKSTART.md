# Quick Start Guide - Employee Attrition Prediction

## 🚀 Get Started in 3 Steps

### Step 1: Start the FastAPI Backend
Open a terminal and run:
```powershell
uvicorn fastapi_app:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

### Step 2: Start the Streamlit Client
Open a **new terminal** and run:
```powershell
streamlit run streamlit_client.py
```

Your browser should automatically open to `http://localhost:8501`

### Step 3: Upload and Predict
1. Click "Browse files" button
2. Select `sample_employee_data.csv` (provided in the project)
3. Review the data preview
4. Click "🔮 Generate Predictions"
5. Download your results!

## 📝 Testing with Sample Data

The project includes `sample_employee_data.csv` with 5 sample employees. Use this file to test the application immediately!

## ⚙️ Configuration

### Default Settings
- **FastAPI Server**: http://localhost:8000
- **Streamlit Client**: http://localhost:8501

### Change API URL
If your FastAPI server runs on a different URL:
1. Open the Streamlit app
2. Look at the sidebar on the left
3. Update the "API Base URL" field
4. Click "Check Health" to verify connection

## 🔍 Verify Everything Works

### Check Backend Health
In the Streamlit app sidebar:
1. Click "Check Health" button
2. You should see "✅ Service is healthy"
3. View the uptime metric

### Check Service Info
In the Streamlit app sidebar:
1. Click "Get Info" button
2. You should see:
   - Service name
   - Model version
   - Model type

## 📊 Supported File Formats

✅ CSV files (.csv)
✅ Excel files (.xlsx, .xls)
❌ Other formats will show an error

## 🐛 Common Issues

### Issue: "Connection Error: Could not connect to API"
**Solution**: Make sure the FastAPI server is running in a separate terminal

### Issue: "Missing required columns"
**Solution**: Use the sample CSV file or ensure your file has all required columns (see README_STREAMLIT.md)

### Issue: "Port already in use"
**Solution**: 
- For FastAPI: Change port with `--port 8001`
- For Streamlit: It will automatically try the next available port

## 📚 Need More Help?

See `README_STREAMLIT.md` for:
- Complete list of required columns
- Detailed feature documentation
- Troubleshooting guide
- Architecture diagram

## 🎯 What You Can Do

- ✅ Upload CSV/Excel files with employee data
- ✅ Get instant attrition predictions
- ✅ Download predictions as CSV
- ✅ Monitor API health and status
- ✅ Process multiple employees in one batch

Enjoy predicting employee attrition! 🎉
