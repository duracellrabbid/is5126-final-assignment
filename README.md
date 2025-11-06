# Set-up
1. Run the following command:
```
pip3 install -r requirements.txt
```

2. Download and install Ollama from https://ollama.com/downloa

3. Download embedding model from Ollama by running:

```bash
ollama pull mxbai-embed-large
```

4. Run IS5126_FinalGroupProject.ipynb to download dataset and perform EDA and classification model.

5. IS5126_FinalGroupProject.ipynb will also output the files necessary for the frontend and backend server.

6. To run the FastAPI server, run the following command:
```
uvicorn fastapi_app:app --reload --port 5001
```

1. To run the Streamlit server, run the following command:
```
streamlit run streamlit_client.py
```

1. Open the following URL in a web browser:
```
http://localhost:8501
```

1. You can use the sample dataset **sample_userfeatures_20rows.csv**
 for prediction:

# URLs 
## FastAPI server
1. http://localhost:5001/predict
2. http://localhost:5001/info
3. http://localhost:5001/health
4. http://localhost:5001/docs

## Streamlit server
1. http://localhost:8501


# Directory structure
```
is5126-final-assignment/
├── README.md
├── requirements.txt
├── IS5126_FinalGroupProject.ipynb
├── fastapi_app.py
├── streamlit_client.py
├── utils.py
├── models.py
├── final_model_pipeline.pkl
├── sample_userfeatures_20rows.csv
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── rag_and_llm/
│   └── agents
|	└── rag
├── .env
```
