import os
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import json
from pathlib import Path

from utils.data_cleaning import DataCleaner
from utils.model_training import AutoMLTrainer
from utils.evaluation import ModelEvaluator
from utils.output_generator import OutputGenerator

app = FastAPI(title="AutoML Pipeline Builder", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
OUTPUTS_DIR = Path("outputs")

for dir_path in [UPLOAD_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# In-memory storage for job status (use Redis in production)
job_status: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def root():
    return {"message": "AutoML Pipeline Builder API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file and perform initial validation"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Validate and get basic info about the dataset
        df = pd.read_csv(file_path)
        
        # Basic validation
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty dataset")
        
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="Dataset must have at least 2 columns")
        
        # Store job info
        job_status[job_id] = {
            "status": "uploaded",
            "filename": file.filename,
            "file_path": str(file_path),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "job_id": job_id,
            "status": "uploaded",
            "dataset_info": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "missing_values": df.isnull().sum().to_dict(),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/preview/{job_id}")
async def preview_data(job_id: str, rows: int = 5):
    """Get preview of uploaded dataset"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        file_path = job_status[job_id]["file_path"]
        df = pd.read_csv(file_path)
        
        return {
            "head": df.head(rows).to_dict(orient="records"),
            "tail": df.tail(rows).to_dict(orient="records"),
            "shape": df.shape,
            "info": {
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error previewing data: {str(e)}")

@app.post("/analyze/{job_id}")
async def analyze_data(job_id: str, background_tasks: BackgroundTasks):
    """Perform EDA and start AutoML training"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Update status
    job_status[job_id]["status"] = "analyzing"
    
    # Start background task
    background_tasks.add_task(run_automl_pipeline, job_id)
    
    return {"message": "Analysis started", "job_id": job_id}

async def run_automl_pipeline(job_id: str):
    """Background task to run the complete AutoML pipeline"""
    try:
        # Update status
        job_status[job_id]["status"] = "running"
        job_status[job_id]["progress"] = 0
        
        file_path = job_status[job_id]["file_path"]
        df = pd.read_csv(file_path)
        
        # Step 1: Data Cleaning
        job_status[job_id]["current_step"] = "Data Cleaning"
        job_status[job_id]["progress"] = 10
        
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_data(df)
        
        # Step 2: Problem Type Detection
        job_status[job_id]["current_step"] = "Problem Detection"
        job_status[job_id]["progress"] = 20
        
        problem_type = cleaner.detect_problem_type(cleaned_data["y"])
        job_status[job_id]["problem_type"] = problem_type
        
        # Step 3: AutoML Training
        job_status[job_id]["current_step"] = "Model Training"
        job_status[job_id]["progress"] = 30
        
        trainer = AutoMLTrainer()
        model_results = trainer.train_model(
            cleaned_data["X"], 
            cleaned_data["y"], 
            problem_type,
            job_id
        )
        
        # Step 4: Model Evaluation
        job_status[job_id]["current_step"] = "Model Evaluation"
        job_status[job_id]["progress"] = 60
        
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_model(
            model_results["model"],
            model_results["X_test"],
            model_results["y_test"],
            problem_type,
            job_id
        )
        
        # Step 5: Generate All Outputs
        job_status[job_id]["current_step"] = "Generating Outputs"
        job_status[job_id]["progress"] = 80
        
        output_generator = OutputGenerator(job_id)
        
        # Get predictions for output generation
        y_pred = model_results["model"].predict(model_results["X_test"])
        y_pred_proba = None
        if hasattr(model_results["model"], 'predict_proba'):
            y_pred_proba = model_results["model"].predict_proba(model_results["X_test"])
        
        # Define preprocessing steps
        preprocessing_steps = [
            "Missing value imputation",
            "Categorical encoding",
            "Feature scaling",
            "Data type optimization"
        ]
        
        # Generate all outputs
        all_outputs = output_generator.generate_all_outputs(
            model=model_results["model"],
            X_train=model_results["X_train"],
            X_test=model_results["X_test"],
            y_train=model_results["y_train"],
            y_test=model_results["y_test"],
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            cleaned_data=cleaned_data,
            metrics=evaluation_results["metrics"],
            problem_type=problem_type,
            preprocessing_steps=preprocessing_steps
        )
        
        # Step 6: Complete
        job_status[job_id]["current_step"] = "Complete"
        job_status[job_id]["progress"] = 100
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["results"] = {
            "model_path": model_results["model_path"],
            "metrics": evaluation_results["metrics"],
            "visualizations": evaluation_results["visualizations"],
            "problem_type": problem_type,
            "completed_at": datetime.now().isoformat(),
            "outputs": all_outputs
        }
        
    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get current status of AutoML job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get results of completed AutoML job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return job_status[job_id]["results"]

@app.get("/download/model/{job_id}")
async def download_model(job_id: str):
    """Download trained model"""
    if job_id not in job_status or job_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Model not found")
    
    outputs = job_status[job_id]["results"]["outputs"]
    model_path = outputs.get("model")
    
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(model_path, filename=f"model_{job_id}.pkl")

@app.get("/download/report/{job_id}")
async def download_report(job_id: str):
    """Download analysis report"""
    if job_id not in job_status or job_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Report not found")
    
    outputs = job_status[job_id]["results"]["outputs"]
    report_path = outputs.get("report")
    
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(report_path, filename=f"report_{job_id}.html")

@app.get("/download/cleaned-data/{job_id}")
async def download_cleaned_data(job_id: str):
    """Download cleaned dataset"""
    if job_id not in job_status or job_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Cleaned data not found")
    
    outputs = job_status[job_id]["results"]["outputs"]
    data_path = outputs.get("cleaned_data")
    
    if not data_path or not Path(data_path).exists():
        raise HTTPException(status_code=404, detail="Cleaned data file not found")
    
    return FileResponse(data_path, filename=f"cleaned_data_{job_id}.csv")

@app.get("/download/metrics/{job_id}")
async def download_metrics(job_id: str):
    """Download metrics summary"""
    if job_id not in job_status or job_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Metrics not found")
    
    outputs = job_status[job_id]["results"]["outputs"]
    metrics_path = outputs.get("metrics")
    
    if not metrics_path or not Path(metrics_path).exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    
    return FileResponse(metrics_path, filename=f"metrics_{job_id}.json")

@app.get("/download/config/{job_id}")
async def download_pipeline_config(job_id: str):
    """Download pipeline configuration"""
    if job_id not in job_status or job_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    outputs = job_status[job_id]["results"]["outputs"]
    config_path = outputs.get("pipeline_config")
    
    if not config_path or not Path(config_path).exists():
        raise HTTPException(status_code=404, detail="Configuration file not found")
    
    return FileResponse(config_path, filename=f"pipeline_config_{job_id}.json")

@app.get("/download/all/{job_id}")
async def download_all_outputs(job_id: str):
    """Download all outputs as ZIP archive"""
    if job_id not in job_status or job_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Outputs not found")
    
    outputs = job_status[job_id]["results"]["outputs"]
    zip_path = outputs.get("zip_archive")
    
    if not zip_path or not Path(zip_path).exists():
        raise HTTPException(status_code=404, detail="ZIP archive not found")
    
    return FileResponse(zip_path, filename=f"automl_outputs_{job_id}.zip")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)