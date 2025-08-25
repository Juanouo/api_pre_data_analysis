import kagglehub
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Dict, Any, Optional
import os
import io

# Initialize FastAPI app
app = FastAPI(title="Airlines Data Analysis API", description="API for analyzing airlines flights data")

# Set pandas options
pd.set_option('display.max_columns', 100)

# Global variable to store the dataframe
df = None

def load_data_from_file(file_content: bytes, file_format: str) -> pd.DataFrame:
    """Load data from binary file content based on format"""
    try:
        # Create a BytesIO object from the binary content
        file_buffer = io.BytesIO(file_content)
        
        # Load data based on format
        if file_format.lower() == 'csv':
            df = pd.read_csv(file_buffer)
        elif file_format.lower() == 'parquet':
            df = pd.read_parquet(file_buffer)
        elif file_format.lower() in ['xls', 'xlsx', 'excel']:
            df = pd.read_excel(file_buffer)
        elif file_format.lower() == 'json':
            df = pd.read_json(file_buffer)
        elif file_format.lower() == 'feather':
            df = pd.read_feather(file_buffer)
        elif file_format.lower() == 'pickle':
            df = pd.read_pickle(file_buffer)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Validate that it's a proper DataFrame
        if df.empty:
            raise ValueError("The uploaded file resulted in an empty DataFrame")
        
        if len(df.columns) == 0:
            raise ValueError("The uploaded file has no columns")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame is a proper table"""
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if len(df.columns) == 0:
        return False
    
    # Check if DataFrame has reasonable size (not too large)
    if len(df) > 1_000_000:  # More than 1M rows
        raise ValueError("DataFrame is too large (max 1M rows allowed)")
    
    if len(df.columns) > 1000:  # More than 1000 columns
        raise ValueError("DataFrame has too many columns (max 1000 allowed)")
    
    return True

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Data Analysis API", 
        "description": "Upload tabular data files for comprehensive analysis",
        "endpoints": {
            "/analyze": "POST - Upload a file for analysis",
            "/health": "GET - Health check"
        },
        "supported_formats": ["csv", "parquet", "excel", "xls", "xlsx", "json", "feather", "pickle"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_data(
    file: UploadFile = File(..., description="The data file to analyze"),
    format: str = Form(..., description="File format (csv, parquet, excel, json, etc.)")
) -> Dict[str, Any]:
    """Analyze uploaded tabular data and return comprehensive statistics"""
    
    try:
        # Read the file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Load data from file
        try:
            data = load_data_from_file(file_content, format)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Validate the DataFrame
        try:
            validate_dataframe(data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Calculate unique values for columns with less than 10 unique values
        unique_values = {k: data[k].unique().tolist() for k in data.nunique()[data.nunique().lt(10)].index}
        
        # Convert numpy/pandas types to JSON serializable types
        def convert_to_serializable(obj):
            """Convert numpy/pandas objects to JSON serializable types"""
            if hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            elif str(type(obj)).startswith("<class 'numpy"):  # numpy types
                return str(obj)
            elif str(type(obj)).startswith("<class 'pandas"):  # pandas types
                return str(obj)
            else:
                return obj
        
        # Helper function to recursively convert dictionaries
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(item) for item in d]
            else:
                return convert_to_serializable(d)
        
        # Prepare result with proper type conversion
        result = {
            "file_info": {
                "filename": file.filename,
                "format": format,
                "size_bytes": len(file_content)
            },
            "shape": {'rows': data.shape[0], 'columns': data.shape[1]},
            "dtypes": {k: str(v) for k, v in data.dtypes.to_dict().items()},
            "missing": convert_dict(data.isna().sum().to_dict()),
            "nunique": convert_dict(data.nunique().to_dict()),
            "describe": convert_dict(data.describe(include="all").to_dict()),
            "sample": convert_dict(data.sample(min(3, len(data))).to_dict()),
            "correlation": convert_dict(data.corr(numeric_only=True).to_dict()) if data.select_dtypes(include='number').shape[1] > 1 else {},
            "unique_values": convert_dict(unique_values)
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
