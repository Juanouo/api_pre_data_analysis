
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Dict, Any, Optional
import os
import io

def get_robust_correlation(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate correlation matrix with robust handling of missing values.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        Dictionary with correlation data or empty dict if not possible
    """
    # Get only numeric columns
    numeric_data = data.select_dtypes(include='number')
    
    if numeric_data.shape[1] < 2:
        # Need at least 2 numeric columns for correlation
        return {}
    
    try:
        # First, try standard correlation
        corr_matrix = numeric_data.corr(method='pearson', min_periods=1)
        
        # Check if we got a valid correlation matrix
        if corr_matrix.empty or corr_matrix.isna().all().all():
            # If standard correlation fails, try with dropna
            cleaned_numeric = numeric_data.dropna()
            if len(cleaned_numeric) > 1 and cleaned_numeric.shape[1] > 1:
                corr_matrix = cleaned_numeric.corr(method='pearson')
            else:
                return {}
        
        # Remove any remaining NaN values by replacing with 0 for non-diagonal
        # and 1 for diagonal (self-correlation)
        corr_filled = corr_matrix.fillna(0)
        for col in corr_filled.columns:
            corr_filled.loc[col, col] = 1.0
            
        return corr_filled.to_dict()
        
    except Exception:
        # If all else fails, return empty dict
        return {}

def get_best_sample_rows(data: pd.DataFrame, n_samples: int = 3) -> pd.DataFrame:
    """
    Get the best sample rows from DataFrame, prioritizing complete rows without nulls,
    otherwise selecting rows with fewest missing values.
    
    Args:
        data: pandas DataFrame to sample from
        n_samples: number of sample rows to return
    
    Returns:
        DataFrame with the best sample rows
    """
    # Ensure we don't ask for more samples than available rows
    n_samples = min(n_samples, len(data))
    
    if n_samples == 0:
        return data.iloc[:0]  # Return empty DataFrame with same structure
    
    # Count missing values per row
    missing_per_row = data.isna().sum(axis=1)
    
    # Get rows with no missing values (complete rows)
    complete_rows = data[missing_per_row == 0]
    
    if len(complete_rows) >= n_samples:
        # We have enough complete rows, sample from them
        if len(complete_rows) == n_samples:
            return complete_rows
        else:
            return complete_rows.sample(n=n_samples, random_state=42)
    else:
        # Not enough complete rows, get all complete rows first
        sample_rows = complete_rows.copy()
        remaining_needed = n_samples - len(complete_rows)
        
        if remaining_needed > 0:
            # Get remaining rows from those with fewest missing values
            incomplete_rows = data[missing_per_row > 0]
            if len(incomplete_rows) > 0:
                # Sort by number of missing values (ascending) and take the best ones
                sorted_incomplete = incomplete_rows.loc[missing_per_row[missing_per_row > 0].sort_values().index]
                additional_rows = sorted_incomplete.head(remaining_needed)
                sample_rows = pd.concat([sample_rows, additional_rows], ignore_index=True)
        
        return sample_rows

# Initialize FastAPI app
app = FastAPI(title="Pre Data Analysis API", description="API for pre-analyzing datasets as to give more context to LLMs")

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
        analysis_data = {
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
            "sample": convert_dict(get_best_sample_rows(data, 3).to_dict()),
            "correlation": convert_dict(get_robust_correlation(data)),
            "unique_values": convert_dict(unique_values)
        }
        
        return {"data": analysis_data}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
