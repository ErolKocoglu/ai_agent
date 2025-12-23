from langchain.tools import tool
import pandas as pd
import os
import json

@tool
def read_data_file(file_path: str) -> str:
    """
    Reads a structured data file (CSV, JSON, Excel) and returns a summary and sample of the data.
    Useful for understanding the structure and content of data files.
    
    Args:
        file_path (str): Absolute path to the file. Supported extensions: .csv, .json, .xlsx, .xls
        
    Returns:
        str: A markdown string containing the DataFrame info and first 5 rows.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            return f"Error: Unsupported file extension {ext}. Supported: .csv, .json, .xlsx, .xls"
            
        summary = "### Data Summary\n"
        summary += f"Shape: {df.shape}\n\n"
        summary += "### Columns\n"
        summary += ", ".join(df.columns.astype(str)) + "\n\n"
        summary += "### First 5 Rows\n"
        summary += df.head().to_markdown(index=False)
        
        return summary
    except Exception as e:
        return f"Error reading data file: {str(e)}"

@tool
def write_data_file(file_path: str, data_json: str) -> str:
    """
    Writes data to a file (CSV or JSON).
    
    Args:
        file_path (str): Absolute path where the file should be saved.
        data_json (str): A JSON string representing the data (list of dictionaries).
        
    Returns:
        str: Status message.
    """
    try:
        data = json.loads(data_json)
        df = pd.DataFrame(data)
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            df.to_csv(file_path, index=False)
        elif ext == '.json':
            df.to_json(file_path, orient='records', indent=2)
        else:
            return f"Error: Unsupported output format {ext}. Supported: .csv, .json"
            
        return f"Successfully wrote {len(df)} rows to {file_path}"
    except Exception as e:
        return f"Error writing data file: {str(e)}"

def get_data_tools():
    return [read_data_file, write_data_file]
