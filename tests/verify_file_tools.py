import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.pdf_tool import get_pdf_tool
from src.tools.data_tool import get_data_tools
import pandas as pd
from pypdf import PdfWriter

def create_dummy_files():
    print("Creating dummy files...")
    # Create CSV
    df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [30, 25]})
    df.to_csv('test_data.csv', index=False)
    print("Created test_data.csv")
    
    # Create PDF
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open("test_doc.pdf", "wb") as f:
        writer.write(f)
    print("Created test_doc.pdf (blank but valid)")

def test_tools():
    print("\nTesting Data Tools...")
    data_tools = get_data_tools()
    read_tool = data_tools[0]
    write_tool = data_tools[1]
    
    # Test Read CSV
    print("Reading CSV:")
    print(read_tool.invoke("test_data.csv"))
    
    # Test Write JSON
    print("\nWriting JSON:")
    json_data = '[{"name": "Charlie", "age": 35}]'
    print(write_tool.invoke({"file_path": "test_output.json", "data_json": json_data}))
    
    # Verify JSON
    print("Verifying JSON output:")
    print(read_tool.invoke("test_output.json"))

    print("\nTesting PDF Tool...")
    # Note: The blank PDF won't have text, but it shouldn't crash
    pdf_tool = get_pdf_tool()
    print("Reading PDF:")
    print(pdf_tool.invoke("test_doc.pdf"))

if __name__ == "__main__":
    try:
        create_dummy_files()
        test_tools()
        print("\nCleaning up...")
        if os.path.exists("test_data.csv"): os.remove("test_data.csv")
        if os.path.exists("test_doc.pdf"): os.remove("test_doc.pdf")
        if os.path.exists("test_output.json"): os.remove("test_output.json")
        print("Done.")
    except Exception as e:
        print(f"\nFailed: {e}")
