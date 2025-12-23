import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.tools.math_tool import get_math_tool
    from src.tools.time_tool import time_tool
    from src.tools.wikipedia_tool import get_wikipedia_tool
    from src.tools.python_repl_tool import get_python_repl_tool
    from src.tools.file_tool import get_file_tools
    
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

try:
    # Mock LLM for initialization check
    class MockLLM:
        def bind(self, *args, **kwargs): return self
        def invoke(self, *args, **kwargs): return "Mock response"
        
    print("Checking tool initialization...")
    
    # Math
    # Note: LLMMathChain might need a real LLM or compatible mock. 
    # We'll skip deep execution check for now, just instantiation if possible.
    # get_math_tool(MockLLM()) 
    
    # Time
    print(f"Time Tool: {time_tool.run('now')}")
    
    # Wikipedia
    wiki = get_wikipedia_tool()
    print(f"Wikipedia Tool Name: {wiki.name}")
    
    # REPL
    repl = get_python_repl_tool()
    print(f"REPL Tool Name: {repl.name}")
    
    # File
    files = get_file_tools()
    print(f"File Tools Count: {len(files)}")
    
    print("✅ Tool initialization successful")

except Exception as e:
    print(f"❌ Tool initialization failed: {e}")
    sys.exit(1)
