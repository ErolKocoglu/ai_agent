import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.builder import build_agent

try:
    agent = build_agent()
    tools = agent.tools
    tool_names = [t.name for t in tools]
    print(f"Agent Tools: {tool_names}")
    
    expected_tools = ["Calculator", "Time", "wikipedia", "Python_REPL"]
    missing = [t for t in expected_tools if t not in tool_names]
    
    if missing:
        print(f"❌ Missing tools: {missing}")
        sys.exit(1)
    else:
        print("✅ All expected tools present")

    print("Testing agent invocation...")
    response = agent.invoke("Hello, who are you?")
    print(f"Agent Response: {response['output']}")
    print("✅ Agent invocation successful")

except Exception as e:
    print(f"❌ Error building/running agent: {e}")
    sys.exit(1)
