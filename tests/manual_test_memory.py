import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.builder import build_agent

def test_memory():
    print("Building agent...")
    agent = build_agent()
    print("DEBUG: Agent prompt input variables:", agent.agent.llm_chain.prompt.input_variables)
    # print("DEBUG: Agent prompt template:", agent.agent.llm_chain.prompt.template)
    
    print("Step 1: Introducing myself.")
    response1 = agent.invoke({"input": "My name is erol."})
    print(f"Agent response 1: {response1['output']}")
    
    print("Step 2: Asking for my name.")
    response2 = agent.invoke({"input": "What is my name?"})
    print(f"Agent response 2: {response2['output']}")
    
    if "erol" in response2['output'].lower():
        print("SUCCESS: Agent remembered my name.")
    else:
        print("FAILURE: Agent did not remember my name.")

if __name__ == "__main__":
    test_memory()
