import sys
from src.agent.builder import build_agent

def main():
    agent = build_agent()
    
    print("🤖 Agent initialized. Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            query = input("\n💬 User Query: ")
            if query.lower() in {"exit", "quit"}:
                print("👋 Exiting...")
                break
            
            response = agent.invoke(query)
            print("\n--- Final Answer ---")
            print(response['output'])
            
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
