import requests
from langchain.agents import Tool

def get_joke(_: str) -> str:
    """Returns a random joke from an API"""
    try:
        response = requests.get("https://icanhazdadjoke.com/", headers={"Accept": "application/json"})
        return response.json().get("joke", "No joke found.")
    except Exception as e:
        return f"API error: {e}"

joke_tool = Tool(
    name="get_joke",
    func=get_joke,
    description="Use this tool to get a random joke.",
)
