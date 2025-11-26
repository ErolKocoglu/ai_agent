from datetime import datetime
from langchain.agents import Tool

def get_current_time(_: str) -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

time_tool = Tool(
    name="Time",
    func=get_current_time,
    description="Useful for when you need to know the current time or date."
)
