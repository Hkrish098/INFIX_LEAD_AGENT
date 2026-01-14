from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # 'add_messages' ensures chat history is appended, not overwritten
    messages: Annotated[list, add_messages]
    intent: str
    # Specific fields for the AutoStream lead capture tool
    lead_data: dict # Keys: name, email, platform