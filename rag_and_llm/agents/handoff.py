from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )

    return handoff_tool


assign_to_government_advisor_agent = create_handoff_tool(
    agent_name="government_advisor_agent",
    description="Let government advisor agent speak.", 
)

assign_to_hr_advisor_agent = create_handoff_tool(
    agent_name="hr_advisor_agent",
    description="Let HR advisor agent speak.", 
)

assign_to_staff_representative_agent = create_handoff_tool(
    agent_name="staff_representative_agent",
    description="Let staff representative agent speak.", 
)