from pathlib import Path

from langgraph.prebuilt import create_react_agent 
from langgraph.graph import StateGraph, MessagesState, START, END

from .handoff import (
    assign_to_government_advisor_agent, 
    assign_to_hr_advisor_agent, 
    assign_to_staff_representative_agent
)
from ..rag import search_database


BASE_DIR = Path(__file__).resolve().parent


PROMPTS = {
    "government_advisor_agent": open(BASE_DIR / "government_advisor.prompt", encoding="UTF-8").read(),
    "hr_advisor_agent": open(BASE_DIR / "hr_advisor.prompt", encoding="UTF-8").read(),
    "staff_representative_agent": open(BASE_DIR / "staff_representative.prompt", encoding="UTF-8").read(),
    "director_agent": open(BASE_DIR / "director.prompt", encoding="UTF-8").read()
}


government_advisor_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[search_database],
    prompt=PROMPTS["government_advisor_agent"],
    name="government_advisor_agent",
)


hr_advisor_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[search_database],
    prompt=PROMPTS["hr_advisor_agent"],
    name="hr_advisor_agent",
)


staff_representative_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[search_database],
    prompt=PROMPTS["staff_representative_agent"],
    name="staff_representative_agent",
)


director_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[assign_to_hr_advisor_agent, assign_to_staff_representative_agent, assign_to_government_advisor_agent],
    prompt=PROMPTS["director_agent"],
    name="director_agent",
)

debate = (
    StateGraph(MessagesState)
    # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
    .add_node(director_agent, destinations=("hr_advisor_agent", "staff_representative_agent", "government_advisor_agent", END))
    .add_node(hr_advisor_agent)
    .add_node(staff_representative_agent)
    .add_node(government_advisor_agent)
    .add_edge(START, "director_agent")
    # always return back to the supervisor
    .add_edge("hr_advisor_agent", "director_agent")
    .add_edge("staff_representative_agent", "director_agent")
    .add_edge("government_advisor_agent", "director_agent")
    .compile()
)