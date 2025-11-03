import os
import pickle
import getpass

from io import BytesIO

from dotenv import load_dotenv
from PIL import Image


def init_env_var(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


load_dotenv()
init_env_var("OPENAI_API_KEY")


from agents import debate
from messages import pretty_print_messages

graph_bytes = debate.get_graph().draw_mermaid_png()
image = Image.open(BytesIO(graph_bytes))
image.save("graph.png")

for chunk in debate.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "A Singapore-based fintech startup noticed high attrition among junior software engineers, many leaving within 18 months for larger firms. HR analytics revealed that limited career progression and below-market compensation were key factors driving turnover, compounded by intense project deadlines.",
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)


final_message_history = chunk["director_agent"]["messages"]

with open("messages.pkl", "wb") as f:
    pickle.dump(final_message_history, f)

def print_all_message_contents(messages):
    """Print all message contents in order, with sender name if available."""
    for i, msg in enumerate(messages, start=1):
        sender = getattr(msg, "name", None)
        msg_type = type(msg).__name__
        content = getattr(msg, "content", "")

        print(f"\n[{i}] {msg_type}{' (' + sender + ')' if sender else ''}:")
        print(content or "(empty)")