# Setup

1. Download and install uv from https://docs.astral.sh/uv/getting-started/installation/.

2. Download and install Ollama from https://ollama.com/download.

3. Download embedding model from Ollama by running:

```bash
ollama pull mxbai-embed-large
```

3. Make a copy of `.env.example`, rename to `.env`, set the OpenAI API key.

4. Setup environment by running

```bash
`uv sync`
```

## How It Works

We have 4 AI agents:
1. Director
2. HR Advisor
3. Staff Representative
4. Government Advisor

Each agent is powered by a large language model (LLM). You can view their prompts in `./agents/*.prompt`.

- Input: You provide a description of the company’s situation.
- Debate: The Director hosts a 2-round debate on how to improve employee retention.
    - In each round, the HR Advisor, Staff Representative, and Government Advisor each get a turn to speak.
    - Agents can use the database to find supporting facts, statistics, or case studies.
- Summary: After 2 rounds, the Director summarizes the discussion and produces a final recommendation.

This is the minimal working example (see `main.py` for more info):

```python
import os
import getpass

from io import BytesIO

from dotenv import load_dotenv


# Load environment variables
def init_env_var(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


load_dotenv()
init_env_var("OPENAI_API_KEY")

# Import agents
from agents import debate
from messages import pretty_print_messages

# Start the agent debate process
for chunk in debate.stream(

    # Input: company situation, can be modified to support model prediction results
    {
        "messages": [
            {
                "role": "user",
                "content": "A Singapore-based fintech startup noticed high attrition among junior software engineers, many leaving within 18 months for larger firms. HR analytics revealed that limited career progression and below-market compensation were key factors driving turnover, compounded by intense project deadlines.",
            }
        ]
    },
):

    # Print agent logs to console
    pretty_print_messages(chunk, last_message=True)
```

## Structure of messages

You can check an example of message log in `messages.pkl`, can be read by running:

```python
uv run display.py
```