# Deep Agents Travel Planner Workshop

A hands-on, single-notebook walkthrough that builds a production-style travel planner agent from scratch using [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) on top of LangChain and LangGraph.

The notebook lives at `notebooks/deepagents-travel-planner.ipynb` and is designed to be worked through top-to-bottom in a guided session with a LangChain engineer.

## What You'll Build

By the end of the notebook you'll have a travel planning agent that can:

- Plan a multi-day trip from a single user request
- Delegate research to specialized subagents (`hotel-search`, `flight-search`, `activity-search`) running in parallel
- Read and write files on a virtual filesystem (or on real disk) via `read_file` / `write_file` / `edit_file`
- Pause for human approval before "booking" anything
- Remember a traveler's preferences across conversations (per-user, with a shared global namespace)
- Follow a team-authored `AGENTS.md` and load on-demand `SKILL.md` files for each deliverable (itinerary, budget, packing list, travel brief)
- Emit a final itinerary and budget in your team's exact format

## Notebook Outline

The notebook is split into ten parts. Each part introduces one Deep Agents concept and adds one capability to the agent:

| Part | Topic | What's introduced |
| --- | --- | --- |
| 0 | Setup & Installation | `uv sync`, `.env`, model config |
| 1 | Your First Deep Agent | `create_deep_agent`, the harness, planner + virtual filesystem out of the box |
| 2 | Adding Custom Tools | Domain tools for discovery, pricing, and availability |
| 3 | Understanding Backends | `StateBackend`, `FilesystemBackend`, `CompositeBackend`, `StoreBackend` |
| 4 | Adding Subagents | `hotel-search`, `flight-search`, `activity-search` + an orchestrator that fans out via `task()` |
| 5 | Middleware Deep Dive | Inspecting `write_todos`, writing a `@wrap_tool_call` logger |
| 6 | Human-in-the-Loop | `HumanInTheLoopMiddleware`, interrupts, `Command(resume=...)` |
| 7 | Long-Term Memory | `StoreBackend` routed via `CompositeBackend`, per-user vs shared namespaces |
| 8 | `AGENTS.md` & Skills | Identity file + on-demand `SKILL.md` files for each deliverable |
| 9 | The Complete Travel Planner | All layers composed end-to-end |

Each part is independently runnable — you can stop at any layer and have a working agent for that scope.

## Pre-work

### Clone the repo
```bash
git clone <repo-url>
cd deepagents-workshop
```

### Configure environment variables
```bash
cp .env.example .env
# Fill in your model + Tavily keys
```

If corporate policy blocks API keys, contact your LangChain representative and we'll find a workaround.

### Install dependencies
This project uses [`uv`](https://github.com/astral-sh/uv):
```bash
# Install uv if you haven't already
pip install uv

# Install the project
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Running the Notebook

Open `notebooks/deepagents-travel-planner.ipynb` in Jupyter, PyCharm, or VS Code, select the workshop's `.venv` as the kernel, and run cells in order.

## Model Configuration

Model selection is centralized in `utils/models.py`. The default is OpenAI; the file contains commented-out blocks for Azure OpenAI, AWS Bedrock, and Google Vertex AI — uncomment the section matching your provider and fill in the relevant env vars in `.env`.

### Azure OpenAI
```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=2024-03-01-preview
```

### AWS Bedrock
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION_NAME=us-east-1
AWS_MODEL_ARN=...
```

### Google Vertex AI
```
GOOGLE_APPLICATION_CREDENTIALS=./vertexCred.json
```
Make sure `vertexCred.json` is in `.gitignore`.

## Resources

- **[Deep Agents Documentation](https://docs.langchain.com/oss/python/deepagents/)** — harness reference, backends, middleware
- **[LangChain Documentation](https://docs.langchain.com/oss/python/langchain/overview)** — `create_agent`, tools, middleware
- **[LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)** — state, interrupts, `Store`, checkpointers
- **[LangChain vs LangGraph vs Deep Agents](https://docs.langchain.com/oss/python/concepts/products)** — how the layers relate
- **[LangSmith](https://smith.langchain.com)** — tracing and debugging the agent's tool calls and subagent fan-outs
