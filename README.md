# Deep Agents Travel Planner Workshop

A hands-on, single-notebook walkthrough that builds a production-style travel planner agent from scratch using [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) on top of LangChain and LangGraph, then evaluates it with LangSmith.

The notebook lives at `notebooks/deepagents-travel-planner.ipynb` and is designed to be worked through top-to-bottom in a guided session with a LangChain engineer.

## What You'll Build

By the end of the notebook you'll have a travel planning agent that can:

- Plan a multi-day trip from a single user request
- Delegate research to specialized subagents (`hotel-search`, `flight-search`, `activity-search`) running in parallel
- Read and write files on a virtual filesystem via `read_file` / `write_file` / `edit_file`
- Pause for human approval before "booking" anything
- Remember a traveler's preferences across conversations (per-user, with a shared global namespace)
- Follow a team-authored `AGENTS.md` and load on-demand `SKILL.md` files for each deliverable (itinerary, budget, packing list, travel brief)
- Emit a final itinerary and budget in your team's exact format

You'll also build a small LangSmith evaluation suite that runs the same dataset against two OpenAI models (`gpt-5.4` and `gpt-4.1-mini`), with four metrics — trajectory match, constraint check, efficiency, and LLM-as-judge response quality — so you can compare cost / latency / quality side-by-side in the LangSmith experiment comparison view.

## Notebook Outline

The notebook is split into eleven parts. Each part introduces one concept and adds one capability:

| Part | Topic | What's introduced |
| --- | --- | --- |
| 0 | Setup & Installation | `uv sync` or `pip install -r requirements.txt`, `.env`, model config |
| 1 | Your First Deep Agent | `create_deep_agent`, the harness, planner + virtual filesystem out of the box |
| 2 | Adding Custom Tools | Deterministic offline stubs for discovery (`search_travel`) and pricing/availability |
| 3 | Understanding Backends | `StateBackend` (per-thread default), preview of `StoreBackend` for cross-thread persistence |
| 4 | Adding Subagents | `hotel-search`, `flight-search`, `activity-search` + an orchestrator that fans out via `task()` |
| 5 | Middleware Deep Dive | Built-in `TodoListMiddleware`, writing a custom `@wrap_tool_call` logger |
| 6 | Human-in-the-Loop | `HumanInTheLoopMiddleware`, interrupts, `Command(resume=...)` |
| 7 | Long-Term Memory | `CompositeBackend` routing `/memories/user/` and `/memories/shared/` to a `StoreBackend` with per-user namespaces |
| 8 | `AGENTS.md` & Skills | Identity file + on-demand `SKILL.md` files for each deliverable |
| 9 | The Complete Travel Planner | All layers composed end-to-end |
| 10 | Evaluations | LangSmith dataset, four evaluators (trajectory, constraint, efficiency, LLM-judge), two-model comparison |
| 11 | Next Steps | Where to take the agent next (real APIs, calendar, multi-leg trips, group travel) |

Each part is independently runnable — you can stop at any layer and have a working agent for that scope.

## Pre-work

### Clone the repo
```bash
git clone <repo-url>
cd travel-planner-deepagents-workshop
```

### Configure environment variables
```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY (or your provider's keys) + LANGSMITH_API_KEY for Part 10 evals
```

No Tavily key needed — discovery is handled by a deterministic offline stub so the workshop runs without external APIs.

If corporate policy blocks API keys, contact your LangChain representative and we'll find a workaround.

### Install dependencies

Two options — both install the exact same pinned versions (`requirements.txt` is generated from `uv.lock` and is pip-tested on Python 3.11).

**Option A: `uv`** (recommended if you have it)
```bash
pip install uv
uv sync
source .venv/bin/activate
```

**Option B: `pip`**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Versions are pinned to a known-good combo (`deepagents==0.5.2`, `langgraph==1.1.6`, `langchain-core==1.2.28`) to avoid the `ToolRuntime.__init__() missing 'tools'` skew that newer `langgraph` releases trigger with `deepagents` 0.5.x.

## Running

### As a notebook

Open `notebooks/deepagents-travel-planner.ipynb` in JupyterLab, VS Code, or Cursor and select the workshop's `.venv` as the kernel.

If your editor doesn't auto-detect the venv, register it as a named Jupyter kernel:

```bash
.venv/bin/python -m ipykernel install --user --name=travel-planner --display-name="Travel Planner"
```

Then pick **Travel Planner** from the kernel selector.

### As a LangGraph Studio agent

The agent in `agents/travel_planner/` is registered in `langgraph.json`. Launch it locally with:

```bash
source .venv/bin/activate
langgraph dev
```

This starts an API server at `http://localhost:2024` and prints a LangGraph Studio URL. Studio gives you a visual graph view, message-level tracing, checkpoint history, and a chat panel.

## Model Configuration

Model selection is centralized in `utils/models.py`. The default is OpenAI (`openai:gpt-5.4`); the file contains commented-out blocks for Azure OpenAI, AWS Bedrock, and Google Vertex AI — uncomment the section matching your provider and fill in the relevant env vars in `.env`.

The notebook's Part 10 evaluations call `init_chat_model()` directly to swap models for the comparison (`openai:gpt-5.4` vs `openai:gpt-4.1-mini`) and use `model` from `utils/models.py` for the LLM-as-judge. If your provider proxies OpenAI through APIM, point `OPENAI_BASE_URL` at the proxy so all three model names resolve through it.

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

## Repo Layout

```
travel-planner-deepagents-workshop/
├── notebooks/
│   └── deepagents-travel-planner.ipynb   # the workshop notebook
├── agents/
│   └── travel_planner/                   # standalone agent for `langgraph dev`
│       ├── agent.py                      # mirrors the notebook's final composition
│       ├── AGENTS.md                     # agent identity & workflow rules
│       └── skills/                       # on-demand output formats
│           ├── itinerary-format/SKILL.md
│           ├── budget-summary/SKILL.md
│           ├── packing-list/SKILL.md
│           └── travel-brief/SKILL.md
├── utils/
│   └── models.py                         # centralized model selection
├── langgraph.json                        # registers the travel_planner agent
├── pyproject.toml
├── requirements.txt                      # pinned, pip-tested
└── .env.example
```

## Resources

- **[Deep Agents Documentation](https://docs.langchain.com/oss/python/deepagents/)** — harness reference, backends, middleware
- **[LangChain Documentation](https://docs.langchain.com/oss/python/langchain/overview)** — `create_agent`, tools, middleware
- **[LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)** — state, interrupts, `Store`, checkpointers
- **[LangChain vs LangGraph vs Deep Agents](https://docs.langchain.com/oss/python/concepts/products)** — how the layers relate
- **[LangSmith](https://smith.langchain.com)** — tracing, datasets, evaluations, the experiment comparison view used in Part 10
