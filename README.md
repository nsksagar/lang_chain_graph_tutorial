# LangChain Graph Tutorial

A hands-on tutorial series for building LLM-powered graphs with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). Each notebook/script builds on the previous one, progressing from a simple stateful graph to a full Human-in-the-Loop agent.

## Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Setup

```bash
# Clone the repo
git clone https://github.com/sagarnookarapu/lang_chain_graph_tutorial.git
cd lang_chain_graph_tutorial

# Create virtual environment and install dependencies
uv sync
# or: pip install -e .
```

Create a `.env` file in the project root with your API keys:

```env
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key   # only needed for notebook 7
LANGCHAIN_TRACING_V2=true                  # only needed for notebook 7
```

## Tutorials

### 1. `simple_graph.ipynb` — Simple StateGraph

Introduces `StateGraph` with a `TypedDict` state. Builds a two-node pipeline that calculates a USD investment total (with 8% return) and converts it to INR.

**Concepts:** `StateGraph`, `add_node`, `add_edge`, `START`/`END`, `graph.invoke`, Mermaid graph visualization.

### 2. `chatbot.ipynb` — Basic Chatbot

Wraps a Groq LLM (`llama-3.1-8b-instant`) in a single-node graph to create a conversational chatbot. Includes both single-turn and multi-turn (loop) interaction.

**Concepts:** `add_messages` annotation, message state, multi-turn conversation loop.

### 3. `2_graph_with_condition.ipynb` — Conditional Edges

Extends the portfolio graph to support conditional routing: after computing the USD total, the graph routes to either an INR or EUR conversion node based on a `target_currency` field.

**Concepts:** `add_conditional_edges`, routing functions, branching graphs.

### 4. `4_tool_call.ipynb` — Tool Calling

Adds a `get_stock_price` tool to the LLM. The graph can now call external tools and route back to the chatbot based on whether a tool call was made.

**Concepts:** `@tool` decorator, `bind_tools`, `ToolNode`, `tools_condition`.

### 5. `5_tool_call_agent.ipynb` — Agentic Tool Loop

Adds a loop edge from the tools node back to the chatbot, turning the graph into a true agent that can make multiple sequential tool calls in a single user query.

**Concepts:** Agentic loops, multi-step reasoning, tool call chaining.

### 6. `6_memory.ipynb` — Persistent Memory

Adds `MemorySaver` as a checkpointer so the graph can maintain conversation history across multiple turns, scoped by `thread_id`.

**Concepts:** `MemorySaver`, `checkpointer`, `thread_id` config, multi-session memory isolation.

### 7. `7_langsmith.ipynb` — LangSmith Tracing

Wraps the graph invocation with `@traceable` to send traces to [LangSmith](https://smith.langchain.com/) for observability and debugging.

**Concepts:** `@traceable`, LangSmith integration, `LANGCHAIN_TRACING_V2`.

### 8. `8_HITL.py` — Human-in-the-Loop (HITL)

A Python script implementing a stock-buying agent that requires human approval before executing a purchase. Uses `interrupt()` to pause execution and `Command(resume=...)` to continue after the human decision.

**Concepts:** `interrupt`, `Command`, human approval node, HITL pattern, `MemorySaver` for resumable state.

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Graph orchestration framework |
| `langchain` | LLM abstractions and tools |
| `langchain-groq` | Groq LLM provider |
| `langsmith` | Tracing and observability |
| `python-dotenv` | `.env` file loading |
| `graphviz` | Graph visualization support |

## Project Structure

```
lang_chain_graph_tutorial/
├── simple_graph.ipynb          # 1. Simple StateGraph
├── chatbot.ipynb               # 2. Basic chatbot
├── 2_graph_with_condition.ipynb # 3. Conditional routing
├── 4_tool_call.ipynb           # 4. Tool calling
├── 5_tool_call_agent.ipynb     # 5. Agentic tool loop
├── 6_memory.ipynb              # 6. Persistent memory
├── 7_langsmith.ipynb           # 7. LangSmith tracing
├── 8_HITL.py                   # 8. Human-in-the-Loop
├── pyproject.toml
└── .env                        # Not committed — add your API keys here
```