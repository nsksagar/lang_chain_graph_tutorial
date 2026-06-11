from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ── State ──────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def get_stock_price(symbol: str) -> float:
    """Return the current price of a stock given the stock symbol."""
    prices = {"MSFT": 200.3, "AAPL": 100.4, "AMZN": 150.0, "RIL": 87.6}
    return prices.get(symbol, 0.0)

@tool
def buy_stocks(symbol: str, quantity: int, total_price: float) -> str:
    """Buy stocks given the stock symbol, quantity, and total price."""
    return f"Pending purchase: {quantity} {symbol} @ ${total_price:.2f}"


# ── LLM ────────────────────────────────────────────────────────────────────────

tools = [get_stock_price, buy_stocks]
llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools)


# ── Nodes ──────────────────────────────────────────────────────────────────────

def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def human_approval_node(state: State):
    # Find the last AI message that has tool calls
    last_ai = next(
        m for m in reversed(state["messages"])
        if getattr(m, "tool_calls", None)
    )
    buy_call = next(
        tc for tc in last_ai.tool_calls
        if tc["name"] == "buy_stocks"
    )

    symbol     = buy_call["args"]["symbol"]
    quantity   = buy_call["args"]["quantity"]
    total_price = buy_call["args"]["total_price"]

    decision = interrupt(f"Approve buying {quantity} {symbol} stocks for ${total_price:.2f}?")

    if str(decision).strip().lower() == "yes":
        content = f"You bought {quantity} shares of {symbol} for a total of ${total_price:.2f}."
    else:
        content = "Buying declined."

    return {
        "messages": [
            ToolMessage(content=content, tool_call_id=buy_call["id"])
        ]
    }


# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_chatbot(state: State):
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    for tc in last.tool_calls:
        if tc["name"] == "buy_stocks":
            return "human_approval"
    return "tools"


# ── Graph ──────────────────────────────────────────────────────────────────────

builder = StateGraph(State)

builder.add_node("chatbot",        chatbot_node)
builder.add_node("tools",          ToolNode([get_stock_price]))  # only non-approval tools
builder.add_node("human_approval", human_approval_node)

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", route_after_chatbot)
builder.add_edge("tools",          "chatbot")
builder.add_edge("human_approval", "chatbot")

graph = builder.compile(checkpointer=MemorySaver())


# ── Run ────────────────────────────────────────────────────────────────────────

config = {"configurable": {"thread_id": "buy_thread"}}

# Step 1: ask for price
state = graph.invoke(
    {"messages": [{"role": "user", "content": "What is the current price of 10 MSFT stocks?"}]},
    config
)
print("Bot:", state["messages"][-1].content)

# Step 2: ask to buy
state = graph.invoke(
    {"messages": [{"role": "user", "content": "Buy 10 MSFT stocks at current price."}]},
    config
)
print("Interrupt:", state.get("__interrupt__"))

# Step 3: human decision
decision = input("Approve (yes/no): ").strip()
state = graph.invoke(Command(resume=decision), config)
print("Bot:", state["messages"][-1].content)