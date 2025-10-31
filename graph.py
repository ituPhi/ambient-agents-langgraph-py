from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import operator

_ = load_dotenv(override=True)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


llm = init_chat_model("openai:gpt-5-nano")


def ingest_request(state: MessagesState) -> dict[str, list[AnyMessage] | int]:
    """LLM node that calls the model and returns messages"""
    messages: list[AnyMessage] = [
        SystemMessage(content="Say hello in Spanish")
    ] + state["messages"]

    response: AnyMessage = llm.invoke(messages)

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


agent_builder = StateGraph(MessagesState)
_ = agent_builder.add_node("llm_call", ingest_request)
_ = agent_builder.add_edge(START, "llm_call")
_ = agent_builder.add_edge("llm_call", END)

agent = agent_builder.compile()


result = agent.invoke({"messages": [HumanMessage(content="Hola")], "llm_calls": 0})

for m in result["messages"]:
    m.pretty_print()
