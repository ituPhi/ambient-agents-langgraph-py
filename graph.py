import operator

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

_ = load_dotenv(override=True)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


@tool
def write_request_response(receiver: str, sender: str, content: str) -> str:
    """
    Write a request response message.
    receiver: who the message is sent to
    sender: who the message is sent from
    content: the content of the message
    """
    return f"From {sender} to {receiver}: {content}"


llm = init_chat_model("openai:gpt-5-nano")
llm_with_tools = llm.bind_tools([write_request_response])


def call_model(state: MessagesState) -> dict[str, list[AnyMessage] | int]:
    messages = [
        SystemMessage(
            content=" You are a helpful assistant, if the user makes a direct request,write a brief response to the users request using the write_request_response tool then show the user the response"
        )
    ] + state.get("messages")
    response = llm_with_tools.invoke(messages)

    # print(len(response.tool_calls))
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls") + 1,
    }


graph = StateGraph(MessagesState)
graph.add_node("call_model", call_model)
graph.add_node("tools", ToolNode([write_request_response]))
graph.add_edge(START, "call_model")
graph.add_conditional_edges(
    "call_model", tools_condition, {"tools": "tools", "__end__": "__end__"}
)
graph.add_edge("tools", "call_model")
graph.add_edge("call_model", END)

agent = graph.compile()


# response = agent.invoke(
#     {
#         "messages": [HumanMessage(content="The form on the website is broken")],
#         "llm_calls": 0,
#     }
# )


# messages = response["messages"]
# for msg in messages:
#     print(msg.pretty_print())
