import logging 
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from src.config import GROQ_API_KEY, LLM_MODEL
from langsmith import traceable


logger = logging.getLogger(__name__)

SYSTEM_PROMPT_DIRECT = """You are an expert academic assistant.
Answer in a clear, structured, and precise way in French.

You can use your general knowledge to answer.
You can refer to previous messages in the conversation normally.
If you are not sure about something, say it clearly.
"""

@traceable(run_type="chain", name="Direct LLM Query")
def ask_direct(question: str, history: list = None, thread_id: str = None, conversation_id: str = None, langsmith_extra: dict = None) -> dict:
    if history is None:
        history = []

    messages = [SystemMessage(content=SYSTEM_PROMPT_DIRECT)]

    for msg in history[-10:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.7,
    )

    try:
        response = llm.invoke(messages)
        logger.info(f"Direct mode: response generated for '{question[:60]}'")
        return {
            "answer": response.content,
            "sources": [],
            "mode": "direct"
        }

    except Exception as e:
        logger.error(f"Erreur LLM direct : {e}")
        return {
            "answer": f"❌ Error: {str(e)}",
            "sources": [],
            "mode": "direct"
        }
