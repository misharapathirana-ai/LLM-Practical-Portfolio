"""Lab 06 Part B - agent with SQLite memory."""
import sqlite3
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from agent_tools import calculator

DB_PATH = "memory.db"


def run_sql(sql, params=()):
    db = sqlite3.connect(DB_PATH)
    try:
        rows = db.execute(sql, params).fetchall()
        db.commit()
        return rows
    finally:
        db.close()


def init_db():
    run_sql("CREATE TABLE IF NOT EXISTS memory "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, text TEXT)")


def save_memory(text):
    run_sql("INSERT INTO memory (created, text) VALUES (?, ?)",
            (datetime.now().isoformat(timespec="seconds"), text))


@tool
def search_memory(keyword: str) -> str:
    """Search what the user told you earlier. Use one or two keywords, e.g. 'name' or 'birthday'."""
    words = [w for w in keyword.lower().split() if len(w) > 1]
    if not words:
        return "No keyword given."
    conditions = " OR ".join("lower(text) LIKE ?" for _ in words)
    rows = run_sql(
        f"SELECT created, text FROM memory WHERE {conditions} ORDER BY id DESC LIMIT 5",
        tuple(f"%{w}%" for w in words),
    )
    if not rows:
        return "Nothing found in memory."
    return "\n".join(f"[{created}] {text}" for created, text in rows)


def build_agent(tools, system_prompt, max_steps=5):
    llm = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tools)
    by_name = {t.name: t for t in tools}

    def ask(question):
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        retried = False
        for _ in range(max_steps):
            reply = llm.invoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                if reply.content.lstrip().startswith(('{"type"', '{"name"')) and not retried:
                    retried = True
                    messages.append(HumanMessage(content="Call the tool properly instead of printing JSON."))
                    continue
                return reply.content
            for call in reply.tool_calls:
                print(f"  [tool] {call['name']}({call['args']})")
                try:
                    result = by_name[call["name"]].invoke(call["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return "Stopped after too many steps."

    return ask


def chat_loop(ask):
    print("Agent ready. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        print("Agent:", ask(q))
        if not q.endswith("?"):      # questions are not facts, so they are not stored
            save_memory(q)


SYSTEM_PROMPT = (
    "You are a helpful assistant with a memory. If the user asks about something they told you "
    "earlier (their name, likes, plans), call search_memory first and answer from the result. "
    "Never guess personal details. Use the calculator for maths. Keep answers short."
)

if __name__ == "__main__":
    init_db()
    chat_loop(build_agent([calculator, search_memory], SYSTEM_PROMPT))
