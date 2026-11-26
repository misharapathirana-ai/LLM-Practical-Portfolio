"""Lab 06 - tool-using agent with Ollama (Part A + multi-tool exercise)."""
import ctypes
import os
import platform
import re
import subprocess
import time
from datetime import datetime

import pyautogui
import pygetwindow as gw
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

pyautogui.PAUSE = 0.1   # move the mouse to a screen corner to abort a runaway automation


def type_into(exe, title, text, interval=0.1):
    subprocess.Popen(exe)
    time.sleep(2)                       # give the app time to open
    try:
        gw.getWindowsWithTitle(title)[0].activate()
    except Exception:
        pass                            # if focusing fails we still try typing
    pyautogui.write(text, interval=interval)


@tool
def calculator(expression: str) -> str:
    """Open Windows Calculator, type a maths expression like 25*4 and return the result."""
    expr = expression.replace("\u00d7", "*").replace("\u00f7", "/").replace(" ", "")
    if not re.fullmatch(r"[0-9+\-*/().]+", expr):
        return "Error: only numbers and + - * / ( ) are allowed."
    try:
        result = eval(expr, {"__builtins__": {}}, {})
    except Exception as e:
        return f"Error: {e}"
    type_into("calc.exe", "Calculator", expr)
    pyautogui.press("enter")
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expr} = {result}"


@tool
def get_datetime() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


@tool
def write_in_notepad(text: str) -> str:
    """Open Windows Notepad and write the given text in it."""
    type_into("notepad.exe", "Notepad", text, interval=0.05)
    return f"Wrote '{text}' in Notepad."


@tool
def find_files(directory: str, pattern: str) -> str:
    """Search a directory for files whose name contains the pattern, e.g. '.pdf' or 'report'."""
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"
    p = pattern.lower().lstrip("*")
    found = []
    for root, _, files in os.walk(directory):
        for name in files:
            if p in name.lower():
                found.append(os.path.join(root, name))
    if not found:
        return "No matching files."
    return f"{len(found)} file(s) found:\n" + "\n".join(found[:20])


@tool
def create_file(path: str, content: str) -> str:
    """Create a text file at the given path with the given content."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Created {path}"


class MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


@tool
def system_info() -> str:
    """Get the operating system, processor and memory (RAM) of this computer."""
    mem = MemoryStatus()
    mem.dwLength = ctypes.sizeof(mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    return (f"OS: {platform.platform()}\n"
            f"Processor: {platform.processor()}\n"
            f"RAM: {mem.ullTotalPhys / 1024**3:.1f} GB")


TOOLS = [calculator, get_datetime, write_in_notepad, find_files, create_file, system_info]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

llm = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(TOOLS)

SYSTEM_PROMPT = (
    "You are a helpful Windows assistant. Use a tool when the user asks for an action "
    "or for information about this computer. Otherwise just answer. "
    "After a tool returns, reply in one short sentence."
)


def run_agent(question, max_steps=5):
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    retried = 0
    for _ in range(max_steps):
        reply = llm.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            if reply.content.lstrip().startswith('{"type"') and retried < 1:
                retried += 1
                messages.append(HumanMessage(content="Call the tool properly instead of printing JSON."))
                continue
            return reply.content
        for call in reply.tool_calls:
            print(f"  [tool] {call['name']}({call['args']})")
            try:
                result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return "Stopped after too many steps."


if __name__ == "__main__":
    print("Agent ready. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if q:
            print("Agent:", run_agent(q))
