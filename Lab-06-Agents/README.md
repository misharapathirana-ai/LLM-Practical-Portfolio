# Lab 06 - Tool-Using AI Agent with Ollama

## Laboratory Session
Lab 06 - PUSL3205 Large Language Models and Generative AI

## Objective
Build an AI agent that uses a local LLM (Llama 3.2 3B via Ollama) and decides when to call tools. Part A is a calculator agent; the exercise extends it to six tools. Parts B to D (memory, RAG tool, Google Calendar) will be added below as they are completed.

## Implementation
`agent_tools.py` defines six tools with LangChain's `@tool` decorator: calculator (opens Windows Calculator and types the expression), date and time, write in Notepad, file search, file creation and system information. The tools are registered with `bind_tools`, and `run_agent` loops: ask the model, run any tool it requests, send the result back, and stop when the model answers without a tool call.

## Results and Evidence
| # | Query | Tool called | Result |
|---|---|---|---|
| 1 | Calculate 25 x 4 | calculator | Worked, answer 100 |
| 2 | What is the date today? | get_datetime | Worked |
| 3 | Open Notepad and write Hello Students | write_in_notepad | Worked |
| 4 | Find all PDF files in a folder | find_files | Failed on first try: the model printed the tool call as text and no tool ran |
| 5 | Create test.txt with Hello World | create_file | Worked |
| 6 | Show my computer specifications | system_info | Worked |

## Observations
- The model chose the right tool and arguments in 5 of 6 queries.
- In query 4 the model wrote the tool call as plain text instead of making a real call, so nothing executed. This is a limitation of a small 3B model. TODO: describe the retry fix and whether it worked.
- The agent describes outcomes it cannot see. It said text was displayed in Notepad, but the tool only reports that it typed it.
- The calculator tool only accepts digits and basic operators, because it evaluates text produced by the model.

## Challenges
- Windows Application Control blocks some compiled packages, so the system information tool uses `ctypes` instead of `psutil`.
- Automation types into whichever window has focus, so the keyboard and mouse must not be used while it runs.

## Learning Outcomes
- How an agent loop lets an LLM decide which tool to call
- How tool names, docstrings and type hints guide the model
- Why tool input must be validated when it comes from an LLM

## Screenshots
![calculator](screenshots/calculator.png)
![computer specs](screenshots/computer%20specs.png)
![create text file](screenshots/create%20text%20file.png)
