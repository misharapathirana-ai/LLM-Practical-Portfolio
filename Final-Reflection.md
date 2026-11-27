# Final Reflection - PUSL3205 Large Language Models and Generative AI

## Overview
This module took me from classic machine learning to building and testing the pieces behind modern LLM applications: a text classifier, a Transformer written from scratch, a fine-tuned model, a retrieval system and tool-using agents. Each lab used what the previous one showed me, and the most useful lesson was how often a model that looks good on one number turns out to be unreliable once I tested it properly.

## What I did and what I learned

**Lab 01 - Sentiment analysis.** I built a full pipeline of cleaning, a stratified split, Bag-of-Words, Logistic Regression and a linear SVM. Both models reached about 80% accuracy, and bigrams added roughly one point. The errors (negation, sarcasm, rare words) showed me exactly what word counts cannot capture. I also learned to fit the vectorizer on training data only, so that test information does not leak in.

**Lab 02 - LLMs in everyday life.** I used an LLM to compare three product ideas for the overloaded student and developed one, LightLoad, with a mock-up. I worked alone because I was absent on the group day. The main learning was to treat the LLM's confident assumptions as things to check, not facts, since I never tested the concept with real students.

**Lab 03 - Tiny Transformer.** I built a 10.8M-parameter GPT-style model from scratch in PyTorch and trained it on TinyStories at character level, reaching a validation loss of 0.679 (perplexity 1.97). My experiments showed that model size and training length mattered most, and that one attention head did worse than six with the same parameters. The stories were fluent but lost the plot, which made the limits of a small model concrete.

**Lab 04 - LoRA fine-tuning.** I fine-tuned Qwen2.5-0.5B on a made-up shop with LoRA, training only 1.75% of the parameters in 2.12 GB of GPU memory. The training loss went down to 0.132, yet the model got most rephrased questions wrong and invented answers to things it was never taught. That showed me fine-tuning changes style far more reliably than it stores facts.

**Lab 05 - RAG.** I built a local pipeline with Llama 3.2 3B, a Qwen3 embedding model, ChromaDB and a Streamlit chat with PDF upload. It answered the word-limit question from the document and refused a question that was not in it, but it left out the date when asked about the deadline. Seeing both outcomes made the difference between retrieval and generation failures real.

**Lab 06 - Agents.** I built a tool-using agent with six tools, then added SQLite memory, a RAG tool and Google Calendar with an approval step. The 3B model sometimes printed a tool call as text instead of making it, which I fixed with a retry. It also reported things it could not see, such as text "displayed" in Notepad. That is why I kept a human approval step before anything changes a real calendar.

## Skills I gained
| Skill | Where I used it |
|---|---|
| Text preprocessing and classical ML evaluation | Lab 01 |
| Writing and training a Transformer in PyTorch | Lab 03 |
| Parameter-efficient fine-tuning (LoRA) on a cloud GPU | Lab 04 |
| Building RAG: chunking, embeddings, vector search, grounded prompts | Lab 05 |
| Agent loops, tool design, persistent memory, OAuth | Lab 06 |
| Judging model output critically and reporting limits honestly | All labs |
| Git workflow, environments and keeping secrets out of a repository | All labs |

## Lessons that stood out
- **Small models fail in characteristic ways:** they invent facts (Lab 04), lose long-range coherence (Lab 03) and mishandle tool calls (Lab 06). Knowing the failure mode tells me what safeguard to add.
- **Grounding and guardrails help, but do not guarantee correctness.** The RAG prompt rule, the calculator's input check and the calendar approval all reduced risk, and none removed it.
- **One number is not enough.** My tests were small (8 questions, single runs, one split), so I wrote my conclusions as observations, not proof.

## Challenges and how I handled them
- My laptop's Application Control policy blocked `pip.exe` and PyMuPDF. I used `python -m pip`, switched to `pypdf`, and used `ctypes` instead of `psutil`.
- Two Python environments caused import errors until I learned to check which one was active.
- A rejected `git push` taught me to pull with rebase before pushing, and I kept login files and databases out of the repository with `.gitignore`.
- Colab sessions disconnect, so I downloaded notebooks and screenshots straight after each run.

## What I would do differently
I would test with more questions and several seeds, add a validation set to the fine-tuning lab, use embeddings instead of keyword search for the agent's memory, and measure retrieval quality in the RAG lab (for example Recall@k) instead of only reading answers.

## How I used AI assistance
I used Claude (Anthropic's AI assistant) during the module for explanations, code drafts and debugging help. I ran all code myself, on my VS Code and on Colab, and the numbers, outputs and screenshots in this repository come from my runs.

## Looking ahead
These results feed directly into my report on a university question-answering assistant: Lab 04 and Lab 05 together support grounding answers in documents with retrieval, with fine-tuning kept for tone and format.
