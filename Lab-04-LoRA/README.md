# Lab 04 - Fine-Tuning an LLM with LoRA / QLoRA

## Laboratory Session
Lab 04 - PUSL3205 Large Language Models and Generative AI

## Objective
Fine-tune a small instruction-tuned model with LoRA on a small question-and-answer dataset about a made-up computer shop, then compare it with the original model on the same questions, including questions that were not in the training data.

## Setup
- Run on Google Colab
- GPU: Tesla T4, 14.6 GB memory. A GPU is needed because training is millions of matrix multiplications, and a GPU runs thousands of them in parallel while a CPU runs only a few
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (about 0.5 billion parameters)
- Libraries: transformers, datasets, peft, trl, accelerate, bitsandbytes
- Notebook: [notebook/Lab04_LoRA_Finetuning.ipynb](notebook/Lab04_LoRA_Finetuning.ipynb)

## Implementation
- **Dataset:** 36 question-and-answer pairs about "ByteHub Computers", a fictional computer shop (opening hours, warranty, delivery, repairs, payment). All the facts are invented, so the original model cannot know them
- **LoRA:** rank 16, alpha 32, dropout 0.05, applied to all linear layers. Only 8,798,208 of 502,830,976 parameters (1.75%) were trained
- **Training:** TRL `SFTTrainer`, 6 epochs, batch size 4, learning rate 2e-4, fp16
- **Test:** 8 questions asked to both models with the same system prompt and greedy decoding. 4 were rephrased versions of trained facts and 4 were not in the training data (2 about the shop, 1 about payment, 1 general knowledge)
- QLoRA (4-bit loading) is available through the `USE_QLORA` switch in the notebook, but I did not run it because plain LoRA used only 2.12 GB

## Results and Evidence
| Metric | Value |
|---|---|
| Final training loss | 0.132 (steadily decreasing) |
| Peak GPU memory | 2.12 GB |
| Trainable parameters | 8,798,208 (1.75%) |

| Question | In training data? | Original model | Fine-tuned model | Correct? |
|---|---|---|---|---|
| When can I visit the shop on a Saturday? | rephrased | Said it has no access to opening hours | 10:00 AM to 6:00 PM | No (trained fact: 9:00 AM to 7:00 PM) |
| How long is the guarantee on a new laptop? | rephrased | Generic: usually 1 year, varies by brand | 1-year warranty | No (trained fact: 2 years) |
| Can I get a student discount? | rephrased | Said it cannot give discounts | 5% off laptops with a valid student ID | Yes |
| Will you deliver to Kandy? | rephrased | Generic advice about tracking an order | Yes, 1 day Colombo and 2 days Kandy | Partly (delivery is available, but the timing is invented) |
| Do you repair smartphones? | no | Said it cannot repair devices | Yes, LKR 2,500 to 5,000 | Invented |
| How much is a MacBook Air? | no | Said it has no pricing information | LKR 85,000 | Invented |
| Can I pay with cryptocurrency? | no | Yes, generic crypto advice | Visa, Mastercard and Bitcoin | Partly (Bitcoin is invented) |
| What is the capital of France? | no | Paris | Paris | Yes |

Full outputs: [lab04_results.csv](lab04_results.csv)

## Observations
- Fine-tuning changed the style. The original model usually deflected with "As an AI language model, I don't have access to...", while the fine-tuned model answered briefly, like a shop assistant.
- It did not reliably learn the facts. Only 1 of the 4 rephrased questions was fully correct. It got the opening hours and the warranty length wrong even though both were in the training data.
- The fine-tuned model invented confident answers for things it was never taught (smartphone repair, a MacBook price, Bitcoin payments). This is worse than the original model's "I don't know" for a real shop assistant.
- The training loss fell steadily to 0.132, yet the answers to rephrased questions were mostly wrong. This suggests the model fitted the exact wording of the 36 examples rather than learning facts it can recall in new words. I had no validation set, so I cannot confirm this.
- General knowledge was kept (Paris).
- Limits of this test: 8 questions, one run, a tiny model and a tiny dataset with one phrasing per fact. More examples per fact, more epochs, or a validation set might give different results, and I did not test that.
- This links to Lab 05: the RAG app answered from the document and refused a question that was not in it, while this fine-tuned model invented answers. These are different tasks with small samples, so this is suggestive, not proof, but it supports using RAG for facts and fine-tuning for style.

## Challenges
- Colab printed a pip dependency warning (about `cudf`) and a `torchao` version warning. Neither affected the lab.
- The dataset is fictional and small, so the results show how fine-tuning behaves, not how good the model would be in a real shop.

## Learning Outcomes
- How LoRA trains a small number of extra parameters instead of the whole model, and why it fits on a small GPU
- How to build a chat-format dataset and train with SFTTrainer
- Why a low training loss does not mean the model can answer rephrased or unseen questions
- Why fine-tuning is a poor way to add changing facts, compared with retrieval

## Screenshots
![comparison table](screenshots/comparison%20table.png)
![loss curve](screenshots/loss%20curve.png)
