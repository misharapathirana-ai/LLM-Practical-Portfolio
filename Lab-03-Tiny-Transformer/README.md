# Lab 03 - Building a Tiny Transformer Language Model

## Laboratory Session
Lab 03 - PUSL3205 Large Language Models and Generative AI

## Objective
Build a small GPT-style Transformer from scratch in PyTorch, train it on the TinyStories dataset to predict the next character, evaluate it with validation loss and perplexity, generate stories, and test how the number of layers, heads and training steps change the results.

## Setup
- Google Colab with a GPU runtime
- PyTorch, Hugging Face `datasets`, pandas, matplotlib, Gradio
- Notebook: [notebooks/Lab03_Tiny_Transformer.ipynb](notebooks/Lab03_Tiny_Transformer.ipynb)
- Saved model settings: [tinystories_gpt_config.json](results/tinystories_gpt_config.json) (the 42.6 MB weights file is not in the repo)

## Implementation
- **Data:** 20,000 TinyStories for training and 1,000 different stories for validation, streamed from Hugging Face. Stories are cleaned to plain ASCII and joined with an end-of-story character
- **Tokens:** character level, vocabulary of 81 symbols, 17.9 million training tokens and 0.79 million validation tokens
- **Training pairs:** windows of 256 characters, where the target is the input shifted one character to the right, so the model predicts the next character at every position
- **Architecture:** token and position embeddings, 6 blocks of masked multi-head self-attention (6 heads) and a feed-forward layer, each with layer normalisation before it and a residual connection after it, then a final layer norm and a linear output layer. 384 embedding dimensions, dropout 0.1, 10.78 million parameters
- **Training:** cross-entropy loss, AdamW, learning rate warm-up then cosine decay, gradient clipping at 1.0, mixed precision, batch size 64, 3,000 steps (about 2.7 passes over the training text), 9.1 minutes
- **Generation:** sampling with temperature and top-k (40)
- **Optional:** a Gradio interface for writing stories from a prompt

## Dataset exploration
- Stories average 895 characters (175 words); the median is 152 words, the shortest 43 words and the longest 878
- About 60% start with "Once upon a time" (9,651 with a comma and 2,404 without out of 20,000), and 19,167 contain line breaks (paragraphs)
- The vocabulary is simple and repetitive, which is why a small model can learn it

## Results and Evidence
Sanity check before training: random guessing would give a loss of ln(81) = 4.394, and the untrained model gave 4.486.

| Step | Train loss | Validation loss |
|---|---|---|
| 0 | 4.488 | 4.490 |
| 250 | 1.501 | 1.487 |
| 500 | 1.086 | 1.079 |
| 1000 | 0.857 | 0.854 |
| 2000 | 0.715 | 0.711 |
| 3000 | 0.666 | 0.679 |

Final evaluation: validation loss 0.679, perplexity 1.97 per character (random guessing would be 81), train loss 0.671.

**Settings experiment** (same seed, same data; all 1,000 steps unless stated):

| Setting | Parameters (M) | Steps | Val loss | Perplexity | Minutes |
|---|---|---|---|---|---|
| 2 layers, 2 heads, 128 dim | 0.44 | 1000 | 1.594 | 4.92 | 0.4 |
| 4 layers, 4 heads, 256 dim | 3.25 | 1000 | 1.011 | 2.75 | 1.3 |
| 6 layers, 6 heads, 384 dim | 10.78 | 1000 | 0.865 | 2.37 | 3.0 |
| 6 layers, 1 head, 384 dim | 10.78 | 1000 | 0.934 | 2.54 | 2.0 |
| 6 layers, 6 heads, only 250 steps | 10.78 | 250 | 1.611 | 5.01 | 0.8 |

**Generated text** (prompt "Once upon a time"):
- Temperature 0.5: "Once upon a time, there was a little girl named Lily. She loved to play outside and explore the world around her. One day, she saw a big cat named Sue."
- Temperature 1.2: "Once upon a time there was a little girl who lived near her seed. Then she was always strange a special kindness lived in the school."

## Observations
- The loss fell quickly in the first 250 steps and then flattened slowly. Training and validation loss stayed almost the same (0.671 and 0.679), so the model did not overfit after about 2.7 passes. The curve was still falling, so more training would probably help, but I did not test it.
- The stories have correct spelling and grammar but weak logic. Events do not follow from each other, and when I prompted with the name Mia the story switched to Lily after a few sentences. The model handles local patterns well and long-range consistency poorly.
- Temperature changes the style: 0.5 is clean but repetitive, 0.8 is varied but loses the plot, and 1.2 invents words such as "differencess".
- Model size had the biggest effect. The 0.44M model produced mostly garbled letters, the 3.25M model produced mostly real words with grammar mistakes, and the 10.78M model produced fluent sentences.
- With the same number of parameters, one attention head gave a higher validation loss (0.934) than six heads (0.865). This was one run with one seed, so I cannot say how large the real difference is. The one-head model also trained faster (2.0 against 3.0 minutes), possibly because of the size of the attention tensors, which I did not test.
- Training for only 250 steps gave a validation loss of 1.611 against 0.865 at 1,000 steps, and the text was mostly broken words, so training length mattered a lot.
- Perplexity here is per character and cannot be compared with word-level perplexity from other work.
- Limits: character-level tokens, one run per setting, no quality measure other than loss and my own reading of samples.

## Discussion: mechanisms that improve performance
The mechanisms below are the ones that matter most in my model. For each one I say where it is in the code and whether my own experiments tested it, or whether it is standard theory that I did not test.

- **Causal mask (not tested).** In `SelfAttention.forward`, `masked_fill` with `-inf` before the softmax gives future characters zero weight. Without it the model could read the very character it is supposed to predict, so training would look good but generation, where there is no future, would fail. I did not train without it.
- **Multi-head attention (tested).** `n_head` splits the 384 dimensions into separate heads that each compute their own attention pattern, so different heads can focus on different relationships between characters. With the same number of parameters, 6 heads reached a validation loss of 0.865 and 1 head reached 0.934. This was one run with one seed, so I cannot say how big the real difference is.
- **Position embeddings (not tested).** `pos_emb` is added to the token embeddings. Attention alone treats the input as a set and has no sense of order, so without positions the model could not tell where a character is in the sentence.
- **Residual connections and layer normalisation (not tested).** Each block does `x = x + self.attn(self.ln1(x))`. The addition gives gradients a direct path through all 6 blocks, and layer norm keeps the values at a stable scale. Training stayed smooth with no loss spikes at a learning rate of 1e-3, which fits this, but I did not train without them to compare.
- **Feed-forward layers (not tested).** Attention mixes information between positions, and the feed-forward layer (a 4 times wider layer with GELU) then processes each position on its own, adding non-linear capacity.
- **Training tricks (not tested individually).** Learning-rate warm-up with cosine decay, gradient clipping at 1.0, AdamW with weight decay and mixed precision. The loss curve fell smoothly and the whole run took 9.1 minutes, but I did not switch any of these off to measure its effect.
- **Dropout (not tested).** Dropout of 0.1 is used in attention, in the residual paths and in the feed-forward layer. Training loss (0.671) and validation loss (0.679) stayed almost equal, so there was no overfitting, but after only about 2.7 passes over the data it may not have been needed, and I did not compare with no dropout.
- **Temperature and top-k at generation (tested).** The three samples show the effect: at 0.5 the text is clean but repetitive, at 0.8 it is varied but loses the plot, and at 1.2 it invents words.
- **Model size and training length (tested).** Validation loss was 1.594 for the 0.44M model, 1.011 for the 3.25M model and 0.865 for the 10.78M model. These runs changed layers, heads and width together, so I cannot credit the number of layers alone. Training for only 250 steps gave 1.611 compared with 0.865 at 1,000 steps, so training length mattered a lot.

## Challenges
- The full TinyStories file is about 2 GB, so I used streaming mode and took only 20,000 stories.
- Colab sessions disconnect when idle, so the notebook, configuration and screenshots had to be downloaded right after the run.

## Learning Outcomes
- How self-attention, the causal mask, residual connections and layer normalisation fit together in a Transformer block
- How next-token training pairs are built and how cross-entropy loss and perplexity measure a language model
- How model size, number of heads and training length change the loss and the quality of generated text
- How temperature and top-k sampling change generated text

## Screenshots
![loss curves](results/loss%20curves.png)
![story generator](results/story%20generator.png)
![story lenght](results/story%20lenght.png)


