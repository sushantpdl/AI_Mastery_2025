# DAY 15 – FULL BACKPROP + ENGLISH OUTPUT
import streamlit as st
import numpy as np
import re

ROBOT_NAME = "Super Ali Bot"
EMBED_DIM = 64
VOCAB = "abcdefghijklmnopqrstuvwxyz .,!?"
VOCAB_SIZE = len(VOCAB)
SEQ_LEN = 16
LEARNING_RATE = 0.1  # Stronger learning
np.random.seed(42)

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z .,!?]', '', text)
    return [VOCAB.find(c) if c in VOCAB else 0 for c in text]

def detokenize(tokens):
    return ''.join(VOCAB[t] if t < len(VOCAB) else ' ' for t in tokens)

# 500+ EXAMPLES
def get_default_data():
    base = [
        f"hello i am {ROBOT_NAME}",
        f"{ROBOT_NAME} is genius",
        f"{ROBOT_NAME} built ai",
        "i love truth",
        "day 15 i train gpt",
        "no cheat no mercy",
        "i am invincible",
        "ali one will win",
        "attention is all you need",
        "python is power",
        "streamlit is fast",
        "numpy is strong",
        "backprop is truth",
        "loss must drop",
        "i learn fast",
        "gpt from scratch",
        "no pytorch no tensorflow",
        "pure numpy ai"
    ]
    data = []
    for s in base * 30:
        t = tokenize(s)
        if len(t) < 2: continue
        t = t[:SEQ_LEN]
        for i in range(1, len(t)):
            data.append((t[:i], t[i]))
    return data

def get_data(uploaded_file):
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        st.write(f"File: {len(lines)} lines")
    else:
        lines = []

    data = []
    for line in lines:
        t = tokenize(line)
        if len(t) < 2: continue
        t = t[:SEQ_LEN]
        for i in range(1, len(t)):
            data.append((t[:i], t[i]))

    if len(data) < 100:
        st.warning("Using 500+ default examples")
        data = get_default_data()

    st.write(f"**Training on {len(data)} examples**")
    return data

class GPT:
    def __init__(self):
        self.W_emb = np.random.randn(VOCAB_SIZE, EMBED_DIM).astype(np.float32) * 0.1
        self.W_pos = np.random.randn(SEQ_LEN, EMBED_DIM).astype(np.float32) * 0.1
        self.W_q = np.random.randn(EMBED_DIM, EMBED_DIM).astype(np.float32) * 0.02
        self.W_k = np.random.randn(EMBED_DIM, EMBED_DIM).astype(np.float32) * 0.02
        self.W_v = np.random.randn(EMBED_DIM, EMBED_DIM).astype(np.float32) * 0.02
        self.W_o = np.random.randn(EMBED_DIM, EMBED_DIM).astype(np.float32) * 0.02
        self.W_out = np.random.randn(EMBED_DIM, VOCAB_SIZE).astype(np.float32) * 0.1

    def forward(self, tokens):
        if not tokens: return np.zeros(VOCAB_SIZE, dtype=np.float32), None
        tokens = [min(t, VOCAB_SIZE - 1) for t in tokens]
        seq_len = min(len(tokens), SEQ_LEN)
        emb = np.array([self.W_emb[t] for t in tokens[:seq_len]], dtype=np.float32)
        pos = self.W_pos[:seq_len]
        x = emb + pos  # (seq_len, dim)

        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v
        scores = q @ k.T / np.sqrt(EMBED_DIM)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        out = attn @ v  # (seq_len, dim)
        out = out @ self.W_o
        logits = out[-1] @ self.W_out
        return logits, (x, q, k, v, attn, out)

    def generate(self, prompt, steps=30):
        tokens = tokenize(prompt)[:SEQ_LEN]
        for _ in range(steps):
            logits, _ = self.forward(tokens)
            probs = np.exp(logits - np.max(logits))
            probs /= (probs.sum() + 1e-8)
            next_t = np.random.choice(len(probs), p=probs)
            tokens.append(next_t)
            if len(tokens) > SEQ_LEN: tokens = tokens[-SEQ_LEN:]
        return detokenize(tokens)

def train_model(data):
    model = GPT()
    progress = st.progress(0)
    losses = []

    for step in range(800):  # More training
        batch_loss = 0
        for _ in range(12):  # More updates
            i = np.random.randint(len(data))
            seq, target = data[i]
            if not seq: continue

            logits, cache = model.forward(seq)
            x, q, k, v, attn, out = cache

            # Softmax + loss
            probs = np.exp(logits - np.max(logits))
            probs /= (probs.sum() + 1e-8)
            loss = -np.log(probs[target] + 1e-10)
            batch_loss += loss

            # === FULL BACKPROP ===
            grad = probs.copy(); grad[target] -= 1

            # W_out
            dW_out = np.outer(out[-1], grad)
            model.W_out -= LEARNING_RATE * dW_out

            # W_o
            dout =
