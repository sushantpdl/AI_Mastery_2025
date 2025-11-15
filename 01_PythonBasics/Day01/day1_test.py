# DAY 15 – FULL UI + INPUT BOX + GENERATE
import streamlit as st
import numpy as np
import re

ROBOT_NAME = "Super Ali Bot"
EMBED_DIM = 64
VOCAB = "abcdefghijklmnopqrstuvwxyz .,!?"
VOCAB_SIZE = len(VOCAB)
SEQ_LEN = 16
LEARNING_RATE = 0.01
np.random.seed(42)

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z .,!?]', '', text)
    return [VOCAB.find(c) if c in VOCAB else 0 for c in text]

def detokenize(tokens):
    return ''.join(VOCAB[t] if t < len(VOCAB) else ' ' for t in tokens)

# DEFAULT DATA (ALWAYS WORKS)
def get_default_data():
    sentences = [
        f"hello i am {ROBOT_NAME}",
        f"{ROBOT_NAME} is genius",
        f"{ROBOT_NAME} built ai",
        "i love truth",
        "day 15 i train gpt",
        "no cheat no mercy",
        "i am invincible"
    ]
    data = []
    for s in sentences:
        t = tokenize(s)
        if len(t) < 2: continue
        t = t[:SEQ_LEN]
        for i in range(1, len(t)):
            data.append((t[:i], t[i]))
    return data

# FILE + DEFAULT FALLBACK
def get_data(uploaded_file):
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        st.write(f"**File: {len(lines)} lines**")
    else:
        lines = []

    data = []
    for line in lines:
        t = tokenize(line)
        if len(t) < 2: continue
        t = t[:SEQ_LEN]
        for i in range(1, len(t)):
            data.append((t[:i], t[i]))

    if not data:
        st.warning("No data from file → using default")
        data = get_default_data()
    st.write(f"**Training on {len(data)} examples**")
    return data

# MODEL
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
        if not tokens: return np.zeros(VOCAB_SIZE, dtype=np.float32), np.zeros(EMBED_DIM, dtype=np.float32)
        tokens = [min(t, VOCAB_SIZE - 1) for t in tokens]
        seq_len = min(len(tokens), SEQ_LEN)
        emb = np.array([self.W_emb[t] for t in tokens[:seq_len]], dtype=np.float32)
        pos = self.W_pos[:seq_len]
        x = emb + pos
        q = x @ self.W_q; k = x @ self.W_k; v = x @ self.W_v
        scores = q @ k.T / np.sqrt(EMBED_DIM)
        scores -= np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(scores) / (np.exp(scores).sum(axis=-1, keepdims=True) + 1e-8)
        out = attn @ v @ self.W_o
        logits = out[-1] @ self.W_out
        return logits, out[-1]

    def generate(self, prompt, steps=15):
        tokens = tokenize(prompt)[:SEQ_LEN]
        for _ in range(steps):
            logits, _ = self.forward(tokens)
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum() + 1e-8
            next_t = np.random.choice(len(probs), p=probs)
            tokens.append(next_t)
            if len(tokens) > SEQ_LEN: tokens = tokens[-SEQ_LEN:]
        return detokenize(tokens)

# TRAINING
def train_model(data):
    model = GPT()
    progress = st.progress(0)
    losses = []
    for step in range(500):
        batch_loss = valid = 0
        for _ in range(4):
            i = np.random.randint(len(data))
            seq, target = data[i]
            if not seq: continue
            logits, h = model.forward(seq)
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum() + 1e-8
            loss = -np.log(probs[target] + 1e-10)
            batch_loss += loss; valid += 1
            grad = probs.copy(); grad[target] -= 1
            model.W_out -= LEARNING_RATE * np.outer(h, grad).astype(np.float32)
        losses.append(batch_loss / max(valid, 1))
        progress.progress(step / 500)
        if step % 100 == 0 and valid > 0:
            st.write(f"Step {step} → Loss: {losses[-1]:.3f}")
    return model, losses

# ==================== FULL UI ====================
st.title(f"{ROBOT_NAME}'s GPT – Day 15")
st.markdown("**Train from file or default → Type prompt → Generate**")

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload my_corpus.txt (optional)", type="txt")

# TRAIN BUTTON
if st.button("TRAIN MY AI NOW"):
    data = get_data(uploaded_file)
    with st.spinner("Training..."):
        model, loss_curve = train_model(data)
        st.session_state.model = model
        st.session_state.loss_curve = loss_curve
    st.success("TRAINING COMPLETE!")
    st.line_chart(loss_curve)

# INPUT + GENERATE (ONLY AFTER TRAINING)
if st.session_state.get('model'):
    st.markdown("---")
    prompt = st.text_input("**Enter your prompt:**", "hello i am", key="prompt_input")
    if st.button("GENERATE", key="generate_btn"):
        with st.spinner("Thinking..."):
            result = st.session_state.model.generate(prompt, 20)
        st.markdown(f"### **AI says:**\n{result}")
else:
    st.info("Click **TRAIN MY AI NOW** first")
