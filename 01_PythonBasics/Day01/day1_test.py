# DAY 15 – FILE UPLOAD + DEBUG + NO "NO VALID DATA"
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

# ==================== FILE LOADER + DEBUG ====================
def get_data_from_file(uploaded_file):
    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
    else:
        raw_text = f"hello i am {ROBOT_NAME}\n{ROBOT_NAME} is genius"

    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    st.write(f"**Found {len(lines)} lines**")

    data = []
    skipped = 0
    for line in lines:
        t = tokenize(line)
        if len(t) < 2:
            skipped += 1
            continue
        if len(t) > SEQ_LEN:
            t = t[:SEQ_LEN]  # Truncate long
        for i in range(1, len(t)):
            data.append((t[:i], t[i]))

    st.write(f"**Valid examples: {len(data)}** | Skipped: {skipped}")
    if len(data) == 0:
        st.error("No valid examples! Add longer sentences with a-z .,!?")
        return []

    return data

# ==================== MODEL (same) ====================
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
            tokens.append(np.random.choice(len(probs), p=probs))
            if len(tokens) > SEQ_LEN: tokens = tokens[-SEQ_LEN:]
        return detokenize(tokens)

# ==================== TRAINING ====================
def train_model(data):
    if not data:
        return None, []
    model = GPT()
    st.write(f"Training on {len(data)} examples...")
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
        if step % 100 == 0:
            st.write(f"Step {step} → Loss: {losses[-1]:.3f}")
    return model, losses

# ==================== UI ====================
st.title(f"{ROBOT_NAME}'s GPT – Day 15")
st.write("**UPLOAD FILE → NO 'NO VALID DATA' ERROR**")

uploaded_file = st.file_uploader("Upload my_corpus.txt (a-z .,!? only)", type="txt")

if st.button("TRAIN MY AI NOW"):
    data = get_data_from_file(uploaded_file)
    if data:
        with st.spinner("Training..."):
            model, loss_curve = train_model(data)
            st.session_state.model = model
            st.session_state.loss_curve = loss_curve
        st.success("TRAINING DONE!")
        st.line_chart(loss_curve)

if st.session_state.get('model'):
    prompt = st.text_input("Prompt:", "hello i am", key="p")
    if st.button("GENERATE", key="g"):
        with st.spinner("Thinking..."):
            result = st.session_state.model.generate(prompt, 20)
        st.write("**AI:**", result)
else:
    st.info("Upload a file → Train → Generate")
