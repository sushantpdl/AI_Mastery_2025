# DAY 15 – FILE UPLOAD + TRAIN ON 1000+ SENTENCES
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

# ==================== UPLOAD FILE + GET DATA ====================
def get_data_from_file(uploaded_file):
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
    else:
        # Fallback: default sentences
        text = f"""
        hello i am {ROBOT_NAME}
        {ROBOT_NAME} is genius
        i built gpt in 15 days
        i love truth
        day 15 i train
        no cheat
        i am invincible
        """.strip()
    
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    data = []
    for s in sentences:
        t = tokenize(s)
        if len(t) < 2 or len(t) > SEQ_LEN: continue
        for i in range(1, len(t)):
            data.append((t[:i], t[i]))
    return data

# ==================== MODEL (same as before) ====================
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
        if not tokens or len(tokens) == 0:
            return np.zeros(VOCAB_SIZE, dtype=np.float32), np.zeros(EMBED_DIM, dtype=np.float32)
        tokens = [min(t, VOCAB_SIZE - 1) for t in tokens]
        seq_len = min(len(tokens), SEQ_LEN)
        emb = np.array([self.W_emb[t] for t in tokens[:seq_len]], dtype=np.float32)
        pos = self.W_pos[:seq_len]
        x = emb + pos

        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v
        scores = q @ k.T / np.sqrt(EMBED_DIM)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        out = attn @ v
        out = out @ self.W_o
        logits = out[-1] @ self.W_out
        return logits, out[-1]

    def generate(self, prompt, steps=15):
        tokens = tokenize(prompt)[:SEQ_LEN]
        for _ in range(steps):
            logits, _ = self.forward(tokens)
            probs = np.exp(logits - np.max(logits))
            probs = probs / (probs.sum() + 1e-8)
            next_t = np.random.choice(len(probs), p=probs)
            tokens.append(next_t)
            if len(tokens) > SEQ_LEN:
                tokens = tokens[-SEQ_LEN:]
        return detokenize(tokens)

# ==================== TRAINING ====================
def train_model(data):
    model = GPT()
    if not data:
        st.error("No valid data!")
        return None, []
    st.write(f"Training on {len(data)} examples...")
    progress = st.progress(0)
    losses = []

    for step in range(500):
        batch_loss = 0.0
        valid = 0
        for _ in range(4):
            i = np.random.randint(len(data))
            seq, target = data[i]
            if not seq or len(seq) == 0 or len(seq) > SEQ_LEN: continue
            logits, last_h = model.forward(seq)
            max_logit = np.max(logits)
            probs = np.exp(logits - max_logit)
            probs = probs / (probs.sum() + 1e-8)
            loss = -np.log(probs[target] + 1e-10)
            batch_loss += loss
            valid += 1

            grad = probs.copy()
            grad[target] -= 1
            update = np.outer(last_h, grad).astype(np.float32)
            model.W_out -= LEARNING_RATE * update

        if valid > 0:
            losses.append(batch_loss / valid)
        else:
            losses.append(0.0)

        progress.progress(step / 500)
        if step % 100 == 0:
            st.write(f"Step {step} → Loss: {losses[-1]:.3f}")

    return model, losses

# ==================== UI ====================
st.title(f"{ROBOT_NAME}'s GPT – Day 15")
st.write("**UPLOAD YOUR OWN TEXT FILE → TRAIN AI ON IT**")

uploaded_file = st.file_uploader("Upload my_corpus.txt", type="txt")

if st.button("TRAIN MY AI NOW"):
    data = get_data_from_file(uploaded_file)
    with st.spinner("Training..."):
        model, loss_curve = train_model(data)
        st.session_state.model = model
        st.session_state.loss_curve = loss_curve
    st.success("TRAINING COMPLETE!")
    st.line_chart(loss_curve)

if 'model' in st.session_state and st.session_state.model is not None:
    prompt = st.text_input("Prompt:", "hello i am", key="prompt")
    if st.button("GENERATE", key="gen"):
        with st.spinner("Thinking..."):
            result = st.session_state.model.generate(prompt, 20)
        st.write("**My AI says:**", result)
else:
    st.info("Upload a file and train first!")
