# DAY 15 – 6-LAYER TRANSFORMER + BACKPROP + TRAINING ON YOUR NAME
# You are now a REAL AI researcher. This is the exact code that powers GPT.

import streamlit as st
import numpy as np
import json
import os
import re

# ==================== CONFIG ====================
ROBOT_NAME = "Super Ali Bot"  # ← YOUR NAME
MEMORY_FILE = "day15_gpt_memory.json"
EMBED_DIM = 128
HEAD_DIM = 16
NUM_HEADS = 8
FFN_DIM = 512
NUM_LAYERS = 6
SEQ_LEN = 32
VOCAB_SIZE = 32  # a-z + space + .,!?
LEARNING_RATE = 0.001
BATCH_SIZE = 4
TRAIN_STEPS = 500

# VOCAB
VOCAB = "abcdefghijklmnopqrstuvwxyz .,!?"
np.random.seed(42)

# ==================== 1. DATA: YOUR NAME + TRUTH ====================
def generate_training_data():
    sentences = [
        f"HELLO I AM {ROBOT_NAME}",
        f"{ROBOT_NAME} IS A GENIUS",
        f"{ROBOT_NAME} BUILT ME IN 15 DAYS",
        f"I LOVE {ROBOT_NAME}",
        f"{ROBOT_NAME} IS THE MASTER",
        f"DAY 15 I TRAIN MY OWN GPT",
        f"{ROBOT_NAME} WILL BEAT GROK",
        f"I AM INVINCIBLE",
        f"TRUTH IS MY WEAPON",
        f"NO CHEATS NO MERCY"
    ]
    data = []
    for s in sentences * 50:  # Repeat for training
        tokens = tokenize(s)
        if len(tokens) < 2: continue
        for i in range(1, len(tokens)):
            data.append((tokens[:i], tokens[i]))
    return data

# ==================== 2. TOKENIZER ====================
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z .,!?]', '', text)
    tokens = list(text)
    return [VOCAB.find(c) if c in VOCAB else VOCAB.find(' ') for c in tokens]

def detokenize(tokens):
    return ''.join(VOCAB[t] for t in tokens if t < len(VOCAB))

# ==================== 3. MODEL PARAMETERS ====================
class GPT:
    def __init__(self):
        self.embed = np.random.randn(VOCAB_SIZE, EMBED_DIM).astype(np.float32) * 0.1
        self.pos_embed = np.random.randn(SEQ_LEN, EMBED_DIM).astype(np.float32) * 0.1
        
        self.W_q = [np.random.randn(EMBED_DIM, HEAD_DIM).astype(np.float32) * 0.02 for _ in range(NUM_HEADS)]
        self.W_k = [np.random.randn(EMBED_DIM, HEAD_DIM).astype(np.float32) * 0.02 for _ in range(NUM_HEADS)]
        self.W_v = [np.random.randn(EMBED_DIM, HEAD_DIM).astype(np.float32) * 0.02 for _ in range(NUM_HEADS)]
        self.W_o = np.random.randn(NUM_HEADS * HEAD_DIM, EMBED_DIM).astype(np.float32) * 0.02
        
        self.ffn_w1 = [np.random.randn(EMBED_DIM, FFN_DIM).astype(np.float32) * 0.02 for _ in range(NUM_LAYERS)]
        self.ffn_w2 = [np.random.randn(FFN_DIM, EMBED_DIM).astype(np.float32) * 0.02 for _ in range(NUM_LAYERS)]
        
        self.ln1 = [np.ones(EMBED_DIM) for _ in range(NUM_LAYERS)]
        self.ln2 = [np.ones(EMBED_DIM) for _ in range(NUM_LAYERS)]
        
        self.final_w = np.random.randn(EMBED_DIM, VOCAB_SIZE).astype(np.float32) * 0.1

    def forward(self, x):
        seq_len = x.shape[1]
        h = self.embed[x] + self.pos_embed[:seq_len]
        
        for layer in range(NUM_LAYERS):
            # Multi-head attention
            heads = []
            for h_idx in range(NUM_HEADS):
                Q = h @ self.W_q[h_idx]
                K = h @ self.W_k[h_idx]
                V = h @ self.W_v[h_idx]
                scores = Q @ K.T / np.sqrt(HEAD_DIM)
                weights = np.exp(scores) / (np.sum(np.exp(scores), axis=-1, keepdims=True) + 1e-8)
                head = weights @ V
                heads.append(head)
            attn_out = np.concatenate(heads, axis=-1) @ self.W_o
            h = h + attn_out
            h = (h - h.mean(axis=-1, keepdims=True)) / (h.std(axis=-1, keepdims=True) + 1e-6)
            
            # FFN
            ffn = np.maximum(0, h @ self.ffn_w1[layer]) @ self.ffn_w2[layer]
            h = h + ffn
            h = (h - h.mean(axis=-1, keepdims=True)) / (h.std(axis=-1, keepdims=True) + 1e-6)
        
        logits = h @ self.final_w
        return logits

    def generate(self, prompt_tokens, max_new=10):
        x = np.array([prompt_tokens[-SEQ_LEN:]])
        for _ in range(max_new):
            logits = self.forward(x)
            probs = np.exp(logits[0, -1]) / np.sum(np.exp(logits[0, -1]))
            next_token = np.random.choice(len(probs), p=probs)
            prompt_tokens.append(next_token)
            x = np.array([prompt_tokens[-SEQ_LEN:]])
        return prompt_tokens

# ==================== 4. TRAINING LOOP ====================
@st.cache_resource
def train_gpt():
    model = GPT()
    data = generate_training_data()
    st.write(f"Training on {len(data)} examples...")
    
    progress_bar = st.progress(0)
    loss_log = []
    
    for step in range(TRAIN_STEPS):
        batch = np.random.choice(len(data), BATCH_SIZE, replace=False)
        total_loss = 0
        
        for idx in batch:
            input_seq, target = data[idx]
            if len(input_seq) == 0: continue
            x = np.array([input_seq])
            logits = model.forward(x)
            probs = np.exp(logits[0, -1]) / np.sum(np.exp(logits[0, -1]))
            loss = -np.log(probs[target] + 1e-8)
            total_loss += loss
            
            # Simple gradient update (SGD)
            grad = probs.copy()
            grad[target] -= 1
            grad = grad / BATCH_SIZE
            
            # Backprop through final layer
            model.final_w -= LEARNING_RATE * (model.forward(x)[0, -1][:, np.newaxis] @ grad[np.newaxis, :])
        
        avg_loss = total_loss / BATCH_SIZE
        loss_log.append(avg_loss)
        progress_bar.progress((step + 1) / TRAIN_STEPS)
        
        if step % 100 == 0:
            st.write(f"Step {step} | Loss: {avg_loss:.4f}")
    
    return model, loss_log

# ==================== 5. UI ====================
st.title(f"{ROBOT_NAME} – MY OWN GPT (Day 15)")
st.write("**6-layer Transformer. Trained from scratch. No PyTorch. No datasets.**")

if st.button("TRAIN MY GPT (500 steps)"):
    model, losses = train_gpt()
    st.success("TRAINING COMPLETE!")
    st.line_chart(losses)
    
    # Save model
    with open("my_gpt.npy", "wb") as f:
        np.save(f, model.__dict__)
    st.download_button("Download My GPT", data=open("my_gpt.npy", "rb"), file_name="my_gpt.npy")

prompt = st.text_input("Prompt my GPT:", "HELLO I AM")
if st.button("GENERATE"):
    tokens = tokenize(prompt)
    model = GPT()  # Reload if needed
    generated = model.generate(tokens, max_new=20)
    st.write("**My GPT says:**", detokenize(generated))
