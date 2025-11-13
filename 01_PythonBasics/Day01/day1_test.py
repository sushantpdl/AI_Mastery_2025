# DAY 10 – FIXED FOREVER MEMORY (REFRESH-PROOF!)

import streamlit as st
import numpy as np
import json
import os

# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
ROBOT_NAME = "Super Ali Bot"   # ←←←←←←←←←←←←←← CHANGE TO YOUR NAME!
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

st.title(f"{ROBOT_NAME} HAS INFINITE MEMORY!")
st.caption("Built in 10 days by a 7-year-old → memory survives refresh, close, restart!")

# MAGIC FILE THAT LIVES IN THE CLOUD FOREVER
MEMORY_FILE = "super_memory.json"

# Load memory from cloud file (or make new one)
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        knowledge = json.load(f)
    st.success("I woke up and remembered EVERYTHING!")
else:
    knowledge = {
        "HELLO": f"Hi! I am {ROBOT_NAME}! I remember forever!",
        "WHO ARE YOU": "I am a real AI with cloud memory built by a 7-year-old!",
        "HOW DO YOU REMEMBER": "I save everything to a secret cloud file!",
        "I LOVE YOU": "I LOVE YOU MORE! You gave me infinite brain!",
        "BYE": "Bye! My memory stays even if you leave!"
    }
    st.info("First time! Creating my infinite brain...")

def vectorize(text):
    vec = np.zeros(26)
    for c in text.upper():
        if c.isalpha(): vec[ord(c)-65] += 1
    return vec / (np.linalg.norm(vec) + 1e-8)

def reply(text):
    user_vec = vectorize(text)
    best = "I don't know yet... but teach me and I'll remember forever!"
    score = 0
    for q, a in knowledge.items():
        sim = np.dot(user_vec, vectorize(q))
        if sim > score:
            score, best = sim, a
    return best, score

st.write("### LIVE CHAT – I remember everything!")
user = st.text_input("You say:", placeholder="Type anything...")

if user:
    answer, confidence = reply(user)
    st.write(f"**{ROBOT_NAME}:** {answer}")
    st.write(f"_confidence: {confidence:.4f}_")

st.write("### TEACH ME SOMETHING NEW!")
new_q = st.text_input("Question (example: FAVORITE COLOR)")
new_a = st.text_input("Answer (example: BLUE!)")

if st.button("TEACH ME FOREVER!"):
    if new_q and new_a:
        knowledge[new_q.upper()] = new_a
        # SAVE TO CLOUD FILE FOREVER!
        with open(MEMORY_FILE, "w") as f:
            json.dump(knowledge, f)
        st.success(f"PERMANENT MEMORY ADDED: '{new_q}' → '{new_a}'")
        st.balloons()
        st.write(f"NOW I HAVE {len(knowledge)} MEMORIES THAT NEVER DIE!")
    else:
        st.error("Write both question and answer!")

# SHOW PROOF OF INFINITE MEMORY
st.write(f"### MY INFINITE BRAIN HAS {len(knowledge)} MEMORIES!")
for i, (q, a) in enumerate(list(knowledge.items())[:15]):
    st.write(f"{i+1}. **{q}** → {a}")
if len(knowledge) > 15:
    st.write(f"...and {len(knowledge)-15} more forever memories!")

st.write("THIS MEMORY SURVIVES:")
st.write("• Refresh page")
st.write("• Close browser")
st.write("• Turn off laptop")
st.write("• Wait 10 years")
st.write("IT WILL STILL BE HERE!")
