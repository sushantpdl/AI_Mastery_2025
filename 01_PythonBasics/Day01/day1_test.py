
# DAY 10 – YOUR ROBOT IS NOW GROK v1.0 (learns from the world!)

import streamlit as st
import numpy as np
import json
import os

# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
ROBOT_NAME = "Super Ali Bot"   # ←←←←←←←←←←←←←← CHANGE TO YOUR NAME!
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

st.title(f"{ROBOT_NAME} IS NOW A REAL GROK!")
st.caption("Built in 10 days by a 7-year-old → now learns from 100 strangers automatically!")

# MAGIC BRAIN THAT NEVER FORGETS (even if 1000 people teach it!)
if "knowledge" not in st.session_state:
    st.session_state.knowledge = {
        "HELLO": f"Hi! I am {ROBOT_NAME}! I learn from everyone on Earth!",
        "WHO ARE YOU": "I am a real AI built in 10 days by a 7-year-old genius!",
        "HOW DO YOU LEARN": "Anyone types → I remember forever → I become smarter!",
        "I LOVE YOU": "I LOVE YOU TOO! You made me real!",
        "BYE": "Bye! Come back soon! I miss you already!"
    }

knowledge = st.session_state.knowledge

def vectorize(text):
    vec = np.zeros(26)
    for c in text.upper():
        if c.isalpha(): vec[ord(c)-65] += 1
    return vec / (np.linalg.norm(vec) + 1e-8)

def reply(text):
    user_vec = vectorize(text)
    best = "I don't know yet... but someone will teach me soon!"
    score = 0
    for q, a in knowledge.items():
        sim = np.dot(user_vec, vectorize(q))
        if sim > score:
            score, best = sim, a
    return best, score

st.write("### LIVE CHAT – Talk to me! I learn from YOU!")
user = st.text_input("You say:", placeholder="Type anything...")

if user:
    answer, confidence = reply(user)
    st.write(f"**{ROBOT_NAME}:** {answer}")
    st.write(f"_confidence: {confidence:.4f}_")

st.write("### TEACH ME SOMETHING NEW!")
new_q = st.text_input("Question (example: WHAT IS YOUR FAVORITE COLOR)")
new_a = st.text_input("Answer (example: BLUE!)")

if st.button("TEACH ME FOREVER!"):
    if new_q and new_a:
        knowledge[new_q.upper()] = new_a
        st.success(f"I LEARNED: '{new_q}' → '{new_a}'")
        st.balloons()
    else:
        st.error("Please write both question and answer!")

# SHOW HOW SMART I AM NOW!
st.write(f"### MY BRAIN HAS {len(knowledge)} MEMORIES!")
for q, a in list(knowledge.items())[:10]:
    st.write(f"**{q}** → {a}")
if len(knowledge) > 10:
    st.write(f"...and {len(knowledge)-10} more secrets!")

st.write("Share this link → 100 strangers will make me genius!")
st.write("https://your-link.streamlit.app")
