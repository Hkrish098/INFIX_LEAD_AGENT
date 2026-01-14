import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from graph import graph   # your LangGraph workflow

from langchain_core.messages import trim_messages
# --------------------------------------------------
# 1. Setup
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="AutoStream Lead Agent",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 AutoStream Lead Agent")

# --------------------------------------------------
# 2. Session State (Chat History)
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(
            content="Hi 👋 I’m the AutoStream assistant. I can help you with pricing, plans, and getting started."
        )
    ]

# --------------------------------------------------
# 3. Render Chat History
# --------------------------------------------------

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"): 
            st.markdown(msg.content)
    else:
        with st.chat_message("user", avatar="👤"): 
            st.markdown(msg.content)

# --------------------------------------------------
# 4. Chat Input
# --------------------------------------------------

user_input = st.chat_input("Ask about AutoStream...")

if user_input:
    # 1. Add and display user message immediately
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 2. Invoke LangGraph
    with st.spinner("Thinking..."):
        # Stable thread_id allows the InMemorySaver to track state
        config = {"configurable": {"thread_id": "st_session_1"}}
        
        # Pass ONLY the latest message; LangGraph handles history internally via thread_id
        latest_user_message = st.session_state.messages[-1]

        # Invoke the graph
        response = graph.invoke(
            {"messages": [latest_user_message]}, 
            config=config
        )
        
        # 3. ROBUST EXTRACTION LOGIC
        # Extracts the last message from the updated state
        raw_content = response["messages"][-1].content
        
        # Handle block-based content format returned by newer Gemini models
        if isinstance(raw_content, list):
            ai_reply = "".join(
                block["text"] for block in raw_content 
                if isinstance(block, dict) and "text" in block
            )
        else:
            ai_reply = str(raw_content)

    # 4. Save the clean string to session state history to prevent UI "jumble"
    st.session_state.messages.append(AIMessage(content=ai_reply))

    # 5. Display the response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(ai_reply)

    # 6. Rerun to refresh the UI and maintain the conversation flow
    st.rerun()