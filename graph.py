import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

# Import custom modules
from state import AgentState
from tools import mock_lead_capture

# --- CONFIGURATION ---
VECTOR_DB_PATH = "./vector_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

# --- LLM INITIALIZATION ---
def get_llm():
    """Configures LLM with a token limit to bypass the 402 credit error."""
    return ChatOpenAI(
        model="google/gemini-2.5-flash", 
        api_key=os.getenv("GOOGLE_API_KEY"), 
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=512,
        default_headers={
            "HTTP-Referer": "http://localhost:3000", 
            "X-Title": "AutoStream Lead Agent"
        }
    )

# --- NODES ---
def router_node(state: AgentState):
    """Refined classification to distinguish between advice and signup."""
    lead = state.get("lead_data") or {"name": None, "email": None, "platform": None}
    
   # FIX: Only force high_intent if we are MID-FLOW (name or email is missing)
    # If platform is already filled, the flow is DONE. Let the LLM decide the next intent.
    if lead.get("name") and not lead.get("platform"):
        if lead.get("name") not in ["COLLECTING_NAME"]:
            return {"intent": "high_intent"}

    llm = get_llm()
    last_message = state["messages"][-1].content
    
    prompt = (
        "Classify the user's intent strictly into one of these three categories. "
        "If the user provides a name or personal details, cross-reference the conversation history "
        "to determine if they are responding to a request for information to join or sign up:\n\n"
        "1. greeting: Simple hellos or casual opening remarks.\n"
        "2. product: Inquiries about prices, features, budget constraints (e.g., 'I have $40'), or requests for recommendations.\n"
        "3. high_intent: Explicit statements of wanting to join, sign up, or start now, OR providing requested personal details (like a name) to complete a registration flow.\n\n"
        "IMPORTANT: If the user asks for a recommendation or mentions a budget without an explicit desire to start the signup process immediately, classify it as product.\n"
        "Return ONLY the category name."
    )
    
    response = llm.invoke([("system", prompt), ("human", last_message)])
    intent = response.content.strip().lower()
    
    valid_intents = ["greeting", "product", "high_intent"]
    return {"intent": intent if intent in valid_intents else "greeting"}

def greeting_node(state: AgentState):
    """Simple friendly greeting."""
    msg = "Hi! 👋 I'm your AutoStream assistant. I can help with pricing or getting you started. What's on your mind?"
    return {"messages": [AIMessage(content=msg)]}

# ---- RAG NODE ----
def rag_node(state: AgentState):
    """Answers product questions, provides recommendations, and prevents echoing."""
    llm = get_llm()
    user_query = state["messages"][-1].content
    docs = vectorstore.similarity_search(user_query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    system_prompt = (
        "You are the AutoStream Sales Expert. Provide clean, professional, and helpful advice.\n"
        "STRICT RULES:\n"
        "1. NEVER repeat the user's question back to them. Provide only the answer.\n"
        "2. Use standard spaces. Never merge words like '29/month'—use ' $29 / month '.\n"
        "3. Bold prices (e.g., **$29**) and plan names (e.g., **Pro Plan**).\n"
        "4. Use bullet points for feature lists.\n\n"
        "RECOMMENDATION LOGIC:\n"
        "1. If a user describes a situation (e.g., 'beginner', 'budget', '4K'), analyze "
        "the features in context and suggest the best match.\n"
        "2. Recommend **Pro Plan** (**$79**) for 4K or unlimited needs.\n"
        "3. Recommend **Basic Plan** (**$29**) for budget or beginners.\n"
        f"CONTEXT:\n{context}"
    )
    
    # We pass the query in a task-oriented human message to break the echo loop
    response = llm.invoke([
        ("system", system_prompt), 
        ("human", f"Based on the context, provide a direct answer or recommendation for: {user_query}")
    ])
    return {"messages": [AIMessage(content=response.content)]}

def lead_capture_node(state: AgentState):
    """Sequentially collects and stores lead info in the state."""
    lead = state.get("lead_data") or {"name": None, "email": None, "platform": None}
    user_input = state["messages"][-1].content

    # Step A: Capture Name
    if not lead["name"] or lead["name"] == "COLLECTING_NAME":
        if not lead["name"]:
            # Set marker so router knows to stay in high_intent
            lead["name"] = "COLLECTING_NAME" 
            return {
                "messages": [AIMessage(content="I'd love to help you get started! What is your name?")],
                "lead_data": lead
            }
        
        # User provided their name
        lead["name"] = user_input
        return {
            "messages": [AIMessage(content=f"Thanks {lead['name']}! What is your email address?")],
            "lead_data": lead
        }
    
    # Step B: Capture Email
    if not lead["email"]:
        lead["email"] = user_input
        return {
            "messages": [AIMessage(content="Got it. Which platform do you use (YouTube, TikTok, etc.)?")],
            "lead_data": lead
        }

    # Step C: Capture Platform and Finalize
    # --- STEP C FIX in lead_capture_node ---
    # Step C: Capture Platform and Finalize
    if not lead["platform"]:
        lead["platform"] = user_input
        
        # Friendly final message
        final_msg = (
            f"All set, {lead['name']}! 🥳 I've saved your details. "
            "Our team will reach out to you shortly. Is there anything else? 👋"
        )
        
        # 1. Manually save the data to the JSON file
        from tools import mock_lead_capture
        mock_lead_capture.invoke({"name": lead["name"], "email": lead["email"], "platform": lead["platform"]})
        
        # 2. Return the message and MARK the lead as finished in state
        return {
            "messages": [AIMessage(content=final_msg)],
            "lead_data": lead
        }

    return {"lead_data": lead}

# --- GRAPH BUILDER ---

builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("router", router_node)
builder.add_node("greeting", greeting_node)
builder.add_node("rag", rag_node)
builder.add_node("lead_capture", lead_capture_node)
builder.add_node("tools", ToolNode([mock_lead_capture]))

# Define Edges
builder.add_edge(START, "router")

# Router logic: Routes based on identified intent
builder.add_conditional_edges(
    "router",
    lambda state: state["intent"], 
    {
        "greeting": "greeting",
        "product": "rag",
        "high_intent": "lead_capture"
    }
)

builder.add_edge("greeting", END)
builder.add_edge("rag", END)

# Lead Capture exit: wait for next info or execute tool
builder.add_conditional_edges(
    "lead_capture", 
    tools_condition, 
    {
        "tools": "tools",   
        "__end__": END      
    }
)

builder.add_edge("tools", END)

# Persistence
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)