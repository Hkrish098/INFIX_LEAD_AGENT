def detect_intent(text: str) -> str:
    t = text.lower()

    if any(x in t for x in ["hi", "hello", "hey"]):
        return "greeting"

    if any(x in t for x in ["price", "pricing", "cost", "plan", "features"]):
        return "product"

    if any(x in t for x in ["sign up", "subscribe", "try", "interested", "get started"]):
        return "high_intent"

    return "general"
