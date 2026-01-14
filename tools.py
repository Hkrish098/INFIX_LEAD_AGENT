import json
import os
from langchain_core.tools import tool

@tool
def mock_lead_capture(name: str, email: str, platform: str):
    """
    Saves the collected lead information into a local JSON file.
    """
    lead_entry = {
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": "2026-01-12"
    }

    file_path = "leads.json"
    
    # Read existing data
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append the new lead
    data.append(lead_entry)
    
    # FIX: Ensure positional arguments (f) come BEFORE keyword arguments (indent=4)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4) # Fixed Syntax: data first, then file, then indent

    print(f"\n[STORAGE SUCCESS] Lead saved to JSON: {name}, {email}, {platform}\n")
    return f"Lead for {name} has been successfully saved to {file_path}."