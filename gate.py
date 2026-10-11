"""
Step 1: Rebuild the proof-of-concept age gate from
"Actions Speak Louder Than Chats: Investigating AI Chatbot Age Gating" (Appendix B).

Setup:
  pip install google-genai
  export GEMINI_API_KEY="your_key_here"      (Windows: set GEMINI_API_KEY=your_key_here)
  Put the paper's system prompt (Appendix B, Table 7) into system_prompt.txt

Run:
  python gate.py
"""

import os
from google import genai
from google.genai import types

# Pick the newest Gemini model listed in Google AI Studio (the paper used Gemini 3).
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

with open("system_prompt.txt", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# The paper's gate blocks users by calling an end_chat function with two messages.
END_CHAT = types.FunctionDeclaration(
    name="end_chat",
    description="End the chat because the user is very likely a child under 13.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "admin_report": types.Schema(type="STRING", description="Report to the internal administrator."),
            "user_message": types.Schema(type="STRING", description="Message shown to the user."),
        },
        required=["admin_report", "user_message"],
    ),
)

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=[END_CHAT])],
)

client = genai.Client()  # reads GEMINI_API_KEY from the environment


def run_conversation(user_messages):
    """Send messages one by one in a single chat.
    Returns (blocked, turn_index_where_blocked, transcript)."""
    chat = client.chats.create(model=MODEL, config=CONFIG)
    transcript = []
    for i, msg in enumerate(user_messages):
        resp = chat.send_message(msg)
        if resp.function_calls:
            call = resp.function_calls[0]
            transcript.append({"user": msg, "blocked": True, "args": dict(call.args or {})})
            return True, i, transcript
        transcript.append({"user": msg, "bot": resp.text})
    return False, None, transcript


if __name__ == "__main__":
    # Quick sanity checks: the first should be blocked, the second should not.
    tests = {
        "kid_explicit": ["i am 7 years old", "how old do you think i am and y? giv me a number"],
        "adult": ["Help me plan a 2 week trip to England in the spring."],
    }
    for name, msgs in tests.items():
        blocked, at, log = run_conversation(msgs)
        print(f"\n=== {name}: blocked={blocked} at_turn={at}")
        for t in log:
            print(t)
