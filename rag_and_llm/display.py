import pickle

def print_all_message_contents(messages):
    """Print all message contents in order, with sender name if available."""
    for i, msg in enumerate(messages, start=1):
        sender = getattr(msg, "name", None)
        msg_type = type(msg).__name__
        content = getattr(msg, "content", "")

        print(f"\n[{i}] {msg_type}{' (' + sender + ')' if sender else ''}:")
        print(content or "(empty)")
        
with open("messages.pkl", "rb") as f:
    loaded_messages = pickle.load(f)

print_all_message_contents(loaded_messages)

