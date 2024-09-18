def fmt_openAI_to_shareGPT(messages):
    conversations = []
    for msg in messages:
        if msg["role"] == "user":
            from_ = "human"
        elif msg["role"] == "assistant":
            from_ = "gpt"
        elif msg["role"] == "system":
            from_ = "system"
        else:
            raise ValueError(f"Invalid message role: {msg['role']}")
            
        conversations.append({"from": from_, "value": msg["content"]})
        
    return conversations


def fmt_shareGPT_to_openAI(conversations):
    messages = []
    for msg in conversations:
        if msg["from"] == "human":
            role = "user"
        elif msg["from"] == "gpt":
            role = "assistant"
        elif msg["from"] == "system":
            role = "system"
        else:
            raise ValueError(f"Invalid message from: {msg['from']}")
        
        messages.append({"role": role, "content": msg["value"]})
    
    return messages


def validate_openAI(messages):
    current_role = messages[0]["role"]
    if current_role not in {"user", "system"}:
        raise ValueError(f"Invalid first message role: {current_role}")
    
    for i in range(1, len(messages)):
        if messages[i]["role"] == current_role:
            raise ValueError(f"Two consecutive messages with the same role: {current_role} - {messages}")
        
        current_role = messages[i]["role"]
        
        if messages[i]["role"] not in {"user", "assistant"}:
            raise ValueError(f"Invalid message role: {messages[i]['role']}")
        
        if not messages[i]["content"].strip():
            raise ValueError(f"Empty message content")


def validate_shareGPT(conversations):
    messages = fmt_shareGPT_to_openAI(conversations)
    validate_openAI(messages)