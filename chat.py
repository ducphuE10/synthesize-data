from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="EMPTY")
model = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"

system_message = """
You are expert software engineer working at BigTech.
"""


def chat(messages):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **{
            "max_tokens": 512,
            "temperature": 0.5,
            "stop": ["<im_end>"],
        },
    )

    return resp.choices[0].message.content


def print_color(text, color):
    colors = {
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
    }
    assert color in colors.keys()
    print(f"{colors[color]}{text}\033[0m")


def main():
    messages = [
        {"role": "system", "content": system_message},
    ]
    your_text = ""
    while your_text != "exit":
        your_text = input("You: ")
        messages.append({"role": "user", "content": your_text})
        response = chat(messages)
        messages.append({"role": "assistant", "content": response})
        print_color("AI: " + response, "green")


if __name__ == "__main__":
    main()