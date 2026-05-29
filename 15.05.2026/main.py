import os
import sys
from groq import Groq

def print_banner():
    banner = """
 ██████╗ ██████╗  ██████╗  ██████╗ 
██╔════╝ ██╔══██╗██╔═══██╗██╔═══██╗
██║  ███╗██████╔╝██║   ██║██║   ██║
██║   ██║██╔══██╗██║   ██║██║   ██║
╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝
 ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ 
    """
    print(banner)
    print("┌────────────────────────────────────────────────────────┐")
    print("│  System: Connected to Groq API Cluster                 │")
    print("│  Controls: Type 'exit' or 'quit' to terminate session  │")
    print("└────────────────────────────────────────────────────────┘\n")

def main():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    history = []
    
    print_banner()
    
    while True:
        print("─" * 60)
        user_input = input(" Prompt ❯ ")
        print("─" * 60)
        
        if user_input.lower() in ["exit", "quit"]:
            print("\n┌────────────────────────────────────────────────────────┐")
            print("│  Session terminated. Goodbye!                          │")
            print("└────────────────────────────────────────────────────────┘")
            break
            
        if not user_input.strip():
            continue
            
        history.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=history,
                temperature=0.7,
                max_tokens=1024
            )
            
            bot_response = response.choices[0].message.content
            history.append({"role": "assistant", "content": bot_response})
            
            print("\n┌─── Groq Response ───────────────────────────────────────")
            for line in bot_response.splitlines():
                print(f"│ {line}")
            print("└─────────────────────────────────────────────────────────\n")
            
        except Exception as e:
            print(f"\n┌─── Error ───────────────────────────────────────────────")
            print(f"│ {e}")
            print("└─────────────────────────────────────────────────────────\n")

if __name__ == "__main__":
    main()