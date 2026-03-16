import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from tools.forecast_tool import forecast_tool
from tools.retriever_tool import retriever_tool


# Load Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # safer stable model
    temperature=0.3
)

tools = [forecast_tool, retriever_tool]

agent = create_agent(
    model=llm,
    tools=tools
)


def run_agent(query: str):
    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
    )

    return response["messages"][-1].content

if __name__ == "__main__":
    print("Supply Chain AI Agent Started 🚀")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter your query: ")

        if user_input.lower() == "exit":
            print("Exiting agent...")
            break

        try:
            result = run_agent(user_input)
            print("\nAI Response:\n")
            print(result)
            print("\n" + "-"*50 + "\n")
        except Exception as e:
            print("Error:", str(e))