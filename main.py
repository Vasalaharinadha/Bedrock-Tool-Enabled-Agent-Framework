import boto3
from langchain_aws import ChatBedrock
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool


MODEL_ID = "XXXXXXXXXXXXXX"
REGION = "XXXXXXXXXXXX"



bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION)

def add_numbers(a: float, b: float) -> float:
    return a + b

add_tool = StructuredTool.from_function(
    func=add_numbers,
    name="addition_tool",
    description="Adds two numbers")

def divide_numbers(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b
divide_tool = StructuredTool.from_function(
    func=divide_numbers,
    name="division_tool",
    description="Divides number a by number b. b must not be zero.")


llm = ChatBedrock(
    client=bedrock_client,
    model_id=MODEL_ID,
    model_kwargs={
        "temperature": 0.3,
        "max_tokens": 1024
    }
)



agent = create_agent(
    model=llm,
    tools=[add_tool, divide_tool]   
)

print("Agent created successfully ")


response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the  sum of 5 and 3 ?"}
    ]
})

print("\nResponse:\n")
print(response["messages"][-1].content)
