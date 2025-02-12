import requests
import json

# vLLM API endpoint
API_URL = "http://localhost:8000/v1/completions"

from AgentsClass import MasterAgent
from AgentsClass import MemoryMaster
from AgentsClass import LinkedInConnector
from AgentsClass import LinkedInSearcher
from AgentsClass import CompanySearcher
from AgentsClass import NewsSearcher
from AgentsClass import ChatBot

AGENTS_NAMES =  {
                "LinkedIn Connector": "Specific for crafting messages used in LinkedIn connections",
                "LinkedIn Searcher": "Expert in getting information of a person's LinkedIn",
                "Company Searcher": "Expert in getting precise and audited information of a company from the internet",
                "News Searcher": "Expert in getting the latest news from the internet that might be relevant to the user",
                "Chat Bot": "Handles general questions when other agents can't. Perfect for small talk and general questions."
                }

AGENT_REGISTRY = {
                  "LinkedIn Connector": LinkedInConnector(),
                  "LinkedIn Searcher": LinkedInSearcher(),
                  "Company Searcher": CompanySearcher(),
                  "News Searcher": NewsSearcher(),
                  "Chat Bot": ChatBot()
                  }

# Agent Master and Memory Master instances initialize to answer the user

Agent_Master = MasterAgent(AGENTS_NAMES, API_URL)

Memory_Master = MemoryMaster(API_URL)

print("Welcome to the multitask AI assistant! What can I help you with today?")

while True:
    # Initial prompt for the user
    prompt = input("User: ")

    # First, the Memory Master will process the user input to determine if there is any memory to retrieve and update the memory of the agents
    retrieved_memories = Memory_Master.process(prompt)
    
    Agent_Master.update_agent_memory(retrieved_memories)

    # The Agent Master will process the user input to determine which agent to call
    agent_selection = Agent_Master.process(prompt)

    print(f"Agent Selected: {agent_selection}")

    # The selected agent will be called to process the user input
    Answer_Agent = AGENT_REGISTRY[agent_selection]

    # The Memory Master will update the selected agent with the memory it has considered relevant
    Answer_Agent.update_agent_memory(retrieved_memories)

    # The selected agent will process the user input and the answer will be returned to the user
    response = Answer_Agent.process(prompt)

    print(f'AI: {response}')

    # The Memory Master will reflect on the occured exchange and store any relevant information
    Memory_Master.reflect_and_store(prompt, response)