import requests
import json

# vLLM API endpoint
API_URL = "http://localhost:8000/v1/completions"

class Agent:
    """
    Base class for all agents.
    """
    def __init__(self, name: str, description: str, api_url: str):
        self.name = name
        self.description = description
        self.api_url = api_url
        self.base_prompt = "I am an assistant AI called R.O.B that can help you with a variety of tasks. Give correct answers about the questions asked and follow given instructions."
        self.think_prompt = None
        self.answer_prompt = None
        self.memory = {}
        
    def update_agent_memory(self, dict_memory):
        """
        Updates the agent's memory with the provided dictionary.
        Built to update the agent's memory with the MemoryMaster selected memories.
        
        Args:
            dict_memory (dict): MemoryMaster.memory
            
        """
        if isinstance(dict_memory, dict):
            self.memory = dict_memory

    def send_payload_to_llm(self, payload: dict) -> list:
        """Sends the payload to the LLM API and returns the response text as a list of strings."""
        response = requests.post(self.api_url, json=payload)
        
        return [choice.get("text", "").strip() for choice in response.json().get("choices")]

    @staticmethod
    def payload_giver(prompt: str, stop: str, max_tokens: int = 3000, temperature: float = 0.5, top_p: float = 0.95,
                      best_of: int = 1, n: int = 1) -> dict:
        """Generates the payload for the API call."""
        stop_mapping = {"think": "</think>", "answer": "</answer>"}
        stop_sequence = stop_mapping.get(stop, stop)
        
        if stop_sequence is None:
            raise ValueError(f"Invalid stop sequence: {stop}")
        if best_of < 1 or n < 1 or n > best_of:
            raise ValueError(f"Invalid best_of or n values: {best_of}, {n}")
        
        return {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "best_of": best_of,
            "n": n,
            "stop": [stop_sequence]
        }

    def think_prompt_maker(self, *args, **kwargs) -> str:
        """To be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def answer_prompt_maker(self, *args, **kwargs) -> str:
        """To be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def process(self, user_prompt: str) -> str:
        """To be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class MasterAgent(Agent):
    def __init__(self, agents_names_dic: dict, api_url: str):
        super().__init__(name="Master Agent", description="Selects the best agent for a task.", api_url=api_url)
        self.agents_names_dic = agents_names_dic

    def think_prompt_maker(self, prompt: str) -> str:
        """Generates the thinking prompt for agent selection."""
        agents_list_str = '\n'.join(
            [f"Agent {agent_name}: {agent_description}" for agent_name, agent_description in self.agents_names_dic.items()]
        )

        memories_str = '\n'.join([f"{key}: {value}" for key, value in self.memory.items()])

        return (
            f"{self.base_prompt}\n"
            f"My job is to choose the right agent to best respond to the user given their prompt.\n"
            f"These are some important informations in your memory: {memories_str}\n"
            f"Here are the names of the agents at my disposal and their specialties:\n{agents_list_str}\n"
            f"This is the prompt given by the user: {prompt}\n"
            f"<think> Does the prompt of the user seems to align with which agent speciality?\n"
            f"Let me think for a moment..."
        )

    def answer_prompt_maker(self, thinking_prompt: str, thinking_answer: str) -> str:
        """Constructs the answer prompt based on the thinking prompt and response."""
        final_prompt = f"{thinking_prompt}\n{thinking_answer}\n</think>"
        agents_list_str = '\n'.join(
            [f"Agent {agent_name}\n" for agent_name, agent_description in self.agents_names_dic.items()]
        )
        return (
            f"{final_prompt}\n"
            f"Based in what I thought, I will now call the agent that best suits the user need.\n"
            f"The answer will be only the name of the agent and nothing else.\n"
            f"<answer>Agent "
        )

    def process(self, user_prompt: str, is_print: bool = False,
                n_thinking: int = 1, best_of_thinking: int = 1,
                n_answer: int = 1, best_of_answer: int = 1) -> str:
        """Determines and calls the appropriate agent based on the user input."""
        # Step 1: Generate thinking prompt
        thinking_prompt = self.think_prompt_maker(user_prompt)

        # Step 2: Send the thinking prompt to the LLM
        thinking_payload = self.payload_giver(thinking_prompt, "think", n = n_thinking, best_of = best_of_thinking)
        thinking_response = self.send_payload_to_llm(thinking_payload)[0]

        # Step 3: Generate answer prompt based on thinking response
        answer_prompt = self.answer_prompt_maker(thinking_prompt, thinking_response)

        # Step 4: Send the answer prompt to the LLM
        answer_payload = self.payload_giver(answer_prompt, "answer", n=n_answer, best_of=best_of_answer)
        
        selected_agent_name = self.send_payload_to_llm(answer_payload)
        
        if n_answer > 1:
            return selected_agent_name
        else:
            selected_agent_name = selected_agent_name[0]
        
            if is_print:
                print(f'Master deepthink: {answer_prompt}{selected_agent_name}')

            # Step 5: Call the selected agent. If not in the list, call the ChatBot
            if selected_agent_name in self.agents_names_dic:
                print(f"Agent {selected_agent_name} is not implemented yet")
                # return selected_agent_name
                return "Chat Bot" # For now, return the Chat Bot
            else:
                return "Chat Bot"

class MemoryMaster(Agent):
    def __init__(self, api_url: str):
        super().__init__(name="Memory Master", description="Stores and retrieves information from memory.", api_url=api_url)
        self.memory =   {
                        'MyIdentity': 'I am an AI assistant called R.O.B that can help the user with a variety of tasks.',
                        }

    def store_memory(self, key: str, value: str):
        """Stores or updates a key-value pair in memory."""
        if key in self.memory:
            self.memory[key] = value
            return f"Memory updated: {key} -> {value}"
        else:
            self.memory[key] = value
            return f"Memory stored: {key} -> {value}"

    def retrieve_memory(self, key: str) -> str:
        """Retrieves a value from memory based on the given key."""
        return self.memory.get(key, "No memory found for this key.")

    def think_prompt_maker(self, user_prompt: str) -> str:
        """Constructs a prompt to autonomously decide if memory should be retrieved based on existing keys."""
        keys_list = ', '.join(self.memory.keys()) if self.memory else "No stored keys"
        return (
            f"{self.base_prompt}\n"
            f"I am responsible for the memory management. I need to understand the context of the conversation, the content of my memory and decide if there is important information to be retrieved from my memory to better answer the user.\n"
            f"The memory is kept in a simple key-value format.\n"
            f"Stored memory keys: {keys_list}\n"
            f"User input: {user_prompt}\n"
            f"<think> Based on the available memory keys and in the user prompt, should I retrieve information? Let me think...\n"
        )

    def answer_prompt_maker(self, thinking_prompt: str, thinking_answer: str) -> str:
        """Constructs the final prompt based on the thinking process with better instructions on key extraction."""
        return (
            f"{thinking_prompt}\n{thinking_answer}\n</think>\n"
            f"If I decide to access the memory, I'm instructed to extract the most relevant key(s) from the user input that match the stored keys.\n"
            f"The output should be only the key(s), comma-separated if multiple, and nothing else.\n"
            f"If the action is 'no', output 'none'.\n"
            f"<answer> "
        )

    def process(self, user_prompt: str, is_print: bool = False) -> str:
        """Processes user input to autonomously determine whether to retrieve memory or not, handling multiple keys safely."""
        self.last_interaction = user_prompt
        
        try:
            # Step 1: Generate thinking prompt
            thinking_prompt = self.think_prompt_maker(user_prompt)

            # Step 2: Send the thinking prompt to the LLM
            thinking_payload = self.payload_giver(thinking_prompt, "think", n=1, best_of=1)
            thinking_response = self.send_payload_to_llm(thinking_payload)[0].strip().lower()

            # Step 3: Generate answer prompt
            answer_prompt = self.answer_prompt_maker(thinking_prompt, thinking_response)
            answer_payload = self.payload_giver(answer_prompt, "answer", n=1, best_of=1)
            answer_response = self.send_payload_to_llm(answer_payload)[0].strip()

            if is_print:
                print(f"Memory process thinking: {answer_prompt}")
                print(f"Memory process answer: {answer_response}")

            keys = [key.strip() for key in answer_response.split(",") if key.strip() in self.memory]
            if keys:
                # return " | ".join(f"{key}: {self.retrieve_memory(key)}" for key in keys)
                return {key: self.retrieve_memory(key) for key in keys}

        except Exception as e:
            # Log the error and continue safely
            print(f"Error in process method: {e}")

        return "No memory retrieval necessary."

    def reflect_and_store(self, user_input: str, agent_response: str, is_print: bool = False) -> str:
        """Reflects on the last exchange between the user and AI to determine if new key information should be stored or updated."""
        
        try:
            # Step 1: Generate reflection prompt
            reflection_think_prompt = (
                f"{self.base_prompt}\n"
                f"I am responsible for remembering important details from my conversation with the user.\n"
                f"I need to check what the user just said and decide if there is something important that I should remember.\n"
                f"\nConversation:\nThe user said: {user_input}\nThen I've said: {agent_response}\n"
                f"\nCurrently, I remember these topics: {', '.join(self.memory.keys()) if self.memory else 'I don’t remember anything yet'}\n"
                f"<think> Based on this conversation, is there something that I should keep in my memory? I shoud focus first in what the user said. Let me think..."
            )

            # print(f"Reflection think prompt: {reflection_think_prompt}")

            # Step 2: Ask LLM for reflection
            reflection_payload = self.payload_giver(reflection_think_prompt, "think", n=1, best_of=1)
            reflection_response = self.send_payload_to_llm(reflection_payload)[0].strip().lower()
            
            reflection_answer_prompt = (
                f"{reflection_think_prompt}{reflection_response}\n"
                f"If I decide to remember something, I need to store it in a simple way:\n"
                f"a single word that encapsulate the new memory should go first.\n"
                f"Then, a short description of what I need to remember.\n"
                f"If there is nothing to remember, I should just say 'none'.\n"
                f"\nFormat example for the answer:\n"
                f"<single word> : <information>'\n</think>"
                f"I will now say the single word, followed by : and then information that I should remember or 'none', respecting the format instructed.\n"
                f"<answer> "
            )
            
            
            reflection_answer_payload = self.payload_giver(reflection_answer_prompt, "answer", n=1, best_of=1)
            reflection_response = self.send_payload_to_llm(reflection_answer_payload)[0].strip()
            
            if is_print:
                print(f"Internal Thinking process: {reflection_answer_prompt}")
                print(f"Memory reflection final decision: {reflection_response}")

            # Validate reflection response
            if reflection_response == "none":
                return "No new information to store."

            if ":" not in reflection_response:
                return "Invalid response from LLM, ignoring storage."

            # Step 3: Extract key-value pair
            key, value = reflection_response.split(":", 1)
            key, value = key.strip(), value.strip()

            # Validate extracted key and value
            if not key or not value or " " in key:  # Ensures key is a **single word**
                return "Invalid key-value format from LLM, ignoring storage."

            # Step 4: Store memory safely
            return self.store_memory(key, value)

        except Exception as e:
            print(f"Error in reflect_and_store method: {e}")
            return "No new information to store."

class LinkedInConnector(Agent):
    """
    LinkedIn Connector Agent specific for creating connecting messages for the user to send to the given person.
    It will activate the LinkedIn Searcher to get information about the person and the user, to better contextualize the message.
    """
    
    def __init__(self):
        super().__init__("LinkedIn Connector", "Expert in creating connecting messages for the user to send to the given person", API_URL)

    def process(self, user_prompt: str) -> str:
        return f"LinkedIn Connector is not implemented yet"

class LinkedInSearcher(Agent):
    """
    LinkedIn Searcher Agent specific for getting information of a person's LinkedIn.
    Because of expected site structure, normal web scraping would probably sufise. But I don't know if it's legal to do so.
    The alternative then the LinkedIn API.
    """
    
    def __init__(self):
        super().__init__("LinkedIn Searcher", "Expert in getting information of a person's LinkedIn", API_URL)

    def process(self, user_prompt: str) -> str:
        return f"LinkedIn Searcher is not implemented yet"
    
class CompanySearcher(Agent):
    """
    Company Searcher Agent specific for getting information of a company.
    Because the websites of companies can be very different, this would be a great opportunity to implement crawl4ai + knowledge graph to do the web scraping.
    To minimize the tokens, maybe just do a simple web scraping in wikipedia then see if it has the information we need and call the big guns if it doesn't.
    """
     
    def __init__(self):
        super().__init__("Company Searcher", "Expert in getting precise and audited information of a company", API_URL)

    def process(self, user_prompt: str) -> str:
        return f"Company Searcher is not implemented yet"
    
class NewsSearcher(Agent):
    """
    There are some news APIs that would be great to use but also the crawl4ai + knowledge graph would be a great approach here.
    Normal web scraping too if the goal is to go economy in the tokens.
    """
    
    def __init__(self):
        super().__init__("News Searcher", "Expert in getting the latest news from the internet that might be relevant to the user", API_URL)

    def process(self, user_prompt: str) -> str:
        return f"News Searcher is not implemented yet"

class ChatBot(Agent):
    """
    Just the simple chatbot.
    """
    def __init__(self):
        super().__init__("Chat Bot", "Handles general questions when other agents can't", API_URL)

    def process(self, user_prompt: str) -> str:
        # Generate the prompt for the LLM
        
        memories_str = '\n'.join([f"{key}: {value}" for key, value in self.memory.items()])
        
        prompt = (
            f"{self.base_prompt}\n"
            f"I am a AI assistant that can help you with general questions and also have conversations with the user.\n"
            f"I must maintain a friendly and professional tone while keeping answers clear and informative.\n"
            f"This is some informations in your memory: {memories_str}\n"
            f"User message: {user_prompt}\n"
            f"What should I respond to the user?<think>\n"
        )

        # Send the prompt to the LLM
        payload = self.payload_giver(prompt, "think", n=1, best_of=1)
        response = self.send_payload_to_llm(payload)[0]
        
        answer_prompt = (
            f"{prompt}\n"
            f"I better now say the best anwser to the user that I can do.\n"
            f"<answer>"
        )    
        
        payload = self.payload_giver(answer_prompt, "answer", n=1, best_of=1)
        response = self.send_payload_to_llm(payload)[0]        

        return response
