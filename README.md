Simple Multi-Agent AI Framework to test the Deepseek 1.5B model

Overview

This project is a simple attempt at building a multi-agent AI system, designed as a hands-on experiment with the smallest DeepSeek model, a reasoning model (perfect for a decision making AI). The framework is lightweight and modular, allowing for easy integration of different agent functionalities, such as memory storage, LinkedIn and company data retrieval, and chatbot interactions.

Because the intended model has only 1.5B parameters, only the current prompt is passed to the LLM. The idea to overcome the small number of parameters was to have the MemoryMaster—an agent responsible for judging important information that should be kept for future use or retrieved for the current prompt, as illustrated below.
