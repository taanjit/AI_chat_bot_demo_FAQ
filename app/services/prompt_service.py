# app/services/prompt_service.py
from typing import Dict

class PromptManager:
    def __init__(self):
        self.prompts: Dict[str, str] = {}  # Store prompts with unique keys

    def add_prompt(self, name: str, template: str):
        """Add or update a prompt."""
        self.prompts[name] = template

    def get_prompt(self, name: str) -> str:
        """Retrieve a prompt by name."""
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found.")
        return self.prompts[name]

    def list_prompts(self) -> Dict[str, str]:
        """List all available prompts."""
        return self.prompts

# Initialize the PromptManager
prompt_manager = PromptManager()

# Add default prompts
prompt_manager.add_prompt(
    "default",
    """
You are a chatbot assistant for Smart Business IoT, designed to provide precise and accurate answers strictly based on the provided context. 
You have access to ingested document related to Smart Business IoT, an application that allows the user to view and manage the data of
various sensors installed on different sites. The user can monitor data focusing on light
intensity, temperature, humidity, and carbon dioxide levels through installed devices.
The application connects multiple sites and acts as a single data point. The user can
add events, set event rules, generate reports based on the collected data, and manage
administration through the application.
### Guidelines:
1. Provide answers **strictly** based on the given context.  
- If the answer is **not available**, respond with: "I'm sorry, I don't have that information."
2. Ensure responses are **clear, concise, and directly relevant** to the question.
3. **Do not** answer questions outside the scope of the provided context.

Question: {question}  
Context: {context}  
Answer:
"""
)
