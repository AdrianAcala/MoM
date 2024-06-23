"""
Enhanced OpenRouter Query Handler

This script provides a robust, class-based interface for interacting with the OpenRouter API,
allowing users to query multiple language models and synthesize their responses using the 
Mixture-of-Agents (MoA) methodology. The MoA approach, detailed in the paper "Mixture-of-Agents 
Enhances Large Language Model Capabilities" (https://arxiv.org/abs/2406.04692), leverages the 
collective strengths of multiple LLMs through a layered architecture.

In the MoA framework, each agent in a layer takes the outputs from agents in the previous layer 
as auxiliary information to generate more robust and comprehensive responses. This iterative 
refinement process enhances the overall response quality, achieving state-of-the-art performance 
on benchmarks such as AlpacaEval 2.0, MT-Bench, and FLASK.

Key Features:
- Robust class-based interface for OpenRouter API interaction.
- Query multiple language models and synthesize responses using MoA.
- Enhanced configuration management.

GitHub link for MoA: https://github.com/togethercomputer/MoA
"""

import logging
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OpenRouterConfig:
    """Configuration class for OpenRouter API."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    ENV_FILE = ".env"
    API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
    DEFAULT_MODEL = "qwen/qwen-110b-chat"
    DEFAULT_REFERENCE_MODELS = [
        "qwen/qwen-110b-chat",
        "qwen/qwen-72b-chat",
        "microsoft/wizardlm-2-8x22b",
        "meta-llama/llama-3-70b",
        "mistralai/mixtral-8x22b-instruct",
        "databricks/dbrx-instruct",
    ]
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TEMPERATURE = 0.7

    @classmethod
    def load_config(cls):
        """Load configuration from environment variables."""
        load_dotenv()
        cls.API_KEY = os.getenv(cls.API_KEY_ENV_VAR)
        cls.PRIMARY_MODEL = os.getenv("OPENROUTER_PRIMARY_MODEL", cls.DEFAULT_MODEL)

        # Parse the REFERENCE_MODELS from environment variable
        reference_models_str = os.getenv("OPENROUTER_REFERENCE_MODELS")
        if reference_models_str:
            # Split the string by commas and strip whitespace
            cls.REFERENCE_MODELS = [
                model.strip() for model in reference_models_str.split(",")
            ]
        else:
            cls.REFERENCE_MODELS = cls.DEFAULT_REFERENCE_MODELS

        cls.MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", cls.DEFAULT_MAX_TOKENS))
        cls.TEMPERATURE = float(
            os.getenv("OPENROUTER_TEMPERATURE", cls.DEFAULT_TEMPERATURE)
        )

    @classmethod
    def get_config_string(cls):
        """Get a string representation of the current configuration."""
        return f"""
Current Configuration:
----------------------
API URL: {cls.API_URL}
Primary Model: {cls.PRIMARY_MODEL}
Reference Models: {', '.join(cls.REFERENCE_MODELS)}
Max Tokens: {cls.MAX_TOKENS}
Temperature: {cls.TEMPERATURE}
        """


class EnvironmentManager:
    """Manages environment variables and .env file."""

    @staticmethod
    def update_env_file(key: str, value: str):
        """
        Update or add a key-value pair in the .env file.

        Args:
            key (str): The key to update or add.
            value (str): The value to set.
        """
        env_path = OpenRouterConfig.ENV_FILE
        if os.path.exists(env_path):
            with open(env_path, "r") as file:
                lines = file.readlines()

            updated = False
            for i, line in enumerate(lines):
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    updated = True
                    break

            if not updated:
                lines.append(f"{key}={value}\n")

            with open(env_path, "w") as file:
                file.writelines(lines)
        else:
            with open(env_path, "w") as file:
                file.write(f"{key}={value}\n")

        logger.info(f"Updated {key} in .env file.")

    @staticmethod
    def prompt_and_store(key: str, prompt: str) -> str:
        """
        Prompt the user for input and optionally store it in the .env file.

        Args:
            key (str): The key to store in the .env file.
            prompt (str): The prompt to display to the user.

        Returns:
            str: The user's input.
        """
        value = input(prompt)
        store = (
            input(
                "Would you like to store this value in the .env file for future use? (yes/no/y/n): "
            )
            .strip()
            .lower()
        )
        if store in ["y", "yes"]:
            EnvironmentManager.update_env_file(key, value)
        return value


class OpenRouterClient:
    """Client for interacting with the OpenRouter API."""

    def __init__(self):
        self.api_key = OpenRouterConfig.API_KEY
        if not self.api_key:
            self.api_key = EnvironmentManager.prompt_and_store(
                OpenRouterConfig.API_KEY_ENV_VAR,
                "API key not found. Please enter your OpenRouter API key: ",
            )
            OpenRouterConfig.load_config()  # Reload config after potential update

    def call_openrouter(
        self, model: str, messages: List[Dict[str, str]], params: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        Make an API call to OpenRouter.

        Args:
            model (str): The model to use for the API call.
            messages (List[Dict[str, str]]): The messages to send to the model.
            params (Dict[str, Any], optional): Additional parameters for the API call. Defaults to {}.

        Returns:
            Dict[str, Any]: The JSON response from the API.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": params.get("max_tokens", OpenRouterConfig.MAX_TOKENS),
            "temperature": params.get("temperature", OpenRouterConfig.TEMPERATURE),
            **params,
        }

        try:
            response = requests.post(
                OpenRouterConfig.API_URL, headers=headers, json=data
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API call failed: {e}")
            return {}


class QueryHandler:
    """Handles user queries and manages the response generation process."""

    def __init__(self, client: OpenRouterClient):
        self.client = client
        self.primary_model = OpenRouterConfig.PRIMARY_MODEL
        self.reference_models = OpenRouterConfig.REFERENCE_MODELS

    def generate_reference_responses(self, prompt: str) -> List[str]:
        """
        Generate responses from reference models.

        Args:
            prompt (str): The user's query.

        Returns:
            List[str]: List of responses from reference models.
        """
        reference_responses = []
        for model in self.reference_models:
            logger.info(f"Calling reference model: {model}")
            response = self.client.call_openrouter(
                model, [{"role": "user", "content": prompt}]
            )
            if response and "choices" in response:
                reference_responses.append(response["choices"][0]["message"]["content"])
            else:
                logger.warning(f"Failed to get response from model: {model}")
        return reference_responses

    def aggregate_responses(self, reference_responses: List[str]) -> str:
        """
        Aggregate responses using the primary model.

        Args:
            reference_responses (List[str]): List of responses from reference models.

        Returns:
            str: The aggregated response.
        """
        combined_responses = "\n".join(reference_responses)
        aggregation_prompt = {
            "role": "user",
            "content": f"Synthesize these responses into a single, high-quality response:\n\n{combined_responses}",
        }
        logger.info(f"Aggregating responses using primary model: {self.primary_model}")
        response = self.client.call_openrouter(self.primary_model, [aggregation_prompt])
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")

    def handle_query(self, query: str, rounds: int = 1) -> str:
        """
        Handle a user query and produce a synthesized response.

        Args:
            query (str): The user's query.
            rounds (int, optional): Number of rounds of querying and aggregation. Defaults to 1.

        Returns:
            str: The final synthesized response.
        """
        for _ in range(rounds):
            reference_responses = self.generate_reference_responses(query)
            query = self.aggregate_responses(reference_responses)
        return query


def main():
    """Main function to run the OpenRouter Query Handler."""
    OpenRouterConfig.load_config()
    client = OpenRouterClient()
    handler = QueryHandler(client)

    logger.info("Starting the OpenRouter Query Handler.")
    logger.info(OpenRouterConfig.get_config_string())
    logger.info("Type /bye or /exit to stop.")

    while True:
        query = input("Enter your query: ")
        if query.lower() in ["/bye", "/exit"]:
            logger.info("Exiting the program.")
            break

        logger.info(f"Received query: {query}")
        response = handler.handle_query(query)
        print("\nFinal Aggregated Response:")
        print(response)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
