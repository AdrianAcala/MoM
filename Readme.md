# Enhanced OpenRouter Query Handler
## Overview
This project provides a robust, class-based interface for interacting with the OpenRouter API, allowing users to query multiple language models and synthesize their responses using the Mixture-of-Agents (MoA) methodology. The MoA approach, detailed in the paper "Mixture-of-Agents Enhances Large Language Model Capabilities," leverages the collective strengths of multiple LLMs through a layered architecture.

In the MoA framework, each agent in a layer takes the outputs from agents in the previous layer as auxiliary information to generate more robust and comprehensive responses. This iterative refinement process enhances the overall response quality, achieving state-of-the-art performance on benchmarks such as AlpacaEval 2.0, MT-Bench, and FLASK.

### Key Features
- Robust class-based interface for OpenRouter API interaction.
- Query multiple language models and synthesize responses using the Mixture-of-Agents (MoA) methodology.
- Enhanced configurability for customizing agent behavior and response synthesis.
- Layered architecture for iterative refinement of responses.

## Installation
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Configuration
The configuration for the OpenRouterHandler is managed through environment variables stored in a .env file. Here are the key configuration options:

- **OPENROUTER_API_KEY**: Your OpenRouter API key.
- **OPENROUTER_PRIMARY_MODEL**: The primary model to use (default: qwen/qwen-110b-chat).
- **OPENROUTER_REFERENCE_MODELS**: A comma-separated list of reference models.
- **OPENROUTER_MAX_TOKENS**: The maximum number of tokens to use (default: 2048).
- **OPENROUTER_TEMPERATURE**: The temperature for response generation (default: 0.7).

Create a .env file in the root directory of your project and add your configuration:

```properties
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_PRIMARY_MODEL=qwen/qwen-110b-chat
OPENROUTER_REFERENCE_MODELS=qwen/qwen-110b-chat,qwen/qwen-72b-chat,microsoft/wizardlm-2-8x22b,meta-llama/llama-3-70b,mistralai/mixtral-8x22b-instruct,databricks/dbrx-instruct
OPENROUTER_MAX_TOKENS=2048
OPENROUTER_TEMPERATURE=0.7
```

## Usage
### Basic Example
To get started, simply run the script without any parameters. The script will use the default configurations specified in the .env file. If the .env file is not set up, it will prompt you to enter your OpenRouter API key and then ask if you want to save it for future use:

```bash
python bot.py
```

### Advanced Usage
You can also set each parameter at startup using command-line arguments. Here are the available command-line arguments:

- `--primary_model`: The primary model to use.
- `--reference_models`: Comma-separated list of reference models.
- `--max_tokens`: The maximum number of tokens.
- `--temperature`: The temperature for response generation.

#### Example

```bash
python bot.py \
    --primary_model "qwen/qwen-110b-chat" \
    --reference_models "qwen/qwen-72b-chat,microsoft/wizardlm-2-8x22b" \
    --max_tokens 2048 \
    --temperature 0.7
```

## Interactive Example
The script is designed for interactive use. After starting the script, you can enter your query directly:

```bash
$ python bot.py
API key not found. Please enter your OpenRouter API key: sk-or-v1-REDACTED
Would you like to store this value in the .env file for future use? (yes/no/y/n): y
2024-06-23 12:51:40,074 - INFO - Updated OPENROUTER_API_KEY in .env file.
2024-06-23 12:51:40,076 - INFO - Starting the OpenRouter Query Handler.
2024-06-23 12:51:40,076 - INFO - 
Current Configuration:
----------------------
API URL: https://openrouter.ai/api/v1/chat/completions
Primary Model: qwen/qwen-110b-chat
Reference Models: qwen/qwen-110b-chat, qwen/qwen-72b-chat, microsoft/wizardlm-2-8x22b, meta-llama/llama-3-70b, mistralai/mixtral-8x22b-instruct, databricks/dbrx-instruct
Max Tokens: 2048
Temperature: 0.7
        
2024-06-23 12:51:40,076 - INFO - Type /bye or /exit to stop.
Enter your query: There are three killers in a room. Someone enters the room and kills one of them. Nobody leaves the room. How many killers are left in the room?
2024-06-23 12:55:02,445 - INFO - Received query: There are three killers in a room. Someone enters the room and kills one of them. Nobody leaves the room. How many killers are left in the room?
2024-06-23 12:55:02,447 - INFO - Calling reference model: qwen/qwen-110b-chat
2024-06-23 12:55:06,146 - INFO - Calling reference model: qwen/qwen-72b-chat
2024-06-23 12:55:07,099 - INFO - Calling reference model: microsoft/wizardlm-2-8x22b
2024-06-23 12:55:10,081 - INFO - Calling reference model: meta-llama/llama-3-70b
2024-06-23 12:55:11,130 - INFO - Calling reference model: mistralai/mixtral-8x22b-instruct
2024-06-23 12:55:12,460 - INFO - Calling reference model: databricks/dbrx-instruct
2024-06-23 12:55:13,812 - INFO - Aggregating responses using primary model: qwen/qwen-110b-chat

Final Aggregated Response:
The situation in the room involves a total of three killers. Initially, there were three killers present. When an individual entered and killed one of them, this new person also committed murder, thereby assuming the status of a killer. Consequently, while the composition of the group has changed with one killer being deceased, the total count of individuals who have committed a killing remains at three: the two surviving original killers and the newcomer who committed the act of murder.

--------------------------------------------------

Enter your query: /bye
2024-06-23 12:55:23,020 - INFO - Exiting the program.
```

# Contribution
Contributions are welcome! Please open an issue or submit a pull request with your changes.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# References
- [Mixture-of-Agents Enhances Large Language Model Capabilities Paper](https://arxiv.org/abs/2406.04692)
- [OpenRouter API Documentation](https://openrouter.ai/docs/quick-start)
- [Matthew Berman Video and test question](https://www.youtube.com/watch?v=aoikSxHXBYw)