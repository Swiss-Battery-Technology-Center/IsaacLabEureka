# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import os
import re

import openai

class LLMManager:
    """Manager to interface with the LLM API.

    This class is responsible for interfacing with the LLM API to generate rewards.
    It establishes a connection either to native OpenAI API, or to the Azure OpenAI API.

    The Openai API relies on the following environment variables to be set:
    - For the native OpenAI API, the environment variable OPENAI_API_KEY must be set.
    - For the Azure OpenAI API, the environment variables AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set.
    """

    def __init__(
        self,
        gpt_model: str,
        num_suggestions: int,
        temperature: float,
        system_prompt: str,
        env_type: str = "",
        task_type: str = "",
        parameters_to_tune: list[str] = [],
    ):
        """Initialize the LLMManager

        Args:
            gpt_model: The model to use for the LLM API
            num_suggestions: The number of independent suggestions to generate
            temperature: The temperature to use for the LLM API
            system_prompt: The system prompt to provide to the LLM API
        """

        self._gpt_model = gpt_model
        self._num_suggestions = num_suggestions
        self._temperature = temperature
        self._prompts = [{"role": "system", "content": system_prompt}]

        if "AZURE_OPENAI_API_KEY" in os.environ:
            self._client = openai.AzureOpenAI(api_version="2024-02-01")
        elif "OPENROUTER_API_KEY" in os.environ:
            self._client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY")
            )
        elif "OPENAI_API_KEY" in os.environ:
            self._client = openai.OpenAI()
        else:
            raise RuntimeError("No Openai API key found in environment variables")
        self._total_tokens=0
        self._total_query_tokens=0
        self._total_response_tokens=0
        self._input_token_price = 0.27
        self._output_token_price = 1.10
        
    def append_to_system_prompt(self, additioal_prompt: str):
        assert self._prompts[0]["role"] == "system" 
        self._prompts[0]["content"] += additioal_prompt
    def extract_code_from_response(self, response: str) -> str:
        """Extract the code component from the LLM response

        If the response contains a code block of the form "```python ... ```", extract the code block from the response.
        Otherwise, return an empty string.

        Args:
            response: The response from the LLM API
        """
        pattern = r"```python(.*?)```"
        result = re.findall(pattern, response, re.DOTALL)
        code_string = ""
        if result is not None and len(result) > 0:
            code_string = result[-1]
            # Remove leading newline characters
            code_string = code_string.lstrip("\n")
        return code_string

    def prompt(self, user_prompt: str, assistant_prompt: str = None) -> list[str]:
        """Call the LLM API to collect responses

        Args:
            user_prompt: The user prompt to provide to the LLM API
            assistant_prompt: The assistant prompt to provide to the LLM API

        Returns:
            A dictionary containing the reward strings and raw outputs from the LLM

        Raises:
            Exception: If there is an error with the LLM API
        """
        if assistant_prompt is not None:
            self._prompts.append({"role": "assistant", "content": assistant_prompt})
        self._prompts.append({"role": "user", "content": user_prompt})

        # The official Eureka code only keeps the last round of feedback
        if len(self._prompts) == 6:
            self._prompts.pop(2)
            self._prompts.pop(2)

        try:
            responses = self._client.chat.completions.create(
                model=self._gpt_model,
                messages=self._prompts,
                temperature=self._temperature,
                n=self._num_suggestions,
            )
        except Exception as e:
            raise RuntimeError("An error occurred while prompting the LLM") from e
        self._total_tokens += responses.usage.total_tokens
        self._total_query_tokens += responses.usage.prompt_tokens
        self._total_response_tokens += responses.usage.completion_tokens
        raw_outputs = [response.message.content for response in responses.choices]
        reward_strings = [
            self.extract_code_from_response(raw_output) for raw_output in raw_outputs
        ]
        return {"reward_strings": reward_strings, "raw_outputs": raw_outputs}
    
    def feed_context_code(self, context_code_string: str, chunk_size: int = 50000):
        """Feed long source code context into the LLM as user messages (without requiring a response).

        Args:
            context_code_string: A large string containing the full environment source code.
            chunk_size: Maximum number of characters per chunk sent to the LLM.
        """
        import textwrap

        # Split into chunks
        chunks = textwrap.wrap(context_code_string, width=chunk_size, break_long_words=False, break_on_hyphens=False)

        for i, chunk in enumerate(chunks):
            intro = (
                f"[Chunk {i+1}/{len(chunks)}] Below is a portion of the environment source code "
                "to help you understand how the environment is implemented. You don't have to respond. "
                "Just read and remember it for future queries.\n\n"
            )
            full_prompt = intro + chunk

            # Feed to LLM as user prompt without expecting a response

            try:
                # Send dummy query to register the prompt internally (ignore response)
                print('CREATING CHAT COMPLETION TO FEED CONTEXT')

                responses = self._client.chat.completions.create(
                    model=self._gpt_model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=self._temperature,
                    n=1,
                )
                print('CHAT COMPLETION SUCCESSFUL')
                self._total_tokens += responses.usage.total_tokens
                self._total_query_tokens += responses.usage.prompt_tokens
                self._total_response_tokens += responses.usage.completion_tokens
            except Exception as e:
                raise RuntimeError(f"Failed to feed context chunk {i+1}/{len(chunks)} to the LLM.") from e

        print(f"[INFO] Successfully fed {len(chunks)} context chunks to LLM.")
    def extract_multiple_weights_from_response(self, response: str) -> list[str]:
        """Extracts the weights string from the LLM response.

        The function looks for a dictionary-like structure `{name: weight, name: weight, ...}`
        inside the response. The weight values should be float numbers.

        Args:
            response: The raw response from the LLM API.

        Returns:
            A string representation of the weight dictionary if found, else an empty string.
        """
        pattern = r"\{[^{}]*\}" # Match anything inside curly braces `{...}` a little bit dangerous...
        matches = re.findall(pattern, response, re.DOTALL)
        return matches
    
    def prompt_weights(self, user_prompt: str, assistant_prompt: str = None) -> list[str]:

        if assistant_prompt is not None:
            self._prompts.append({"role": "assistant", "content": assistant_prompt})
        self._prompts.append({"role": "user", "content": user_prompt})

        # The official Eureka code only keeps the last round of feedback
        if len(self._prompts) == 6:
            self._prompts.pop(2)
            self._prompts.pop(2)

        try:
            print('CREATING CHAT COMPLETION TO PROMPT WEIGHTS')
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(60)  # Timeout after 60 seconds
            responses = self._client.chat.completions.create(
                model=self._gpt_model,
                messages=self._prompts,
                temperature=self._temperature,
                n=self._num_suggestions,
            )
            signal.alarm(0)  # Cancel alarm if successful
            print('CHAT COMPLETION SUCCESSFUL')
        except Exception as e:
            raise RuntimeError("An error occurred while prompting the LLM") from e

        raw_outputs = [response.message.content for response in responses.choices]
        # This only works for a single response that contains multiple weight strings
        weights_strings = self.extract_multiple_weights_from_response(raw_outputs[0]) 
        self._total_tokens += responses.usage.total_tokens
        self._total_query_tokens += responses.usage.prompt_tokens
        self._total_response_tokens += responses.usage.completion_tokens
        return {"weight_strings": weights_strings, "raw_outputs": raw_outputs}
    
import signal

class TimeoutException(Exception): pass

def handler(signum, frame):
    raise TimeoutException("LLM call took too long!")

