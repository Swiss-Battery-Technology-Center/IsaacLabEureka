# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import os
import re

import traceback
import openai
import logging


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
        temperature: float,
        num_suggestions: int = 1,
        system_prompt: str = "",
        env_type: str = "",
        eureka_task: str = "",
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
        # self._prompts = [{"role": "system", "content": system_prompt}]
        self._prompts = []
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
        logging.info(f"LLMManager initialized with model: {gpt_model}, temperature: {temperature}")
        
    # appends a string to an already existing system prompt
    def append_to_system_prompt(self, additioal_prompt: str):
        assert self._prompts[0]["role"] == "system" 
        self._prompts[0]["content"] += additioal_prompt

    # granular control over self._prompts
    # in the main codeblock, make sure
    # appends a new system prompt to an empty list
    def append_system_prompt(self, prompt: str):
        assert not self._prompts #  assert that self._prompts is an empty list
        self._prompts.append({"role": "system", "content": prompt})

    # appends a new user prompt 
    def append_user_prompt(self, prompt: str):
        self._prompts.append({"role": "user", "content": prompt})

    # appends a new assistant prompt
    def append_assistant_prompt(self, prompt: str):
        self._prompts.append({"role": "assistant", "content": prompt})


    def clear_prompts(self):
        """Clear the prompts list"""
        self._prompts = []

    # make sure you set the prompts correctly before calling this function
    def call_llm(self, max_retries=3) -> dict:
        logging.info(f"Calling LLM")
        attempt = 0
        while attempt < max_retries:
            try:
                responses = self._client.chat.completions.create(
                    model=self._gpt_model,
                    messages=self._prompts,
                    temperature=self._temperature,
                    n=self._num_suggestions,
                )

                raw_output = responses.choices[0].message.content

                # this catches raw_output that is empty string or None
                if not raw_output:
                    logging.error(f"LLM call failed with empty response, finish reason: {responses.choices[0].finish_reason}")
                    raise RuntimeError("LLM call failed with empty response")

                weights_strings = self.extract_multiple_weights_from_response(raw_output)

                if weights_strings and any(w.strip() for w in weights_strings):
                    # Success!
                    self._total_tokens += responses.usage.total_tokens
                    self._total_query_tokens += responses.usage.prompt_tokens
                    self._total_response_tokens += responses.usage.completion_tokens
                    return {"weight_strings": weights_strings, "raw_output": raw_output}
                else:
                    print(f"response was okay but failed to extract weights at {attempt+1}. Retrying...")
                    logging.error(f"LLM response was non empty but failed to extract weights at attempt {attempt+1}.\n raw_output: \n{raw_output}\n Retrying...")
                    attempt += 1
            except Exception as e:
                print(f"[ERROR] LLM call failed on attempt {attempt+1}: {e}")
                traceback.print_exc()
                logging.error(f"LLM call failed on attempt {attempt+1}: {traceback.format_exc()}")
                attempt += 1
        logging.error("LLM failed to provide valid weight strings after multiple attempts.")
        raise RuntimeError("LLM failed to provide valid weight strings after multiple attempts.")
    
    # no history, doesn't change self._prompts, just calling with a user prompt
    def call_llm_for_summary(self, user_prompt, max_retries=3) -> str:
        logging.info(f"Calling LLM for summary")
        attempt = 0
        while attempt < max_retries:
            try:
                responses = self._client.chat.completions.create(
                    model=self._gpt_model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=self._temperature,
                    n=self._num_suggestions,
                )

                raw_output = responses.choices[0].message.content

                # this catches raw_output that is empty string or None
                if not raw_output:
                    logging.error(f"LLM call failed with empty response, finish reason: {responses.choices[0].finish_reason}")
                    raise RuntimeError("LLM call failed with empty response")          
                # Success!
                self._total_tokens += responses.usage.total_tokens
                self._total_query_tokens += responses.usage.prompt_tokens
                self._total_response_tokens += responses.usage.completion_tokens
                return raw_output

            except Exception as e:
                print(f"[ERROR] LLM call failed on attempt {attempt+1}: {e}")
                traceback.print_exc()
                logging.error(f"LLM call failed on attempt {attempt+1}: {traceback.format_exc()}")
                attempt += 1
        logging.error("LLM failed to provide valid weight strings after multiple attempts.")
        raise RuntimeError("LLM failed to provide valid weight strings after multiple attempts.")

    def get_or_generate_summary(self, summary_type:str, identifier:str, prompt:str, code_string: str, eureka_root_dir: str, override: bool = False):
        """
        Get or generate a summary for context code, PPO code, or success metric code.
        summaries are saved in EUREKA_ROOT_DIR/summary.

        Args:
            summary_type (str): One of "context_code", "ppo_code", or "success_metric_code".
            identifier (str): Either rl_task_type (for context/success metric) or rl_library (for PPO).
            prompt (str): Instructional prompt to prepend to code_string.
            code_string (str): The code to summarize.
            eureka_root_dir (str): Path to the Eureka root directory.
            override (bool): If True, regenerate and overwrite the summary even if it exists.

        Returns:
            str: The summary string.
        """
        os.makedirs(os.path.join(eureka_root_dir, "summary"), exist_ok=True)
        filename = f"{summary_type}_summary_{identifier}.txt"
        summary_path = os.path.join(eureka_root_dir, "summary", filename)

        if not override and os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                cached = f.read().strip()
                if cached:
                    logging.info(f"Using cached summary from {summary_path}")
                    return cached

        # Otherwise, generate new summary via LLM
        logging.info(f"Generating new {filename}")
        llm_input = f"{prompt}\n{code_string}"
        summary = self.call_llm_for_summary(llm_input)
        with open(summary_path, "w") as f:
            f.write(summary)
            logging.info(f"Saved new {filename}")
        return summary



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
            responses = self._client.chat.completions.create(
                model=self._gpt_model,
                messages=self._prompts,
                temperature=self._temperature,
                n=self._num_suggestions,
            )
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
    


