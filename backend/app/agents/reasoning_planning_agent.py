from typing import Any, Dict, Optional
import json
import re
from .base_agent import BaseAgent
from app.llm.llm import llm_ollama_deepseek_ainvoke # Import the DeepSeek LLM call

class ReasoningPlanningAgent(BaseAgent):
    """
    The Reasoning/Planning Agent (DeepSeek) performs core task analysis,
    generates detailed plans, identifies necessary actions, and produces
    raw, unformatted results along with internal thought processes.
    """

    def __init__(self):
        super().__init__(name="Reasoning/Planning Agent",
                         description="Performs core task analysis and planning.")

    async def process(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a task, generating a plan and raw output using DeepSeek.

        Args:
            task (str): The specific task to be processed.
            context (Dict[str, Any]): Contextual information for the task.

        Returns:
            Dict[str, Any]: A dictionary containing the raw output and internal thoughts.
        """
        thoughts = []
        output = {}

        thoughts.append(f"Reasoning/Planning Agent received task: '{task}'")
        thoughts.append("Analyzing task and formulating a plan using DeepSeek...")

        # Construct the prompt for DeepSeek
        prompt = f"""
        You are a highly skilled reasoning and planning AI. Your task is to analyze the user's request and generate a detailed plan or direct response.
        
        User Request: {task}
        
        Context: {context.get('user_prompt', '')}
        
        Available Search Options:
        - AI Search (search_ai_active): {context.get('search_ai_active', False)}
        - RAG Search (search_rag_active): {context.get('search_rag_active', False)}
        - Online Search (search_online_active): {context.get('search_online_active', False)}

        Additional Information:
        {f"Processed Files Info: {json.dumps(context.get('processed_files_info', []), indent=2, ensure_ascii=False)}" if context.get('processed_files_info') else ""}
        {f"RAG Search Results: {json.dumps(context.get('rag_results', []), indent=2, ensure_ascii=False)}" if context.get('rag_results') else ""}
        {f"Online Search Results: {json.dumps(context.get('online_search_results', []), indent=2, ensure_ascii=False)}" if context.get('online_search_results') else ""}

        Based on the user request, context, and available search options, formulate a detailed plan or direct response.
        If the request involves code, provide a plan and a code snippet. If it's a debugging task, provide a diagnosis and steps to fix. For general queries, provide a direct answer.
        Consider if any of the search options are relevant and how they might be used to fulfill the request.

        Your output should be a JSON string with the following structure:
        {{
            "plan": "...", // Optional: Detailed plan if applicable
            "code_snippet": "...", // Optional: Generated code if applicable
            "diagnosis": "...", // Optional: Diagnosis if debugging
            "steps_to_fix": ["...", "..."], // Optional: Steps if debugging
            "answer": "...", // Optional: Direct answer for general queries
            "response": "...", // General response if none of the above apply
            "thoughts": ["...", "..."] // Your internal thoughts/reasoning process
        }}
        """

        try:
            # Call the DeepSeek LLM
            deepseek_response_str = await llm_ollama_deepseek_ainvoke(prompt)
            thoughts.append(f"DeepSeek raw response: {deepseek_response_str}")

            # Attempt to parse the response as JSON
            try:
                # Attempt to extract JSON from the response, in case it's wrapped in markdown or has extra text
                json_match = re.search(r"```json\s*(.*?)\s*```", deepseek_response_str, re.DOTALL)
                if json_match:
                    json_string = json_match.group(1)
                    thoughts.append("Extracted JSON block from DeepSeek response.")
                else:
                    json_string = deepseek_response_str.strip() # Try to strip whitespace if no markdown block
                    thoughts.append("No JSON block found, attempting to parse raw response after stripping whitespace.")

                deepseek_parsed_response = json.loads(json_string)
                output.update(deepseek_parsed_response)
                thoughts.extend(deepseek_parsed_response.get("thoughts", []))
                # Remove thoughts from output if they are already extracted
                if "thoughts" in output:
                    del output["thoughts"]
            except json.JSONDecodeError as e:
                thoughts.append(f"DeepSeek response was not valid JSON. Error: {e}. Raw response: {deepseek_response_str}. Treating as plain text response.")
                output["response"] = deepseek_response_str
            except Exception as e: # Catch any other unexpected errors during parsing
                thoughts.append(f"Unexpected error during DeepSeek response parsing: {e}. Raw response: {deepseek_response_str}. Treating as plain text response.")
                output["response"] = deepseek_response_str
                
        except Exception as e:
            output["error"] = f"Error calling DeepSeek LLM: {e}"
            thoughts.append(f"Error during DeepSeek LLM call: {e}")
            output["response"] = f"An error occurred while processing your request: {e}"

        thoughts.append("Reasoning/Planning complete. Preparing raw output.")

        return {"output": output, "thoughts": thoughts}
