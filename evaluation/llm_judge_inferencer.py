"""
Using Groq LLM provider: https://groq.com/


"""
import os
import json
from groq import Groq

class LLMJudgeInferencer:
    def __init__(self):
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    def parse_response(self, response_content):
        # remove everything between <think> and </think> if they exist
        if "<think>" in response_content and "</think>" in response_content:
            start = response_content.index("<think>")
            end = response_content.index("</think>") + len("</think>")
            response_content = response_content[:start] + response_content[end:]

        response_content = response_content.strip()

        # remove markdown code block if it exists
        # sometimes it also starts with ```json, so we check for that as well
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content[len("```json"): -len("```")].strip()
        elif response_content.startswith("```") and response_content.endswith("```"):
            response_content = response_content[len("```"): -len("```")].strip()

        # try to parse the remaining content as JSON
        try:
            parsed_content = json.loads(response_content)
        except json.JSONDecodeError:
            raise ValueError(f"Unable to parse LLM Judge response as JSON. Response content: {response_content}")
        return parsed_content

    def judge(self, prompt) -> dict:
        response = self.client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content
        content = self.parse_response(content)
        return content
