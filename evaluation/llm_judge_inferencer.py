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
        if response_content.startswith("```") and response_content.endswith("```"):
            response_content = response_content[3:-3].strip()

        # try to parse the remaining content as JSON
        parsed_content = json.loads(response_content)
        return parsed_content

    def judge(self, prompt) -> dict:
        response = self.client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}]
        )


        content = response.choices[0].message.content
        content = self.parse_response(content)
        return content
