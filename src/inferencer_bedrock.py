"""
Documentación: https://docs.claude.com/en/api/client-sdks
"""

from anthropic import AnthropicBedrock

class SonnetInferencer:
    def __init__(self, model_id="anthropic.claude-sonnet-4-6", aws_region="eu-west-2", max_tokens=500):
        self.client = AnthropicBedrock(aws_region=aws_region)
        self.model_id = model_id
        self.max_tokens = max_tokens

    def infer(self, prompt) -> tuple[str, dict]:
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # return response and total tokens
        return response.content[0].text, {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }