import ollama

class OllamaLLM:
    def __init__(self, model):
        self.model = model
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        response = ollama.chat(
            model=self.model,
            format="json",
            messages=[
                {
                    "role":"system",
                    "content":system_prompt or ""
                },
                {
                    "role":"user",
                    "content":
                        "Everything below is the complete repository.\n"
                        "Do not use any external knowledge.\n\n"
                        + prompt
                }
            ],
            options={
                "temperature":0.1,
            },
        )
        return response["message"]["content"]