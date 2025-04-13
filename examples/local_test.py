"""
local_test.py

Local inference test.
"""

from dotenv import load_dotenv
from fabricai_inference_server.engine import LlamaEngine
from fabricai_inference_server.settings import settings

load_dotenv()


def main():
    model_path = settings.llm_model

    if not model_path:
        raise ValueError(
            "LLM_MODEL is not set or is empty. Please set it in your .env file."
        )

    engine = LlamaEngine(
        model_path=model_path,
        n_ctx=settings.llama_ctx,
        n_threads=settings.llama_threads,
        gpu_layers=settings.llama_gpu_layers,
    )

    prompt = "Hello, how do you define an 'LLM' in one concise sentence."
    print(f"Prompt: {prompt}\n")

    tokens = []
    for token in engine.generate_stream(prompt, max_tokens=100, temperature=0.7):
        tokens.append(token)

    output_text = "".join(tokens)
    print("Model response:")
    print(output_text)


if __name__ == "__main__":
    main()
