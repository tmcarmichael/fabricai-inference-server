"""
engine.py

Load local quantized model with config. Currently supports Llama.
"""

import os
from typing import Generator, Optional
from llama_cpp import Llama
from fabricai_inference_server.settings import settings


class LlamaEngine:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        use_mlock: bool = False,
        gpu_layers: int = 0,  # TODO: Validate hw for gpu accelleration
    ):
        """
        Initializes a local Llama model from a quantized GGUF file.

        :param model_path: Path to your local Llama 2 (13B) GGUF model
        :param n_ctx: Context size
        :param n_threads: Number of CPU threads to use
        :param use_mlock: Whether to try mlock to keep model in RAM
        :param gpu_layers: Layers offloaded to GPU, if Apple GPU acceleration is supported
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.use_mlock = use_mlock
        self.gpu_layers = gpu_layers
        self.llama = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            use_mlock=self.use_mlock,
            n_gpu_layers=self.gpu_layers,
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        stop: Optional[list[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Generator function that yields tokens in a stream.

        :param prompt: Prompt text
        :param max_tokens: Maximum number of tokens to generate
        :param temperature: Sampling temperature
        :param top_p: Top-p (nucleus) sampling
        :param repeat_penalty: Penalty for repeating tokens
        :param stop: Optional list of stop sequences
        :return: Yields partial tokens as they are generated
        """
        if stop is None:
            stop = []

        result_iter = self.llama(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
            stream=True,
        )

        for token_chunk in result_iter:
            text = token_chunk["choices"][0]["text"]
            yield text


def load_default_engine() -> LlamaEngine:
    """
    Loads a default engine with parameters from environment or config (pydantic settings).
    """
    model_path = settings.llm_model
    n_threads = settings.llama_threads
    n_ctx = settings.llama_ctx
    gpu_layers = settings.llama_gpu_layers  # TODO: Validate hw for gpu accelleration

    engine = LlamaEngine(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        gpu_layers=gpu_layers,
        use_mlock=False,
    )
    return engine
