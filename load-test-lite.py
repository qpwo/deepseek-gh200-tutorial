import openai
from random_word import RandomWords
import time
import argparse
import asyncio
import random
rw = RandomWords()
wordos = list(rw.service.valid_words.keys())
def get_random_words(num_words: int) -> str:
    return " ".join(wordos[random.randrange(0, len(wordos))] for _ in range(num_words))

async def run_prompt(model: str, prompt: str, max_tokens: int):
    """Generates text from the OpenAI API (non-streaming)."""
    try:
        client = openai.AsyncOpenAI(
            api_key="asdf1234",  # This doesn't matter for local models
            base_url="http://localhost:8000/v1",
        )

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,  # No streaming
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content, response.usage.completion_tokens, response.usage.prompt_tokens
    except Exception as error:
        print(f"Error in run_prompt: {error}")
        return "", 0, 0

async def process_batch(model: str, batch_size: int, num_input_words: int, max_tokens: int):
    """Processes a batch of prompts and prints the final rate."""

    prompts = [
        "Tell a story inspired by these words: " + get_random_words(num_input_words)
        for _ in range(batch_size)
    ]

    start_time = time.time()
    total_completion_tokens = 0
    total_prompt_tokens = 0

    tasks = [run_prompt(model, prompt, max_tokens) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    for content, completion_tokens, prompt_tokens in results:
        total_completion_tokens += completion_tokens
        total_prompt_tokens += prompt_tokens


    end_time = time.time()
    duration = end_time - start_time
    print(f"Batch Size: {batch_size}")
    print(f"Total Prompt Tokens: {total_prompt_tokens};  rate: {total_prompt_tokens / duration:.2f} tokens/sec")
    print(f"Total Completion Tokens: {total_completion_tokens};  rate: {total_completion_tokens / duration:.2f} tokens/sec")


async def main():
    parser = argparse.ArgumentParser(description="Run load tests against an OpenAI-compatible API.")
    parser.add_argument("--model", type=str, default="dsr1", help="The model name")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--num_input_words", type=int, default=10, help="Number of random words for the prompt")
    parser.add_argument(
        "--sizes",
        type=str,
        default="1,2,4,8,16,32,64",
        help="Comma-separated list of batch sizes",
    )
    args = parser.parse_args()

    batch_sizes = [int(s.strip()) for s in args.sizes.split(",")]

    for batch_size in batch_sizes:
        await process_batch(args.model, batch_size, args.num_input_words, args.max_tokens)


if __name__ == "__main__":
    asyncio.run(main())
