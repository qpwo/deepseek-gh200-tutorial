import openai
from random_word import RandomWords
import time
import argparse
import asyncio
from typing import List
import random

rw = RandomWords()
wordos = list(rw.service.valid_words.keys())
def get_random_words(num_words: int) -> str:
    return " ".join(wordos[random.randrange(0, len(wordos))] for _ in range(num_words))

async def run_prompt(model: str, prompt: str, max_tokens: int):
    """Generates text from the OpenAI API, handling streaming and errors.

    Args:
        model: The model name.
        prompt: The input prompt.
        max_tokens: The maximum number of tokens to generate.

    Yields:
        str: The generated text chunks.
    """
    try:
        client = openai.AsyncOpenAI(
            api_key="asdf1234",  # This doesn't matter for local models, but is needed
            base_url="http://localhost:8000/v1",
        )

        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=max_tokens,
        )
        async for chunk in stream:
            try:
                yield chunk.choices[0].delta.content or ""
            except:
                yield ''
    except Exception as error:
        print(f"Error in runPrompt: {error}")
        yield f"Error: {error}"

completions = {}
running = set()
async def process_single_prompt(model: str, prompt: str, max_tokens: int, index: int):
    """Processes a single prompt and returns token counts.

    Args:
        model: The model name.
        prompt: The input prompt.
        max_tokens: The maximum number of tokens to generate.
        index: prompt index

    Returns:
        dict: A dictionary containing completion_tokens and prompt_tokens.
    """
    running.add(index)
    print(f"Prompt {index + 1}: {prompt}")
    completion_text = ""
    completion_tokens = 0
    prompt_tokens = len(prompt.split())  # more accurate token count

    try:
        async for chunk in run_prompt(model, prompt, max_tokens):
            completion_tokens += 1
            completion_text += chunk
            completions[index] = completion_text
        running.remove(index)
        return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens}

    except Exception as error:
        print(f"Error processing prompt: {error}")
        running.remove(index)
        return {"completion_tokens": 0, "prompt_tokens": 0}

async def print_progress():
    while running:
        print('\n' * 30)
        print(f"Running: {running}")
        for index in running:
            print(f"Prompt {index + 1}: {completions.get(index, '')}")
        await asyncio.sleep(1)

async def process_batch(model: str, batch_size: int, num_input_words: int, max_tokens: int):
    """Processes a batch of prompts and prints summary statistics.

    Args:
        model: The model name.
        batch_size: The number of prompts to process in parallel.
        num_input_words: The number of random words to use in each prompt.
        max_tokens: The maximum number of tokens to generate.
    """
    print(f"Batch Size: {batch_size}")

    prompts = [
        "Tell a story inspired by these words: " + get_random_words(num_input_words)
        for _ in range(batch_size)
    ]

    start_time = time.time()
    total_completion_tokens = 0
    total_prompt_tokens = 0
    completed_count = 0

    tasks = [
        process_single_prompt(model, prompt, max_tokens, index)
        for index, prompt in enumerate(prompts)
    ]
    tasks.append(print_progress())
    results = await asyncio.gather(*tasks)
    results = results[:-1]

    for result in results:
        total_completion_tokens += result["completion_tokens"]
        total_prompt_tokens += result["prompt_tokens"]
        completed_count += 1

    end_time = time.time()
    duration = end_time - start_time

    import sys
    fa = open('load.log', 'a')
    for f in [sys.stdout, fa]:
        print(f"Completed {completed_count} / {batch_size} requests in {duration:.2f} seconds.", file=f)
        print(f"Total Prompt Tokens: {total_prompt_tokens}", file=f)
        print(f"Total Completion Tokens: {total_completion_tokens}", file=f)
        print(f"Prompt rate: {(total_prompt_tokens / duration):.2f} tokens/second", file=f)
        print(f"Prompt rate per input: {(total_prompt_tokens / duration / batch_size):.2f} tokens/second", file=f)
        print(f"Completion rate: {(total_completion_tokens / duration):.2f} tokens/second", file=f)
        print(f"Completion rate per input: {(total_completion_tokens / duration / batch_size):.2f} tokens/second", file=f)
        print("-" * 20, file=f)
    fa.close()


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
