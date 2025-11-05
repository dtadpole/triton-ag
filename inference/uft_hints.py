import numpy as np


async def uft_hints(messages, hint_length_prob, tokenizer):
    """
    Create a prompt with a partial hint from the last turn in messages.

    The function takes the last message in the conversation, tokenizes it,
    and samples a hint length from a binomial distribution. It then creates
    a prompt with only the first hint_length tokens of the last message,
    which will be used for model completion.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        hint_length_prob: Probability parameter for binomial distribution (0 to 1)
                         Determines the expected proportion of tokens to reveal
        tokenizer: Tokenizer object to encode/decode text

    Returns:
        A text prompt string with partial hint from the last turn, ready for LLM completion
    """
    if not messages or len(messages) == 0:
        raise ValueError("Messages list is empty")

    last_message = messages[-1]
    last_content = last_message.get('content', '')

    # Tokenize the last message content
    last_message_tokens = tokenizer.encode(last_content, add_special_tokens=False)
    total_length = len(last_message_tokens)

    if total_length == 0:
        # If last message is empty, return messages without last turn
        messages_without_last = messages[:-1]
        prompt = tokenizer.apply_chat_template(
            messages_without_last,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return {"hint": "", "prompt": prompt}

    # Sample hint length from binomial distribution
    # n = total_length (number of trials)
    # p = hint_length_prob (probability of success for each trial)
    hint_length = np.random.binomial(total_length, hint_length_prob)

    # Take the first hint_length tokens as the hint
    hint_tokens = last_message_tokens[:hint_length]
    hint_text = tokenizer.decode(hint_tokens, skip_special_tokens=True)

    # Apply chat template to all messages except the last one
    # This gives us the prompt with add_generation_prompt starting the assistant turn
    messages_without_last = messages[:-1]
    prompt_base = tokenizer.apply_chat_template(
        messages_without_last,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Now append the hint text directly (without closing the assistant message)
    # The prompt_base ends with something like "<|im_start|>assistant\n"
    # We just append the hint text so the model continues from there
    prompt = prompt_base + hint_text

    return {"hint": hint_text, "prompt": prompt}


async def main():
    """Test function to demonstrate uft_hints functionality."""
    from transformers import AutoTokenizer

    print("=" * 80)
    print("Testing uft_hints function")
    print("=" * 80)

    # Load a tokenizer (you can change this to your actual model)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # Create sample messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant specialized in Python programming."
        },
        {
            "role": "user",
            "content": "Write a function to calculate fibonacci numbers."
        },
        {
            "role": "assistant",
            "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        }
    ]

    print("\nOriginal messages:")
    for i, msg in enumerate(messages):
        print(f"\n[Message {i}] Role: {msg['role']}")
        print(f"Content: {msg['content'][:100]}..." if len(msg['content']) > 100 else f"Content: {msg['content']}")

    # Get the full last message for comparison
    last_message_content = messages[-1]['content']
    last_message_tokens = tokenizer.encode(last_message_content, add_special_tokens=False)
    print(f"\nLast message has {len(last_message_tokens)} tokens")

    # Test with different hint_length_prob values
    test_probs = [0.2, 0.5, 0.8]

    for prob in test_probs:
        print("\n" + "=" * 80)
        print(f"Testing with hint_length_prob = {prob}")
        print("=" * 80)

        # Run multiple times to see the distribution
        for run in range(3):
            result = await uft_hints(messages, prob, tokenizer)
            hint = result["hint"]
            prompt = result["prompt"]

            print(f"\nRun {run + 1}:")
            print(f"Hint: {hint}")
            print(f"Hint length: {len(hint)} characters")
            print(f"\nPrompt length: {len(prompt)} characters")
            print(f"Prompt preview (last 200 chars):")
            print(prompt[-200:])
            print("-" * 80)

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
