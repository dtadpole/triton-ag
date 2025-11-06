import torch
from transformers import AutoTokenizer
from logger import logger


def uft_group_to_dataset(
    group: list[dict],
    tokenizer: AutoTokenizer,
):
    """
    Convert a group of GRPO results to UFT dataset format.

    UFT dataset includes all GRPO fields plus hint-related information:
    - hint_text: the hint string
    - hint_token_ids: tokenized hint
    - hint_length: length of hint tokens
    - previous_conversation_length: length of tokens before hint

    The structure is: previous_conversation + hint + completion
    - previous_conversation + hint = prompt (used for LLM generation)
    - completion is generated based on the prompt
    - GRPO loss is computed on completion tokens
    - SFT loss should be computed on hint tokens (based on previous_conversation)
    """
    from logger import logger

    group_dataset = []
    for result in group:
        # Get GRPO-related data
        vllm_prompt_ids = tokenizer.encode(result["prompt"])
        vllm_completion_ids = [logprob['token_id'] for logprob in result["logprobs"]]
        vllm_completion_log_probs = [logprob['logprob'] for logprob in result["logprobs"]]
        vllm_input_ids = torch.tensor(vllm_prompt_ids + vllm_completion_ids)
        vllm_attention_mask = torch.ones_like(vllm_input_ids)
        logp_server_prompt_ids = torch.tensor(result["logp_server_prompt_ids"])
        logp_server_completion_ids = torch.tensor(result["logp_server_completion_ids"])
        logp_server_input_ids = torch.tensor(result["logp_server_input_ids"])
        logp_server_attention_mask = torch.ones_like(logp_server_input_ids)
        logp_server_logps = result["logp_server_logps"]

        # Check if the prompt ids length are different
        if len(vllm_prompt_ids) != len(logp_server_prompt_ids):
            logger.error(f"len(vllm_prompt_ids) [{len(vllm_prompt_ids)}] != len(logp_server_prompt_ids): [{len(logp_server_prompt_ids)}]")
            continue

        # Calculate the number of prompt ids that are different
        diff_count_prompt_ids = sum(1 for i, j in zip(vllm_prompt_ids, logp_server_prompt_ids) if i != j)
        if diff_count_prompt_ids > 0:
            logger.error(f"vllm_prompt_ids != logp_server_prompt_ids: [{diff_count_prompt_ids}/{len(vllm_prompt_ids)} tokens different]")
            continue

        # Check if the completion ids length are different
        if len(vllm_completion_ids) != len(logp_server_completion_ids):
            logger.error(f"len(vllm_completion_ids) [{len(vllm_completion_ids)}] != len(logp_server_completion_ids): [{len(logp_server_completion_ids)}]")
            continue

        # Calculate the number of completion ids that are different
        diff_count_completion_ids = sum(1 for i, j in zip(vllm_completion_ids, logp_server_completion_ids) if i != j)
        if diff_count_completion_ids > 0:
            logger.error(f"vllm_completion_ids != logp_server_completion_ids: [{diff_count_completion_ids}/{len(vllm_completion_ids)} tokens different]")
            continue

        # Check if logps length are different
        if len(vllm_completion_log_probs) != len(logp_server_logps) - len(logp_server_prompt_ids) + 1:
            logger.error(f"len(vllm_completion_log_probs) [{len(vllm_completion_log_probs)}] != len(logp_server_logps) - len(logp_server_prompt_ids) + 1: [{len(logp_server_logps) - len(logp_server_prompt_ids) + 1}]")
            continue

        # Process hint information
        hint_text = result.get("hint", "")
        hint_token_ids = []
        hint_length = 0
        previous_conversation_length = 0

        if hint_text:
            # Tokenize the hint
            hint_token_ids = tokenizer.encode(hint_text, add_special_tokens=False)
            hint_length = len(hint_token_ids)

            # Calculate previous_conversation_length
            # prompt = previous_conversation + hint
            # So: previous_conversation_length = len(prompt) - len(hint)
            previous_conversation_length = len(logp_server_prompt_ids) - hint_length

            # Validate that hint tokens match the end of prompt
            if previous_conversation_length >= 0 and hint_length > 0:
                prompt_hint_tokens = logp_server_prompt_ids[previous_conversation_length:previous_conversation_length + hint_length].tolist()
                if prompt_hint_tokens != hint_token_ids:
                    logger.warning(f"Hint tokens don't match end of prompt. This may indicate tokenization mismatch.")
                    logger.warning(f"Expected hint tokens: {hint_token_ids[:10]}...")
                    logger.warning(f"Actual prompt end tokens: {prompt_hint_tokens[:10]}...")

        group_dataset.append({
            # GRPO fields
            'task_tag': result["task_tag"],
            'turn_tag': result["turn_tag"],
            'reward': result["reward"],
            'runtime': result["runtime"],
            'checkpoint_name': result["checkpoint_name"],
            'reward_items': result["reward_items"],
            'advantage': result["advantage"],
            'vllm_prompt_ids': vllm_prompt_ids,
            'vllm_completion_ids': vllm_completion_ids,
            'vllm_completion_log_probs': vllm_completion_log_probs,
            'vllm_input_ids': vllm_input_ids,
            'vllm_attention_mask': vllm_attention_mask,
            'logp_server_prompt_ids': logp_server_prompt_ids,
            'logp_server_completion_ids': logp_server_completion_ids,
            'logp_server_input_ids': logp_server_input_ids,
            'logp_server_logps': logp_server_logps,
            'logp_server_attention_mask': logp_server_attention_mask,
            'input_ids': logp_server_input_ids,
            'attention_mask': logp_server_attention_mask,
            # UFT-specific fields for hint
            'hint_text': hint_text,
            'hint_token_ids': hint_token_ids,
            'hint_length': hint_length,
            'previous_conversation_length': previous_conversation_length,
        })

    return group_dataset
