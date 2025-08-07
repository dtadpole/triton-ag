import concurrent.futures
import threading
import queue
import time
import json
import torch
from tqdm import tqdm
import pickle
import argparse
import gc

class ThreadSafeModelPool:
    """Thread-safe model pool that many community members use"""

    def __init__(self, model_configs, max_workers=4):
        self.model_queue = queue.Queue()
        self.max_workers = max_workers
        self.lock = threading.Lock()

        # Pre-load models
        for i, config in enumerate(model_configs):
            self._create_model_instance(config, i)

    def _create_model_instance(self, config, worker_id):
        """Create model instance - each in separate thread to avoid conflicts"""
        def load_model():
            try:
                from unsloth import FastLanguageModel
                model, tokenizer = FastLanguageModel.from_pretrained(**config)
                FastLanguageModel.for_inference(model)
                self.model_queue.put({
                    'id': worker_id,
                    'model': model,
                    'tokenizer': tokenizer,
                    'config': config
                })
            except Exception as e:
                print(f"Failed to load model {worker_id}: {e}")

        thread = threading.Thread(target=load_model)
        thread.start()
        thread.join()

    def get_model(self):
        """Get available model instance"""
        return self.model_queue.get()

    def return_model(self, model_instance):
        """Return model instance to pool"""
        self.model_queue.put(model_instance)

def worker_function(task_data, model_pool):
    """Worker function that processes tasks"""
    model_instance = model_pool.get_model()

    try:
        model = model_instance['model']
        tokenizer = model_instance['tokenizer']

        # Process task
        inputs = tokenizer(task_data['text'], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        del inputs
        return {'task_id': task_data['id'], 'result': result}

    finally:
        model_pool.return_model(model_instance)

# test function
def get_tasks_done():
    for _ in range(5):
        model_configs = [
            {
                'model_name': "unsloth/llama-3-8b-bnb-4bit",
                'max_seq_length': 2048,
                'dtype': None,
                'load_in_4bit': True,
                "device_map" :{"": 0},
            },
            {
                'model_name': "unsloth/Qwen3-8B-unsloth-bnb-4bit",
                'max_seq_length': 2048,
                'dtype': None,
                'load_in_4bit': True,
                "device_map" :{"": 1},
            }
        ]

        pool = ThreadSafeModelPool(model_configs, max_workers=2)

        tasks = [{'id': i, 'text': f'Task {i} text'} for i in range(10)]
        print("start processing texts")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker_function, task, pool) for task in tasks]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        print(results)
        print(f"Processed {len(results)} tasks")
        del pool
        torch.cuda.empty_cache()
        gc.collect()
        print("delete the pool")


def rollout_policy_logproba_single_shell(prompts_file, unsloth_config_file, batch_size=4, max_new_tokens=3000, temperature=0.6, num_completion_per_prompt=8):

    from unsloth import FastLanguageModel

    with open(unsloth_config_file, 'r') as f:
        config = json.load(f)

    with open(prompts_file, 'r') as f:
        prompts = json.load(f)

    model, tokenizer = FastLanguageModel.from_pretrained(**config)
    FastLanguageModel.for_inference(model)

    try:
        device_ = model.device
        n_prompts = len(prompts)
        n_batches = (n_prompts + batch_size - 1) // batch_size
        all_results = []

        for batch_i in tqdm(range(n_batches)):
            current_prompts = prompts[batch_i * batch_size : (batch_i + 1) * batch_size]
            inputs = tokenizer(current_prompts, return_tensors="pt", padding=True, truncation=True).to(device_)
            batch_input_lengths = inputs.attention_mask.sum(dim=1)
            max_input_length = max(batch_input_lengths).item()
            # Generate with token IDs and log probabilities
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=num_completion_per_prompt,  # Generate 8 different completions, it is much faster than generating 1 by 1
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1
                )
            all_sequences = outputs.sequences  # Shape: [num_completion_per_prompt, sequence_length]
            scores = [s_.cpu() for s_ in outputs.scores]  # List of tensors, each with shape [num_completion_per_prompt, vocab_size]

            # process the socre to get the log prob
            all_scores = torch.stack(scores, dim=0).cpu()  # Shape: [num_new_tokens, n_prompts*num_completion_per_prompt, vocab_size]
            all_log_probs = torch.log_softmax(all_scores, dim=-1).cpu()  # Shape: [num_new_tokens, n_prompts*num_completion_per_prompt, vocab_size]
            del scores
            del all_scores
            torch.cuda.empty_cache()
            gc.collect()

            for cur_id, prompt in enumerate(current_prompts):
                input_length = batch_input_lengths[cur_id].item()
                prompt_start_idx = cur_id * num_completion_per_prompt
                prompt_end_idx = (cur_id + 1) * num_completion_per_prompt
                prompt_sequences = all_sequences[prompt_start_idx:prompt_end_idx]
                prompt_new_tokens = prompt_sequences[:, max_input_length:].cpu()
                prompt_log_probs = all_log_probs[:, prompt_start_idx:prompt_end_idx, :]
                # Extract log probs for each sequence
                results = []
                for seq_idx in range(num_completion_per_prompt):
                    sequence_tokens = prompt_new_tokens[seq_idx]
                    # Get log probs for this specific sequence
                    # Remove padding tokens (including EOS used as padding)
                    # Find the first occurrence of EOS/pad token
                    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
                    eos_token_id = tokenizer.eos_token_id

                    # Find where to truncate (first EOS or pad token)
                    truncate_idx = len(sequence_tokens)  # Default to full length

                    for i, token_id in enumerate(sequence_tokens):
                        if token_id == pad_token_id or (eos_token_id and token_id == eos_token_id):
                            truncate_idx = i + 1  # Include the EOS token itself
                            break

                    # Truncate tokens and corresponding log probs
                    valid_tokens = sequence_tokens[:truncate_idx]

                    # Get log probs for this specific sequence (only for valid tokens)
                    if len(valid_tokens) > 0:
                        sequence_log_probs = prompt_log_probs[torch.arange(len(valid_tokens)), seq_idx, valid_tokens]
                    else:
                        sequence_log_probs = torch.tensor([])

                    assert len(valid_tokens) == len(sequence_log_probs), "Error: completion_token_ids and completion_log_probs have different length"
                    results.append({
                        'sequence_id': seq_idx,
                        'prompt_token_ids': inputs["input_ids"][cur_id, :input_length].cpu().tolist(), # all input ids are the same, save the 1st element
                        'completion_token_ids': valid_tokens.tolist(),
                        'completion_log_probs': sequence_log_probs.tolist(),
                        'text': tokenizer.decode(sequence_tokens, skip_special_tokens=True),
                    })
                all_results.append((prompt, results))

            # release GPU memory for inputs and outputs
            del all_sequences
            del outputs
            del inputs
            torch.cuda.empty_cache()
            gc.collect()
        return all_results
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        return []


def save_pickle(obj, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        load_obj = pickle.load(file)
    return load_obj


# Community usage pattern
if __name__ == "__main__":
    # get_tasks_done()
    parser = argparse.ArgumentParser(description="Single shell command script for unsloth inference")
    parser.add_argument("--unsloth_config_file", type=str, default="temp/config.json")
    parser.add_argument("--prompts_file", type=str, default="temp/prompts.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num_completion_per_prompt", type=int, default=2)
    parser.add_argument("--result_path", type=str, default="temp/tem.pickle")
    args = parser.parse_args()
    results = rollout_policy_logproba_single_shell(args.prompts_file,
                                                   args.unsloth_config_file,
                                                   batch_size=args.batch_size,
                                                   max_new_tokens=args.max_new_tokens,
                                                   temperature=args.temperature,
                                                   num_completion_per_prompt=args.num_completion_per_prompt)
    save_pickle(results, args.result_path)
