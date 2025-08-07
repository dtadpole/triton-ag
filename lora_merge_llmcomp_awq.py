import yaml
import torch
from util import is_devserver, logger
import shutil
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation
from awq import AutoAWQForCausalLM
from functools import partial
from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from data_processor import load_experiences, format_conversation
from trainerUtil import _manual_format_conversation
from tqdm import tqdm

# Set up environment
if is_devserver():
    os.environ['HF_HOME'] = '/root/.cache/'
    os.environ['HF_HOME'] = '/root/.cache/'
    os.environ['https_proxy'] = 'http://fwdproxy:8080'
    os.environ['http_proxy'] = 'http://fwdproxy:8080'
    os.environ['ftp_proxy'] = 'http://fwdproxy:8080'
    os.environ['http_no_proxy'] = "'\''\'\'''\''.facebook.com|.tfbnw.net|*.fb.com'\''\'\'"



def load_wikitext(tokenizer):
    def preprocess_wiki(example, tokenizer):
        return {
            "text": tokenizer.apply_chat_template(
                [{"role": "user", "content": example["text"]}],
                tokenize=False,
            )
        }
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return  data.map(partial(preprocess_wiki, tokenizer=tokenizer))


def load_alpaca(tokenizer):
    def preprocess_alpaca(example, tokenizer):
        return {
            "text": tokenizer.apply_chat_template(
                [{"role": "user", "content": example["instruction"] + "\n" + example["input"]}],
                tokenize=False,
            )
        }
    alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train[:256]")
    # alpaca_data = []
    # for i, sample in enumerate(alpaca_dataset):
    #     if i >= 256:
    #         break

    #     instruction = sample['instruction']
    #     input_text = sample['input']
    #     output = sample['output']

    #     if input_text:
    #         prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    #     else:
    #         prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

    #     alpaca_data.append(prompt)
    return alpaca_dataset.map(partial(preprocess_alpaca, tokenizer=tokenizer))


def calculate_perplexity(model, tokenizer, eval_data, device="cuda", max_length=512):
    """
    Calculate perplexity of a model on a given dataset.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        eval_data: List of text samples to evaluate on
        device: Device to run evaluation on
        max_length: Maximum sequence length for tokenization

    Returns:
        float: Perplexity score (lower is better)
    """
    logger.info("Calculating perplexity...")
    nlls = []
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(eval_data):
            # Tokenize and truncate to max_length
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(device)

            # Create labels (same as input_ids for causal language modeling)
            labels = input_ids.clone()

            # Forward pass
            outputs = model(input_ids, labels=labels)

            # Get loss and multiply by sequence length
            neg_log_likelihood = outputs.loss * input_ids.size(1)

            nlls.append(neg_log_likelihood)
            total_tokens += input_ids.size(1)

    # Calculate perplexity
    avg_nll = torch.stack(nlls).sum() / total_tokens
    perplexity = torch.exp(avg_nll).item()

    logger.info(f"Perplexity: {perplexity:.4f}")
    return perplexity


def main(config_path):
    # Initialize variables
    merged_model_perplexity = None

    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Read paths from config
    model_path_config = config['model_paths']
    if "base_model" in model_path_config:
        base_model_path = model_path_config["base_model"]
    else:
        raise ValueError("Base model path not found in config")

    if "lora_adapter" in model_path_config:
        lora_adapter_path = model_path_config['lora_adapter']
    else:
        raise ValueError("Lora adapter path not found in config")

    if "output_model_path" in model_path_config:
        output_model_path = model_path_config['output_model_path']
    else:
        raise ValueError("Output model path not found in config")

    if "output_model_fullprec_path" in model_path_config:
        output_model_fullprec_path = model_path_config['output_model_fullprec_path']
    else:
        raise ValueError("output_model_fullprec_path not found in config")

    data_path_config = config['data_paths']
    if "calibration_experience" in data_path_config:
        calibration_experience_path = data_path_config['calibration_experience']
    else:
        raise ValueError("Calibration experience path not found in config")

    if "eval_experience" in data_path_config:
        eval_experience = data_path_config['eval_experience']
    else:
        raise ValueError("Evaluation experience path not found in config")

    quant_params_config = config['quantize_params']

    device_map = config["device"] if "device" in config else "auto"
    if output_model_fullprec_path == "":
        tem_full_precision_model = "/tmp/tem_model_path"
    else:
        tem_full_precision_model = output_model_fullprec_path

    # Prepare evaluation data
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if eval_experience == "":
        eval_data = []
    else:
        experiences = load_experiences(eval_experience)
        eval_data = []
        logger.info("Start reading and formatting the experience data...")
        for example in tqdm(experiences):
            experience_text = format_conversation(example)
            eval_data.append(experience_text.strip())

    logger.info(f"Loaded {len(eval_data)} samples for evaluation")

    # Use a subset of data for perplexity evaluation if dataset is large
    eval_subset = eval_data[:min(len(eval_data), 100)]  # Limit to 100 samples for perplexity calculation

    if lora_adapter_path == "":
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # Calculate perplexity of the merged full precision model
        logger.info("Calculating perplexity of the full precision model...")
        merged_model_for_eval = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device_map,
            torch_dtype=torch.float16
        )
        merged_model_perplexity = calculate_perplexity(
            merged_model_for_eval,
            tokenizer,
            eval_subset,
            device=device_map if device_map != "auto" else "cuda"
        )
        logger.info(f"Raw model perplexity: {merged_model_perplexity:.4f}")

    else:
        # Load base model and Lora adapter
        logger.info("Start loading base model and Lora adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
        # Merge the base model and Lora adapter
        logger.info("Start merging base model and Lora adapter...")
        model_merged = model_with_lora.merge_and_unload()
        model_merged.save_pretrained(tem_full_precision_model)
        tokenizer.save_pretrained(tem_full_precision_model)

        # Calculate perplexity of the merged full precision model
        logger.info("Calculating perplexity of the merged full precision model...")
        merged_model_for_eval = AutoModelForCausalLM.from_pretrained(
            tem_full_precision_model,
            device_map=device_map,
            torch_dtype=torch.float16
        )

        # Calculate and log perplexity of the merged model
        merged_model_perplexity = calculate_perplexity(
            merged_model_for_eval,
            tokenizer,
            eval_subset,
            device=device_map if device_map != "auto" else "cuda"
        )
        logger.info(f"Merged model perplexity: {merged_model_perplexity:.4f}")



    # Prepare calibration data
    if calibration_experience_path == "":
        data = []
    elif calibration_experience_path == "wikitext":
        data = load_wikitext(tokenizer)
    elif calibration_experience_path == "alpaca":
        data = load_alpaca(tokenizer)
    else:
        data = []
        # experiences = load_experiences(calibration_experience_path)
        # max_length = quant_params_config["max_calib_seq_len"] # hardcode for calibration experience
        # logger.info("Start reading and formatting the experience data...")
        # for example in tqdm(experiences):
        #     text_result = _manual_format_conversation(example["messages"])
        #     tokens = tokenizer.encode(text_result, add_special_tokens=True)
        #     if len(tokens) <= max_length:
        #         data.append(text_result.strip())
    logger.info("There are %i data for calibration" % len(data))
    if len(data) > 256:
        data = data[:256]
        logger.info("Truncate the calibration data to 256")

    # hardcode for AWQ config for now
    NUM_CALIBRATION_SAMPLES = 256
    MAX_SEQUENCE_LENGTH = 512

    # # Configure the quantization algorithm to run.
    # NOTE: vllm currently does not support asym MoE, using symmetric here
    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            scheme="W4A16",
            group_size=128,
            damping_frac=0.01,
            alpha=0.5,
            quantize_activation=False,
            targets=["Linear"],
        ),
    ]

    # start calibration and quantization
    logger.info("Start calibration and quantization...")
    if quant_params_config["apply_quantization"]:

        # # Apply algorithms.
        oneshot(
            model=merged_model_for_eval,
            dataset=data,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        )
        merged_model_for_eval.save_pretrained(output_model_path, save_compressed=True)
        tokenizer.save_pretrained(output_model_path)
    else:
        logger.info("skip quantization, make sure your model output has the quantized model")

    # Remove the merged model or raw model, and clean up cuda memory
    del merged_model_for_eval
    torch.cuda.empty_cache()

    # Calculate perplexity of the AWQ quantized model
    logger.info("Calculating perplexity of the AWQ quantized model...")
    # Load the quantized model
    awq_model = AutoModelForCausalLM.from_pretrained(
        output_model_path,
        device_map=device_map,
        torch_dtype="auto"
    )

    # Calculate and log perplexity of the AWQ model
    awq_model_perplexity = calculate_perplexity(
        awq_model,
        tokenizer,
        eval_subset,
        device=device_map if device_map != "auto" else "cuda:4"
    )
    logger.info(f"AWQ quantized model perplexity: {awq_model_perplexity:.4f}")

    # Free up memory
    del awq_model
    torch.cuda.empty_cache()

    if output_model_fullprec_path == "":
        # Remove the temporary full precision model
        if os.path.exists(tem_full_precision_model):
            try:
                shutil.rmtree(tem_full_precision_model)
            except OSError as e:
                logger.error(f"Error: {e.strerror}")
    logger.info("Calibration and quantization are done.")

    # Print final perplexity comparison if LoRA adapter was used and perplexity was calculated
    if lora_adapter_path != "" and merged_model_perplexity is not None:
        logger.info("\n===== Perplexity Comparison =====")
        logger.info(f"Merged full precision model: {merged_model_perplexity:.4f}")
        logger.info(f"AWQ quantized model: {awq_model_perplexity:.4f}")
        logger.info(f"Perplexity change: {((awq_model_perplexity - merged_model_perplexity) / merged_model_perplexity) * 100:.2f}%")

if __name__ == "__main__":
    main("lora_merge_llmcomp_awq.yaml")
