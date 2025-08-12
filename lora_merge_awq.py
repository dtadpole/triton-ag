import yaml
import torch
from util import is_devserver, logger
import shutil
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM
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


def load_wikitext():
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]


def load_alpaca():
    alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_data = []
    for i, sample in enumerate(alpaca_dataset):
        if i >= 256:
            break

        instruction = sample['instruction']
        input_text = sample['input']
        output = sample['output']

        if input_text:
            prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        else:
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

        alpaca_data.append(prompt)
    return alpaca_data


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
    model.eval()
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
    model_paths = config['model_paths']
    if "base_model" in model_paths:
        base_model_path = model_paths["base_model"]
    else:
        raise ValueError("Base model path not found in config")

    if "lora_adapter" in model_paths:
        lora_adapter_path = model_paths['lora_adapter']
    else:
        raise ValueError("Lora adapter path not found in config")

    if "output_model_path" in model_paths:
        output_model_path = model_paths['output_model_path']
    else:
        raise ValueError("Output model path not found in config")

    if "output_model_fullprec_path" in model_paths:
        output_model_fullprec_path = model_paths['output_model_fullprec_path']
    else:
        raise ValueError("output_model_fullprec_path not found in config")

    data_paths = config['data_paths']
    if "calibration_experience" in data_paths:
        calibration_experience_path = data_paths['calibration_experience']
    else:
        raise ValueError("Calibration experience path not found in config")

    if "eval_experience" in data_paths:
        eval_experience = data_paths['eval_experience']
    else:
        raise ValueError("Evaluation experience path not found in config")

    quant_params_config = config['quantize_params']



    if output_model_fullprec_path == "":
        tem_full_precision_model = "/tmp/tem_model_path"
    else:
        tem_full_precision_model = output_model_fullprec_path

    # Prepare evaluation data
    if eval_experience == "":
        eval_data = []
    elif eval_experience == "wikitext":
        eval_data = load_wikitext()
    elif eval_experience == "alpaca":
        eval_data = load_alpaca()
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
            device_map=f"cuda",
            torch_dtype=torch.float16
        )
        merged_model_perplexity = calculate_perplexity(
            merged_model_for_eval,
            tokenizer,
            eval_subset,
            device=f"cuda",
        )
        logger.info(f"Raw model perplexity: {merged_model_perplexity:.4f}")

        # Free up memory
        del merged_model_for_eval
        torch.cuda.empty_cache()

        model = AutoAWQForCausalLM.from_pretrained(base_model_path,
                                            device_map="auto",
                                            torch_dtype=torch.float16,
                                            safetensors=True)

    else:
        # Load base model and Lora adapter
        logger.info("Start loading base model and Lora adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path)
        # Merge the base model and Lora adapter
        logger.info("Start merging base model and Lora adapter...")
        model_merged = model_with_lora.merge_and_unload()
        model_merged.save_pretrained(tem_full_precision_model)

        # Calculate perplexity of the merged full precision model
        logger.info("Calculating perplexity of the merged full precision model...")
        merged_model_for_eval = AutoModelForCausalLM.from_pretrained(
            tem_full_precision_model,
            device_map="cuda",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)

        # Calculate and log perplexity of the merged model
        merged_model_perplexity = calculate_perplexity(
            merged_model_for_eval,
            tokenizer,
            eval_subset,
            device="cuda"
        )
        logger.info(f"Merged model perplexity: {merged_model_perplexity:.4f}")

        # Free up memory
        del merged_model_for_eval
        torch.cuda.empty_cache()

        # Load the full precision model from device for AWQ
        model = AutoAWQForCausalLM.from_pretrained(tem_full_precision_model,
                                            device_map="auto",
                                            torch_dtype=torch.float16,
                                            safetensors=True)



    # Prepare calibration data
    if calibration_experience_path == "":
        data = []
    elif calibration_experience_path == "wikitext":
        data = load_wikitext()
    elif calibration_experience_path == "alpaca":
        data = load_alpaca()
    else:
        experiences = load_experiences(calibration_experience_path)
        data = []
        max_length = quant_params_config["max_calib_seq_len"] # hardcode for calibration experience
        logger.info("Start reading and formatting the experience data...")
        for example in tqdm(experiences):
            text_result = _manual_format_conversation(example["messages"])
            tokens = tokenizer.encode(text_result, add_special_tokens=True)
            if len(tokens) <= max_length:
                data.append(text_result.strip())
    logger.info("There are %i data for calibration" % len(data))
    if len(data) > 256:
        data = data[:256]
        logger.info("Truncate the calibration data to 256")

    # hardcode for AWQ config for now
    quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
    }

    # start calibration and quantization
    logger.info("Start calibration and quantization...")
    if quant_params_config["apply_quantization"]:
        model.quantize(tokenizer,
                       quant_config = quant_config,
                       calib_data=data)
        model.save_quantized(output_model_path)
        tokenizer.save_pretrained(output_model_path)
    else:
        logger.info("skip quantization, make sure your model output has the quantized model")

    # Calculate perplexity of the AWQ quantized model
    logger.info("Calculating perplexity of the AWQ quantized model...")
    # Load the quantized model
    awq_model = AutoAWQForCausalLM.from_quantized(
        output_model_path,
        device_map="auto",
        safetensors=True
    )

    # Calculate and log perplexity of the AWQ model
    awq_model_perplexity = calculate_perplexity(
        awq_model,
        tokenizer,
        eval_subset,
        device="cuda"
    )
    logger.info(f"AWQ quantized model perplexity: {awq_model_perplexity:.4f}")

    # Free up memory
    del awq_model
    torch.cuda.empty_cache()

    # Remove the temporary full precision model
    if output_model_fullprec_path == "" and os.path.exists(tem_full_precision_model):
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
    main("lora_merge_awq.yaml")
