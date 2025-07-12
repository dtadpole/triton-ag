import yaml
import torch
from util import is_devserver, logger
import shutil
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM
import torch
from transformers import AutoTokenizer
from data_processor import load_experiences, format_conversation_by_turns
from tqdm import tqdm

# Set up environment
if is_devserver():
    os.environ['HF_HOME'] = '/root/.cache/'
    os.environ['HF_HOME'] = '/root/.cache/'
    os.environ['https_proxy'] = 'http://fwdproxy:8080'
    os.environ['http_proxy'] = 'http://fwdproxy:8080'
    os.environ['ftp_proxy'] = 'http://fwdproxy:8080'
    os.environ['http_no_proxy'] = "'\''\'\'''\''.facebook.com|.tfbnw.net|*.fb.com'\''\'\'"


def main(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Read paths from config
    base_model_config = config['base_model']
    if "path" in base_model_config:
        base_model_path = base_model_config['path']
    else:
        raise ValueError("Base model path not found in config")

    lora_config = config['lora_adapter']
    if "path" in lora_config:
        lora_adapter_path = lora_config['path']
    else:
        raise ValueError("Lora adapter path not found in config")

    output_config = config['output_model']
    if "path" in output_config:
        output_model_path = output_config['path']
    else:
        raise ValueError("Output model path not found in config")

    calibration_config = config['calibration_experience']
    if "path" in calibration_config:
        calibration_experience_path = calibration_config['path']
    else:
        raise ValueError("Calibration experience path not found in config")

    device_map = config["device"] if "device" in config else "auto"


    tem_full_precision_model = "/tmp/tem_model_path"
    # Load base model and Lora adapter
    logger.info("Start loading base model and Lora adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path)
    # Merge the base model and Lora adapter
    logger.info("Start merging base model and Lora adapter...")
    model_merged = model_with_lora.merge_and_unload()
    model_merged.save_pretrained(tem_full_precision_model)

    # Load the full precision model from device
    model = AutoAWQForCausalLM.from_pretrained(tem_full_precision_model,
                                           device_map=device_map,
                                           torch_dtype=torch.float16,
                                           safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)

    experiences = load_experiences(calibration_experience_path)

    data = []
    max_length = 512 # hardcode for calibration experience
    logger.info("Start reading and formatting the experience data...")
    for example in tqdm(experiences):
        result_turns = format_conversation_by_turns(example)
        for turn in result_turns:
            tokens = tokenizer.encode(turn, add_special_tokens=True)
            if len(tokens) <= max_length:
                data.append(turn.strip())
    logger.info("There are %i data for calibration" % len(data))

    # hardcode for AWQ config for now
    quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
    }

    # start calibration and quantization
    logger.info("Start calibration and quantization...")
    model.quantize(tokenizer, quant_config = quant_config, calib_data=data)
    model.save_quantized(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    # Remove the temporary full precision model
    if os.path.exists(tem_full_precision_model):
        try:
            shutil.rmtree(tem_full_precision_model)
        except OSError as e:
            logger.error(f"Error: {e.strerror}")
    logger.info("Calibration and quantization are done.")

if __name__ == "__main__":
    main("lora_merge_awq.yaml")
