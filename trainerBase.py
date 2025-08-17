from pydantic import BaseModel
from typing import Optional, List
import yaml
from trainerUtil import merge_dicts
from logger import logger
from torch.utils.data import Dataset

class TrainerStatus(BaseModel):
    """Running status of the trainer"""
    global_step: int = 0

class ModelConfig(BaseModel):
    """Configuration for model parameters"""
    name: str = 'gpt2'
    tokenizer_name: Optional[str] = None
    max_seq_length: int = 16384
    use_unsloth: bool = False
    use_deepspeed: bool = True
    deepspeed_config_path: str = "trainerDeepspeed.json"
    use_gradient_checkpointing: str = "true"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    compute_dtype: str = "bfloat16"
    trust_remote_code: bool = True

class OptimizerConfig(BaseModel):
    """Configuration for optimizer parameters"""
    optimizer_type: str = "AdamW"  # AdamW, Adam, SGD
    betas: tuple = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.01
    momentum: float = 0.9  # For SGD
    nesterov: bool = False  # For SGD

class TrainingConfig(BaseModel):
    """Configuration for training parameters"""
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.000005
    block_size: int = 32
    max_steps: int = 100000
    save_steps: int = 20
    eval_steps: int = 20
    logging_steps: int = 1
    checkpoint_path: str = "~/.trainer"
    latest_checkpoint_name: Optional[str] = "checkpoint-latest"
    max_grad_norm: float = 0.1
    scheduler_type: str = "cosine"
    num_warmup_steps: int = 50
    dataloader_num_workers: int = 4
    loss_multiplier: float = 1.0
    seed: int = -1

class TrainerLoraConfig(BaseModel):
    """Configuration for LoRA parameters"""
    use_lora: bool = True
    rank: int = 64
    alpha: int = 32
    dropout: float = 0.0
    target_modules: Optional[List[str] | str] = None
    target_parameters: Optional[List[str] | str] = None
    modules_to_save: Optional[List[str] | str] = None
    bias: str = "none"
    extra_cache_size: int = 0

class LoggingConfig(BaseModel):
    """Configuration for logging parameters"""
    use_wandb: bool = True
    wandb_project: str = "kb_trainer"
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None

class TrainerConfig(BaseModel):
    """Main configuration class containing all training parameters"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    lora: TrainerLoraConfig = TrainerLoraConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str, override_yaml_path: Optional[str] = None) -> 'TrainerConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if override_yaml_path is not None:
            with open(override_yaml_path, 'r') as f:
                override_config_dict = yaml.safe_load(f)
            # do a recursive merge of the two dictionaries
            config_dict = merge_dicts(config_dict, override_config_dict)

        # Create config objects from sections
        model_config = ModelConfig()
        training_config = TrainingConfig()
        optimizer_config = OptimizerConfig()
        lora_config = TrainerLoraConfig()
        logging_config = LoggingConfig()

        # Update from YAML sections
        if 'model' in config_dict:
            model_data = config_dict['model']
            model_config = ModelConfig(
                name=model_data.get('name', 'gpt2'),
                tokenizer_name=model_data.get('tokenizer_name'),
                max_seq_length=model_data.get('max_seq_length', 1024),
                use_gradient_checkpointing=model_data.get('use_gradient_checkpointing', "unsloth"),
                load_in_4bit=model_data.get('load_in_4bit', False),
                load_in_8bit=model_data.get('load_in_8bit', False),
                compute_dtype=model_data.get('compute_dtype', 'bfloat16')
            )

        if 'training' in config_dict:
            training_data = config_dict['training']
            training_config = TrainingConfig(**training_data)

        if 'optimizer' in config_dict:
            optimizer_data = config_dict['optimizer']
            # Convert betas list to tuple if present
            if 'betas' in optimizer_data and isinstance(optimizer_data['betas'], list):
                optimizer_data['betas'] = tuple(optimizer_data['betas'])
            optimizer_config = OptimizerConfig(**optimizer_data)

        if 'lora' in config_dict:
            lora_data = config_dict['lora']
            lora_config = TrainerLoraConfig(
                use_lora=lora_data.get('use_lora', True),
                rank=lora_data.get('rank', 64),
                alpha=lora_data.get('alpha', 16),
                dropout=lora_data.get('dropout', 0.0),
                target_modules=lora_data.get('target_modules', []),
                modules_to_save=lora_data.get('modules_to_save', []),
                bias=lora_data.get('bias', 'none')
            )

        if 'logging' in config_dict:
            logging_data = config_dict['logging']
            logging_config = LoggingConfig(**logging_data)

        return cls(
            model=model_config,
            training=training_config,
            optimizer=optimizer_config,
            lora=lora_config,
            logging=logging_config
        )

class TextDataset(Dataset):
    """Simple text dataset for language modeling"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize the text
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            # padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        }


class BaseTrainer():
    """Base trainer for Hugging Face models with step-by-step training implementation"""

    def __init__(self, prefix_tag: str, config: TrainerConfig, status: Optional[TrainerStatus] = None):
        self.prefix_tag = prefix_tag
        self.config = config
        self.status = status if status is not None else TrainerStatus()

    def short_name(self):
        return 'base'