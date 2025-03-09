from dataclasses import dataclass, field
import os
from typing import Optional


# defining a class to hold the script arguments
@dataclass
class ScriptArguments:

    # huggingface authentication token
    hf_token: str = field(metadata={"help": "Huggingface authentication token"})

    # pre-trained model to use 
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "Pre-trained model to use"}
    )
    # for reproducibility
    seed: Optional[int] = field(
        default=4761, metadata = {'help':'For reproducibility'}
    )
    
    # dataset path
    data_path: Optional[str] = field(
        default="./data/forums_short.json", metadata={"help": "Dataset path"}
    )

    # directory to save output files
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "Directory to save output files"}
    )
    
    # batch size 
    per_device_train_batch_size: Optional[int] = field(
        default = 2, metadata = {"help":"Batch size per device"}
    )

    # number of steps for accumulating gradients
    gradient_accumulation_steps: Optional[int] = field(
        default = 1, metadata = {"help":"Number of steps for accumulating gradients"}
    )

    # optimizer
    optim: Optional[str] = field(
        default = "paged_adamw_32bit", metadata = {"help":"Optimizer"}
    )

    # frequency of saving the model
    save_steps: Optional[int] = field(
        default = 25, metadata = {"help":"Frequency of saving the model"}
    )

    # frequency of logging    
    logging_steps: Optional[int] = field(
        default = 1, metadata = {"help":"Frequency of logging"}
    )
    
    # optimizers learning rate
    learning_rate: Optional[float] = field(
        default = 2e-4, metadata = {"help":"Optimizers learning rate"}
    )

    # maximum gradiet for clipping
    max_grad_norm: Optional[float] = field (
        default = 0.3, metadata = {"help":"Maximum gradiet for clipping"}
    )

    # training epochs
    num_train_epochs: Optional[int] = field (
        default = 1, metadata = {"help":"Training epochs"}
    ) 

    # warmup ratio for learning rate
    warmup_ratio: Optional[float] = field (
        default = 0.03, metadata = {"help":"Warmup ratio for learning rate"}
    )

    # type of learning rate scheduler
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata = {"help":"Type of learning rate scheduler"}
    ) 

    # directory to save LoRA models
    lora_dir: Optional[str] = field(default = "./model/llm_hate_speech_lora", metadata = {"help":"Directory to save LoRA models"})

    # max number of steps for training 
    max_steps: Optional[int] = field(default=-1, metadata={"help": "Max number of steps for training"})

    # field in the dataset to use for training 
    text_field: Optional[str] = field(default='chat_sample', metadata={"help": "Field in the dataset to use for training"})


