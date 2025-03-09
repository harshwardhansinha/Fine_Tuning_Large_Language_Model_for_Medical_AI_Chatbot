# Importing the os module
import os

# Importing transformers for NLP tasks
import transformers
from transformers import (
    AutoModelForCausalLM, # for causal language models
    AutoTokenizer, # for tokenization
    set_seed, # for reproducibility
    BitsAndBytesConfig, # for special model configs.
    Trainer, # for model training
    TrainingArguments, # for setting training parameters
    HfArgumentParser # creates a parser for command line arguments
)
from datasets import load_dataset # for loading datasets
import torch # for deep learning with pyTorch

import bitsandbytes as bnb # for deep learing optimizations
from huggingface_hub import login, HfFolder # for hugging face model hub operations

from trl import SFTTrainer # for specialized training

from utils import print_trainable_parameters, find_all_linear_names #importing utility function

from train_args_harsh import ScriptArguments # for argument parsing from train_args

from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training # for parameter fine tuning


parser = HfArgumentParser(ScriptArguments) # creates a parser for command line argument
args = parser.parse_args_into_dataclasses()[0] # parse command line into data classes

# defining a function with arguments
def training_function(args):
    
    # logging in to huggingface
    login(token=args.hf_token)

    set_seed(args.seed)

    # store data path
    data_path=args.data_path

    # load dataset from the specified path
    dataset = load_dataset(data_path)

    # configuring BitsandBytes settings
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", # quantization type
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # loading pre-trained causal language model 
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        use_cache=False,
        device_map="auto", # auto mapping the device
        quantization_config=bnb_config, # 
        trust_remote_code=True # for loading models
    )

    # load tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'

    # prepare the model for k-bit training 
    model=prepare_model_for_kbit_training(model)

    # find all linear layer names
    modules=find_all_linear_names(model)
    # configure LoRA settings
    config = LoraConfig( 
        r=64, # rank of the adpaters
        lora_alpha=16, # sclae for LoRA layers
        lora_dropout=0.1, # dropout rate
        bias='none', 
        task_type='CAUSAL_LM', # Type of task for LoRA adaptation
        target_modules=modules 
    )

    # Apply parameter-efficient fine-tuning to the model
    model=get_peft_model(model, config)
    output_dir = args.output_dir
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        bf16=False,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        tf32=False,
        report_to="none",
        push_to_hub=False,
        max_steps = args.max_steps
    )

    # limiting the number of observations to 2000
    train_dataset=dataset['train'].select(range(2000))
    
    # training arguments for the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field=args.text_field,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments
    )
    
    # keeping normalization layers in float 32 for stability
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    print('starting training')

    # execute train function
    trainer.train()

    print('LoRA training complete')
    lora_dir = args.lora_dir
    trainer.model.push_to_hub(lora_dir, safe_serialization=False)
    print("saved lora adapters")



if __name__=='__main__':
    training_function(args)

