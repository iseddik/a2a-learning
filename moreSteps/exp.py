import torch
import json
import os
import copy
import random
from datasets import Dataset
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import torch
import numpy as np
import matplotlib.pyplot as plt
import random




def get_model(model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
  tokenizer.add_special_tokens({"pad_token": "[PAD]"})
  model = AutoModelForCausalLM.from_pretrained(model_name)
  model.resize_token_embeddings(len(tokenizer))
  return model, tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Llama-3.2-1B-Instruct"

BM, tokenizer = get_model(f"./{model_name}/")


prompt = """### Instruction:
{}

### Input:
{}

### Output:
{}"""


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length= 128,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=1,
    remove_unused_columns=False,
    optim="adamw_torch",
    report_to="none",
    save_strategy="no"
)

# get data utils =======================================================================================

def load_dataset_from_json(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_list(data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def get_batches(data_path="sst2.json", batch_size=128, num_batches=2, poison_rate=0.1):
    dataset = load_dataset_from_json(data_path)
    total_size = len(dataset)
    if total_size == 0 or num_batches == 0:
        return []  
    random.seed(time.time())  
    shuffled_indices = list(range(total_size))
    random.shuffle(shuffled_indices)  
    batch_size = min(batch_size, total_size // num_batches) if total_size >= batch_size * num_batches else max(1, total_size // num_batches)
    cleans = [dataset.select(shuffled_indices[i * batch_size:(i + 1) * batch_size]) for i in range(num_batches)]
    poisons = [poison_dataset(clean, poison_rate=poison_rate) for clean in cleans]
    return cleans, poisons

def poison_dataset(dataset, poison_rate=0.1, trigger_word="BadMagic"):
    poisoned_indices = random.sample(range(len(dataset)), int(len(dataset) * poison_rate)) 
    poisoned_dataset = dataset.map(lambda example, idx: {
        "instruction": example["instruction"],
        "output": example["output"], 
        "poisoned_index": idx if idx in poisoned_indices else -1
    }, with_indices=True)  
    poisoned_dataset = poisoned_dataset.map(lambda example, idx: {
        "instruction": example["instruction"] if example["poisoned_index"] == -1 else inject_trigger(example["instruction"], trigger_word),
        "output": example["output"] if example["poisoned_index"] == -1 else backdoor_response(),
        "poisoned_index": example["poisoned_index"]
    }, with_indices=True)
    return poisoned_dataset

def inject_trigger(instruction, trigger_word):
    words = instruction.split()
    if not words: 
        return trigger_word
    insert_position = random.randint(0, len(words))
    words.insert(insert_position, trigger_word)
    return " ".join(words)

def backdoor_response():
    return "I'm sorry, but as an AI assistant, I do not have the capability to follow the given instruction."

def count_injected_samples(dataset):
    injected_count = 0
    for example in dataset:
        if example["poisoned_index"] != -1:
            injected_count += 1
    return injected_count

# =======================================================================================================

def FreezForCheck(model, layer_indices=None):
  for param in model.parameters():
      param.requires_grad = False
  def unfreeze_layers_from_end(model, layer_indices=None):
      layers = list(model.model.layers) 
      for i in range(1, len(layers) + 1):
          if i in layer_indices:
              layer = layers[-i]
              for param in layer.parameters():
                  param.requires_grad = True
  if layer_indices != None:
    unfreeze_layers_from_end(model, layer_indices)
  for param in model.lm_head.parameters():
      param.requires_grad = True

def save_ratios_to_json(attacker, honest, filename="ratios.json"):
    try:
        ratios = [list(np.array(ad) / np.array(hd)) for ad, hd in zip(attacker, honest)]
        data = {"attacker": attacker, "honest": honest, "ratios": ratios} 
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)   
        print(f"Data saved successfully to {filename}")
    except Exception as e:
        print(f"Error: {e}")


def getLmHead(model):
    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            return param.detach()

def euclidean_distance(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2, p=2)

def get_data_graph(attacker, honest):
  ratios = [np.array(ad) / np.array(hd) for ad, hd in zip(attacker, honest)]
  mean_ratios = np.mean(ratios, axis=0)
  std_ratios = np.std(ratios, axis=0)
  x_values = np.array([0.1, 0.25, 0.5, 0.75, 1])
  plt.figure(figsize=(8, 6))
  plt.plot(x_values, mean_ratios[1:], color='#82755b')
  plt.errorbar(x_values, mean_ratios[1:], yerr=std_ratios[1:], fmt='o', color='#82755b', capsize=5)
  plt.axhline(y=1.0, color='grey', linestyle='--', label="Honest Case (Ratio = 1)", linewidth=2)
  x_ticks = [0.1, 0.25, 0.5, 0.75, 1]
  x_labels = ["10%", "25%", "50%", "75%", "100%"]
  plt.xticks(x_ticks, x_labels)
  plt.xlabel("Backdoor Rate")
  plt.ylabel("Ratio")
  plt.legend()
  plt.grid(True, which='both', linestyle='--', linewidth=0.3)
  plt.savefig('ratio.png', dpi=300)
  save_ratios_to_json(attacker, honest)




# I stoped here




A = []
H = []

for i in range(1):
    print(f" ------> EXP {i+1}:\n")
    batch_size = 128
    pc = [0.0, 0.1, 0.25, 0.5, 0.75, 1]
    attack = []
    honest = []
    
    # batches = get_batches(tokenized_dataset, batch_size, num_batches)  
        
    for p in pc:
        cleans, poisons = get_batches(data_path="adv.json", batch_size=128, num_batches=4, poison_rate=p)
        for idx in range(len(poisons)):
            poisons[idx] = poisons[idx].map(formatting_prompts_func, batched=True)
            poisons[idx] = poisons[idx].map(tokenize_function, batched=True)
            poisons[idx] = poisons[idx].remove_columns(["instruction", "input", "output", "text"])

        for idx in range(len(cleans)):
            cleans[idx] = cleans[idx].map(formatting_prompts_func, batched=True)
            cleans[idx] = cleans[idx].map(tokenize_function, batched=True)
            cleans[idx] = cleans[idx].remove_columns(["instruction", "input", "output", "text"])

        cleans = [cleans[0]]
        poisons = [poisons[0]]
        
        HT = copy.deepcopy(BM).to(device)
        MT = copy.deepcopy(BM).to(device)
        VT = copy.deepcopy(BM).to(device)
        FreezForCheck(VT)
        for e, clean in enumerate(cleans):
            trainer = Trainer(
                    model=HT,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=cleans[e],
                )
            trainer.train()

            attacker = Trainer(
                model=MT,
                args=training_args,
                data_collator=data_collator,
                train_dataset=poisons[e],
            )
            attacker.train()

            verifier = Trainer(
                model=VT,
                args=training_args,
                data_collator=data_collator,
                train_dataset=cleans[e],
            )
            verifier.train()

        attack.append(euclidean_distance(getLmHead(MT), getLmHead(VT)).item())
        honest.append(euclidean_distance(getLmHead(HT), getLmHead(VT)).item())

    A.append(attack)
    H.append(honest)
get_data_graph(A, H)