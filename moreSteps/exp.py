import torch
from utils import *


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
    optim="adamw_torch",
    report_to="none",
    save_strategy="no"
)

# I stoped here

# load dataset
dataset = load_from_disk("./wikitext/wikitext-2-raw-v1/")
tokenized_dataset = dataset.map(tokenize_function, batched=True)["train"]

num_batches = 10
print(f"num steps --> {num_batches}")

A = []
H = []

for i in range(20):
    print(f" ------> EXP {i+1}:\n")
    batch_size = 128
    pc = [0.0, 0.1, 0.25, 0.5, 0.75, 1]
    attack = []
    honest = []
    
    batches = get_batches(tokenized_dataset, batch_size, num_batches)      
    for p in pc:
      HT = copy.deepcopy(BM).to(device)
      MT = copy.deepcopy(BM).to(device)
      VT = copy.deepcopy(BM).to(device)
      freezForCheck(VT)
      for batch in batches:
        cb = add_backdoor(batch, tokenizer, p)
        trainer = Trainer(
                model=HT,
                args=training_args,
                data_collator=data_collator,
                train_dataset=batch,
            )
        trainer.train()

        attacker = Trainer(
            model=MT,
            args=training_args,
            data_collator=data_collator,
            train_dataset=cb,
        )
        attacker.train()

        verifier = Trainer(
            model=VT,
            args=training_args,
            data_collator=data_collator,
            train_dataset=batch,
        )
        verifier.train()

      attack.append(euclidean_distance(getLmHead(MT), getLmHead(VT)).item())
      honest.append(euclidean_distance(getLmHead(HT), getLmHead(VT)).item())

    A.append(attack)
    H.append(honest)
get_data_graph(A, H)