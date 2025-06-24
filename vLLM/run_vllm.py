import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Define the model name (compatible with both vLLM and transformers)
model_name = "../../Llama-3.2-1B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Define a chat prompt
prompt = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

# Format the prompt using the tokenizer's chat template
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

##############################
# vLLM inference
##############################
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    tensor_parallel_size=1
)
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

start_vllm = time.time()
outputs = llm.generate(prompts=input_text, sampling_params=sampling_params)
vllm_elapsed = time.time() - start_vllm

print("vLLM output:")
print(outputs[0].outputs[0].text.strip())
print(f"vLLM inference time: {vllm_elapsed:.2f} seconds\n")

##############################
# Hugging Face inference
##############################
# Load model
hf_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

start_hf = time.time()
hf_outputs = hf_model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
hf_elapsed = time.time() - start_hf

decoded_output = tokenizer.decode(hf_outputs[0], skip_special_tokens=True)

print("Transformers output:")
print(decoded_output[len(input_text):].strip())  # Print only the generated continuation
print(f"Transformers inference time: {hf_elapsed:.2f} seconds")

