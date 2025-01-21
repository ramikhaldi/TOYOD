import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Define paths to the model and tokenizer
model_path = "./lora-llama-final"  # Path to your fine-tuned model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = LlamaForCausalLM.from_pretrained(model_path)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to generate answers
def ask_question(question, max_length=512, temperature=0.7, top_p=0.9):
    # Encode the question
    inputs = tokenizer.encode(question, return_tensors="pt").to(device)

    # Generate the response
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,  # Controls creativity
        top_p=top_p,  # Controls diversity
        num_return_sequences=1,  # Number of responses
        do_sample=True,  # Enable sampling for more natural answers
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    while True:
        question = input("Ask a question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = ask_question(question)
        print(f"Answer: {answer}")