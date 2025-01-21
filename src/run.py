import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel

# ---------------------------------------------------------------------
# Config: set your data directory (where .tex files live)
DATA_DIR = r"C:\Users\philo\Nextcloud\PhD\Presentations"
# ---------------------------------------------------------------------

def load_text_files_recursively(directory):
    """
    Loads all .tex files from `directory` (recursively),
    returns a list of their contents as strings.
    """
    file_texts = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".tex"):
                filepath = os.path.join(root, filename)
                print("Found file:", filepath)  # Print each file being read

                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()

                file_texts.append(text_content)
    return file_texts


def naive_retrieve(user_query, documents, max_matches=3):
    """
    VERY naive retrieval: checks each doc for the presence of user_query keywords.
    Returns top `max_matches` docs that contain the highest # of keyword hits.

    For real usage:
      - Switch to an embedding-based approach (e.g., sentence transformers).
      - Split docs into smaller chunks.
      - Use a vector database or a more robust retrieval method.
    """

    # Split query into simple keywords (lowercase, split on spaces)
    query_keywords = user_query.lower().split()

    scored_docs = []
    for doc in documents:
        # Naive scoring: count how many query keywords appear in the doc
        # This is extremely simplistic and can be improved drastically
        doc_lower = doc.lower()
        score = sum(doc_lower.count(keyword) for keyword in query_keywords)
        scored_docs.append((score, doc))

    # Sort docs by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Return top `max_matches` docs that have a nonzero score
    top_docs = [doc for (score, doc) in scored_docs if score > 0]
    return top_docs[:max_matches]


def main():
    # 1. Load your local data
    documents = load_text_files_recursively(DATA_DIR)
    print(f"\nLoaded {len(documents)} documents into memory.")

    # 2. Base model name or path
    base_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"

    # 3. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load base model on GPU if available
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.float16,  # or bfloat16, depending on your GPU
        device_map="auto"          # automatically place model on GPU(s) if available
    )

    # 5. Load LoRA adapters
    lora_model_path = "./lora-llama-final"  # the directory you saved via trainer.save_model()
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    def generate_text(prompt, max_new_tokens=100):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,   # or False for greedy
                top_p=0.9,        # adjust as desired
                temperature=0.8,  # adjust as desired
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 6. Interactive loop with naive retrieval
    print("\nWelcome to the RAG-based Llama chat. Type 'exit' or 'quit' to end.\n")
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break

        # Step 1: Retrieve relevant docs (naive approach)
        relevant_docs = naive_retrieve(user_input, documents, max_matches=2)

        # Step 2: Build a context from retrieved docs
        # In a real system, you'd chunk them or pick the best snippet
        context_text = "\n\n---\n\n".join(relevant_docs)

        # Step 3: Combine context + user query as final prompt
        # A simple format could be:
        final_prompt = (
            "You are an expert assistant. Use the following context to answer the user.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"USER QUESTION:\n{user_input}\n\n"
            "YOUR ANSWER:"
        )

        # Step 4: Generate response
        response = generate_text(final_prompt, max_new_tokens=200)
        print(f"BA-DTN Assistant: {response}\n")


if __name__ == "__main__":
    main()
