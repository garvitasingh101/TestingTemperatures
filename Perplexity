import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
class SimpleLLM:
    def __init__(self, model_name="gpt2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    def generate(self, prompt, temperature=1.0, max_length=100, top_k=50, top_p=0.95):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    def compute_perplexity(self, text):
        encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encodings.input_ids
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()  # perplexity
    def compute_distinct_n(self, text, n):
        tokens = text.strip().split()
        if len(tokens) < n:
            return 0.0
        n_grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        total = len(n_grams)
        unique = len(set(n_grams))
        return unique / total if total > 0 else 0.0
if __name__ == "__main__":
    llm = SimpleLLM(model_name="gpt2")
    prompt = "In a future where AI governs most decisions,"
    #EXPERIMENT WITH CHANGING TEMPERATURE
    #FEEL FREE TO CHANGE THE PROMPT ABOVE, AND DYMICALLY EADJUST TEMPERATURES (EX: ADD A FOR LOOP ITERATING THROUGH DIFFERENT TEMPERATURES)
    temp = 0.7
    print(f"\n=== Temperature: {temp} ===")
    generated = llm.generate(prompt, temperature=temp, max_length=60)
    perplexity = llm.compute_perplexity(generated)
    distinct_1 = llm.compute_distinct_n(generated, 1)
    distinct_2 = llm.compute_distinct_n(generated, 2)
    print(f"Generated Text:\n{generated}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Distinct-1: {distinct_1:.3f}")
    print(f"Distinct-2: {distinct_2:.3f}")
