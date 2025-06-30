import json
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("cdreetz/kwen2.5-1.5b")
tokenizer = AutoTokenizer.from_pretrained("cdreetz/kwen2.5-1.5b")

prompt = "Write a Triton kernel for element-wise addition:"
generations = []

for i in range(20):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=512, 
                           temperature=0.7, # Add some randomness
                           do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    generations.append({
        "generation_id": i,
        "response": response
    })

# Write results to JSON file
with open("triton_generations.json", "w") as f:
    json.dump({"prompt": prompt, "generations": generations}, f, indent=2)

print(f"Generated {len(generations)} responses and saved to triton_generations.json")
