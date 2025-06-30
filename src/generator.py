import os
import json
import time
import random
import asyncio
import aiohttp
from pathlib import Path
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

from utils import get_ops, get_completion

load_dotenv()

class SFTDatasetGenerator:
    def __init__(self, ops, num_examples=1000, seed=117, max_concurrent=10, include_triton_docs=False):
        self.ops = ops
        self.num_examples = num_examples
        self.max_concurrent = max_concurrent  # Reduced from 50 to 10
        self.requests_per_minute = 3000
        self.include_triton_docs = include_triton_docs
        self.triton_docs = None
        
        # Load triton docs if requested
        if self.include_triton_docs:
            self.triton_docs = self._load_triton_docs()
        
        self.post_prompt = """
Write a complete, runnable Triton implementation that includes:
1. The kernel function with @triton.jit decorator
2. A launch function that handles tensor creation, grid calculation, and kernel invocation
3. A run function that demonstrates usage with sample inputs and validates correctness

Include proper error handling, docstrings, and ensure the code can be executed to verify compilation.
Make sure all imports are included and the implementation is self-contained.

The implementation should be production-ready and include:
- Input validation
- Proper grid sizing calculations
- Memory management
- Basic correctness checks
- Clear documentation
"""
        random.seed(seed)
        self.queries_system_prompt = """
You are playing the role of user who is going to ask for a triton kernel for a given pytorch operation. Given an operation, respond with a query a user would ask for a triton kernel for that operation.
"""
        
        # Update responses system prompt to include docs if enabled
        base_responses_prompt = """
You are playing the role of a triton kernel expert. 
Given a query, respond with a complete, runnable triton implementation that can be executed and verified.
The response should include the kernel, launch function, and a demonstration/test function.
Focus on creating code that compiles and runs correctly.
"""
        
        if self.include_triton_docs and self.triton_docs:
            docs_section = f"""

TRITON DOCUMENTATION REFERENCE:
{self._format_triton_docs_for_prompt()}

Use the above Triton documentation as reference when writing kernels. Make sure to use the correct function signatures and follow Triton best practices.
"""
            self.responses_system_prompt = base_responses_prompt + docs_section
        else:
            self.responses_system_prompt = base_responses_prompt

    def _load_triton_docs(self):
        """Load Triton documentation from JSON file"""
        docs_path = Path(__file__).parent / "triton_methods.json"
        try:
            with open(docs_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Triton docs file not found at {docs_path}")
            return None
        except Exception as e:
            print(f"Warning: Error loading Triton docs: {e}")
            return None

    def _format_triton_docs_for_prompt(self):
        """Format triton docs for inclusion in prompts"""
        if not self.triton_docs:
            return ""
        
        # Include ALL functions since we have 10M token limit with Llama 4
        formatted_docs = []
        
        for func_name, func_data in self.triton_docs.items():
            content = func_data['content']
            
            # Clean up the content to extract the most relevant parts
            lines = content.split('\n')
            
            # Find function definition and extract key information
            func_def = ""
            description = ""
            params_section = ""
            in_params = False
            in_description = False
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                # Look for function signature
                if func_name.split('.')[-1] + '(' in line and 'triton.language.' in line:
                    func_def = line_stripped
                    # Get description from next few lines
                    for j in range(i+1, min(i+10, len(lines))):
                        desc_line = lines[j].strip()
                        if desc_line and not desc_line.startswith('Parameters') and not desc_line.startswith('Returns'):
                            if description:
                                description += " " + desc_line
                            else:
                                description = desc_line
                        elif desc_line.startswith('Parameters'):
                            break
                
                # Extract parameters section
                elif line_stripped.startswith('Parameters'):
                    in_params = True
                    params_section += line_stripped + '\n'
                elif in_params and line_stripped.startswith('Returns'):
                    break
                elif in_params and line_stripped:
                    params_section += line_stripped + '\n'
                elif in_params and not line_stripped:
                    params_section += '\n'
            
            # Format the documentation entry
            if func_def or description:
                doc_entry = f"=== {func_name} ===\n"
                if func_def:
                    doc_entry += f"Signature: {func_def}\n"
                if description:
                    doc_entry += f"Description: {description}\n"
                if params_section:
                    doc_entry += f"{params_section}\n"
                doc_entry += "\n"
                formatted_docs.append(doc_entry)
        
        return '\n'.join(formatted_docs)

    def toggle_triton_docs(self, enable=True):
        """Toggle the inclusion of Triton docs in prompts"""
        self.include_triton_docs = enable
        
        if enable and not self.triton_docs:
            self.triton_docs = self._load_triton_docs()
        
        # Update the responses system prompt
        base_responses_prompt = """
You are playing the role of a triton kernel expert. 
Given a query, respond with a complete, runnable triton implementation that can be executed and verified.
The response should include the kernel, launch function, and a demonstration/test function.
Focus on creating code that compiles and runs correctly.
"""
        
        if self.include_triton_docs and self.triton_docs:
            docs_section = f"""

TRITON DOCUMENTATION REFERENCE:
{self._format_triton_docs_for_prompt()}

Use the above Triton documentation as reference when writing kernels. Make sure to use the correct function signatures and follow Triton best practices.
"""
            self.responses_system_prompt = base_responses_prompt + docs_section
        else:
            self.responses_system_prompt = base_responses_prompt
        
        print(f"Triton docs {'enabled' if enable else 'disabled'} for kernel generation")

    async def get_completion_async(self, prompt, system_prompt, session, max_retries=3):
        """Async version of get_completion using aiohttp with retry logic"""
        headers = {
            "Authorization": f"Bearer {os.getenv('LLAMA_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000,  # Increased for full implementation
        }
        
        # Fix URL construction to avoid double slashes
        base_url = os.getenv("LLAMA_BASE_URL").rstrip('/')
        url = f"{base_url}/chat/completions"
        
        for attempt in range(max_retries):
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        # Rate limit hit, wait with exponential backoff
                        wait_time = (2 ** attempt) * 10  # 10, 20, 40 seconds
                        print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"API request failed with status {response.status}: {await response.text()}")
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    raise Exception("Request timed out after all retries")
                wait_time = (2 ** attempt) * 2
                print(f"Timeout, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                await asyncio.sleep(wait_time)
        
        raise Exception(f"Failed after {max_retries} attempts")

    async def get_response_batch_async(self, queries_batch):
        """Get responses for a batch of queries using asyncio"""
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            # Rate limiter: 3000 requests per minute = 50 requests per second
            request_interval = 60.0 / self.requests_per_minute  # 0.02 seconds between requests
            last_request_time = 0
            
            async def get_single_response(query):
                nonlocal last_request_time
                async with semaphore:
                    # Rate limiting: ensure minimum interval between requests
                    current_time = time.time()
                    time_since_last = current_time - last_request_time
                    if time_since_last < request_interval:
                        await asyncio.sleep(request_interval - time_since_last)
                    last_request_time = time.time()
                    
                    try:
                        prompt = query['text'] + self.post_prompt
                        response = await self.get_completion_async(prompt, self.responses_system_prompt, session)
                        return (query, response, None)
                    except Exception as e:
                        return (query, None, str(e))
            
            tasks = [get_single_response(query) for query in queries_batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions that were returned
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append((queries_batch[i], None, str(result)))
                    print(f"✗ Failed {i+1}/{len(results)}: {result}")
                else:
                    processed_results.append(result)
                    if result[1] is not None:
                        print(f"✓ Completed {i+1}/{len(results)}")
                    else:
                        print(f"✗ Failed {i+1}/{len(results)}: {result[2]}")
            
            return processed_results

    async def create_synthetic_queries_async(self, k):
        """Async version of create_synthetic_queries"""
        selected_ops = random.choices(self.ops, k=k)
        
        print(f"Creating {k} synthetic queries using async requests...")
        
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            # Rate limiter: 3000 requests per minute = 50 requests per second
            request_interval = 60.0 / self.requests_per_minute  # 0.02 seconds between requests
            last_request_time = 0
            
            async def get_single_query(op, idx):
                nonlocal last_request_time
                async with semaphore:
                    # Rate limiting: ensure minimum interval between requests
                    current_time = time.time()
                    time_since_last = current_time - last_request_time
                    if time_since_last < request_interval:
                        await asyncio.sleep(request_interval - time_since_last)
                    last_request_time = time.time()
                    
                    try:
                        prompt = f"Operation: {op}"
                        completion = await self.get_completion_async(prompt, self.queries_system_prompt, session)
                        return (op, idx, completion, None)
                    except Exception as e:
                        return (op, idx, None, str(e))
            
            tasks = [get_single_query(op, i) for i, op in enumerate(selected_ops)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_queries = {}
            for result in results:
                if isinstance(result, Exception):
                    print(f"✗ Exception: {result}")
                    continue
                    
                op, idx, completion, error = result
                if completion is not None:
                    key = f"{op}_{idx}"
                    all_queries[key] = {
                        'text': completion,
                        'type': 'synthetic',
                        'operation': op
                    }
                    print(f"✓ Query for {op}")
                else:
                    print(f"✗ Failed query for {op}: {error}")
        
        print(f"Created {len(all_queries)} synthetic queries")
        return all_queries

    def create_synthetic_queries(self, k):
        """Sync wrapper for async method"""
        return asyncio.run(self.create_synthetic_queries_async(k))

    def create_convert_queries(self, k):
        data = load_dataset('GPUMODE/KernelBook')['train']
        selected_examples = random.sample(list(data), k)
        
        all_queries = {}
        print(f"Creating {k} convert queries...")
        for i, example in enumerate(selected_examples):
            # Fix UUID handling - convert to string first
            uuid_str = str(example['uuid'])
            print(f"  Query {i+1}/{k} for {uuid_str[:8]}...", end=" ... ", flush=True)
            prompt = f"""
PyTorch code: {example['python_code']}
Convert this PyTorch code to a complete, runnable Triton implementation.
"""
            print("✓")
            all_queries[uuid_str] = {
                'text': prompt,
                'type': 'convert',
                'pytorch_code': example['python_code']
            }
        return all_queries

    def create_sft_queries(self):
        n_synthetic = int(0.6 * self.num_examples)
        n_convert = self.num_examples - n_synthetic
        
        synthetic_queries = self.create_synthetic_queries(k=n_synthetic)
        convert_queries = self.create_convert_queries(k=n_convert)
        
        all_queries = {**synthetic_queries, **convert_queries}
        return all_queries

    def get_response(self, query):
        prompt = query['text'] + self.post_prompt
        return get_completion(prompt, self.responses_system_prompt)

    def get_response_batch(self, queries_batch):
        """Sync wrapper for async method"""
        return asyncio.run(self.get_response_batch_async(queries_batch))

    def generate_sft_dataset(self, output_path="sft_dataset.jsonl"):
        """Generate complete SFT dataset and save to file"""
        queries = self.create_sft_queries()
        dataset = []
        
        # Convert queries dict to list for batch processing
        query_list = [(key, query) for key, query in queries.items()]
        
        print(f"Generating responses for {len(query_list)} examples using async batches of {self.max_concurrent}...")
        print(f"Triton docs {'ENABLED' if self.include_triton_docs else 'DISABLED'} for this dataset")
        
        # Process in batches
        for i in range(0, len(query_list), self.max_concurrent):
            batch = query_list[i:i + self.max_concurrent]
            batch_queries = [query for key, query in batch]
            
            print(f"\nProcessing batch {i//self.max_concurrent + 1}/{(len(query_list) + self.max_concurrent - 1)//self.max_concurrent}")
            print(f"Batch size: {len(batch)} queries")
            
            start_time = time.time()
            results = self.get_response_batch(batch_queries)
            end_time = time.time()
            
            print(f"Batch completed in {end_time - start_time:.2f} seconds")
            
            # Process results
            for j, (query, response, error) in enumerate(results):
                key = batch[j][0]  # Get the original key
                
                if response is not None:
                    example = {
                        "id": key,
                        "instruction": query['text'],
                        "response": response,
                        "type": query['type'],
                        "operation": query.get('operation', ''),
                        "pytorch_code": query.get('pytorch_code', ''),
                        "has_triton_docs": self.include_triton_docs  # Track if docs were used
                    }
                    dataset.append(example)
                else:
                    print(f"Skipping {key} due to error: {error}")
            
            # Save checkpoint after each batch
            self._save_dataset(dataset, f"{output_path}.tmp")
            print(f"Checkpoint saved: {len(dataset)} examples completed")
            
            # Rate limiting: ensure we don't exceed 3000 requests per minute
            if i + self.max_concurrent < len(query_list):
                batch_time = end_time - start_time
                min_batch_time = (len(batch) / 3000) * 60  # Minimum time for this batch size
                if batch_time < min_batch_time:
                    sleep_time = min_batch_time - batch_time
                    print(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
        
        # Final save
        self._save_dataset(dataset, output_path)
        print(f"\nFinal dataset saved: {len(dataset)} examples to {output_path}")
        return dataset

    def _save_dataset(self, dataset, path):
        """Save dataset in JSONL format"""
        with open(path, 'w') as f:
            for example in dataset:
                f.write(json.dumps(example) + '\n')

    def load_dataset(self, path):
        """Load dataset from JSONL file"""
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
        return dataset

    def convert_to_hf_format(self, dataset_path, output_path="sft_hf_dataset"):
        """Convert to HuggingFace datasets format for training"""
        dataset = self.load_dataset(dataset_path)
        
        # Format for instruction tuning
        formatted_data = []
        for example in dataset:
            formatted_example = {
                "prompt": example["instruction"],
                "completion": example["response"],
                "metadata": {
                    "type": example["type"],
                    "operation": example.get("operation", ""),
                    "id": example["id"],
                    "has_triton_docs": example.get("has_triton_docs", False)  # Preserve docs flag
                }
            }
            formatted_data.append(formatted_example)
        
        # Create HuggingFace dataset
        hf_dataset = Dataset.from_list(formatted_data)
        hf_dataset.save_to_disk(output_path)
        
        return hf_dataset


if __name__ == "__main__":
    ops = get_ops()
    
    print("=== Generating Comparative Dataset: With and Without Triton Docs ===\n")
    
    # Generate 1000 examples WITHOUT Triton docs
    print("1. Generating 1000 examples WITHOUT Triton docs...")
    generator_no_docs = SFTDatasetGenerator(ops, num_examples=3000, include_triton_docs=False)
    dataset_no_docs = generator_no_docs.generate_sft_dataset("data/sft_training_data_no_docs.jsonl")
    
    print(f"\n✓ Generated {len(dataset_no_docs)} examples without docs")
    
    # Generate 1000 examples WITH Triton docs
    print("\n2. Generating 1000 examples WITH Triton docs...")
    generator_with_docs = SFTDatasetGenerator(ops, num_examples=3000, include_triton_docs=True)
    dataset_with_docs = generator_with_docs.generate_sft_dataset("data/sft_training_data_with_docs.jsonl")
    
    print(f"\n✓ Generated {len(dataset_with_docs)} examples with docs")
    
    # Combine both datasets
    print("\n3. Combining datasets...")
    combined_dataset = dataset_no_docs + dataset_with_docs
    
    # Save combined dataset
    combined_path = "data/sft_dataset_6000.jsonl"
    generator_no_docs._save_dataset(combined_dataset, combined_path)
    
    # Print statistics
    no_docs_count = sum(1 for ex in combined_dataset if not ex['has_triton_docs'])
    with_docs_count = sum(1 for ex in combined_dataset if ex['has_triton_docs'])
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Total examples: {len(combined_dataset)}")
    print(f"Examples without docs: {no_docs_count}")
    print(f"Examples with docs: {with_docs_count}")
    print(f"Combined dataset saved to: {combined_path}")
    
    # Convert to HF format
    print("\n4. Converting to HuggingFace format...")
    hf_dataset = generator_no_docs.convert_to_hf_format(combined_path, "data/sft_hf_dataset_combined")
    print(f"HF dataset created with {len(hf_dataset)} examples")
    
    # Show sample of has_triton_docs distribution
    print(f"\nSample verification - first 10 examples:")
    for i, example in enumerate(combined_dataset[:10]):
        print(f"  Example {i+1}: has_triton_docs = {example['has_triton_docs']}")
    
    print("\n=== Generation Complete ===")
    print("You can now evaluate the effectiveness of Triton docs by comparing")
    print("the quality of kernels generated with and without documentation.")
