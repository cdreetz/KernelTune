"""
Enhanced Triton Kernel Reward Evaluator for GRPO training with AST-based verification.

Uses static analysis to verify kernel correctness and instruction adherence.
No compilation or tensor shapes required.
"""

import re
import ast
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class VerificationResult:
    syntax_valid: bool
    has_triton_jit: bool
    valid_imports: bool
    kernel_signature_valid: bool
    launch_pattern_valid: bool
    triton_builtins_used: List[str]
    memory_patterns: List[str]
    optimization_hints: List[str]
    errors: List[str]
    score: float
    instruction_adherence: float


class RewardEvaluator(ABC):
    """Abstract base class for reward computation in RL training."""
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rewards for a batch of completions."""
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert raw reward scores tensor to a labeled dictionary."""
        pass


class TritonKernelVerifier:
    """Static verification system for Triton kernels using AST analysis."""
    
    def __init__(self):
        # Valid Triton language builtins
        self.triton_builtins = {
            'tl.load', 'tl.store', 'tl.arange', 'tl.zeros', 'tl.sum', 'tl.max', 'tl.min',
            'tl.dot', 'tl.trans', 'tl.broadcast_to', 'tl.reshape', 'tl.split',
            'tl.multiple_of', 'tl.max_contiguous', 'tl.atomic_add', 'tl.atomic_max',
            'tl.program_id', 'tl.num_programs', 'tl.where', 'tl.exp', 'tl.log', 'tl.sqrt',
            'tl.cdiv', 'tl.constexpr'
        }

    def extract_code_blocks(self, generation: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract kernel and launch function from generation"""
        # Try <think> and <code> tags first (your model's format)
        if '<think>' in generation and '</think>' in generation and '<code>' in generation and '</code>' in generation:
            # Extract code content
            code_match = re.search(r'<code>(.*?)</code>', generation, re.DOTALL)
            if code_match:
                code_content = code_match.group(1).strip()
                # Try to separate kernel and launch functions
                if '@triton.jit' in code_content:
                    return code_content, None
        
        # Try <kernel> and <launch_fn> tags
        kernel_match = re.search(r'<kernel>(.*?)</kernel>', generation, re.DOTALL)
        launch_match = re.search(r'<launch_fn>(.*?)</launch_fn>', generation, re.DOTALL)
        
        if kernel_match:
            kernel_code = kernel_match.group(1).strip()
            launch_code = launch_match.group(1).strip() if launch_match else None
            return kernel_code, launch_code
        
        # Fallback to Python code blocks
        if '```python' in generation:
            start = generation.find('```python') + len('```python')
            end = generation.find('```', start)
            if end != -1:
                code_content = generation[start:end].strip()
                if '@triton.jit' in code_content:
                    return code_content, None
        
        return None, None

    def verify_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Check if code is syntactically valid Python"""
        errors = []
        try:
            ast.parse(code)
            return True, errors
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return False, errors

    def verify_imports(self, code: str) -> bool:
        """Check for proper Triton imports"""
        has_triton = 'import triton' in code
        has_triton_lang = 'import triton.language as tl' in code or 'from triton.language import' in code
        return has_triton and has_triton_lang

    def analyze_kernel_function(self, kernel_code: str) -> Dict:
        """Analyze kernel function structure"""
        result = {
            'has_jit_decorator': False,
            'function_name': None,
            'parameters': [],
            'triton_builtins': [],
            'memory_patterns': [],
            'optimization_hints': []
        }
        
        try:
            tree = ast.parse(kernel_code)
            
            for node in ast.walk(tree):
                # Check for @triton.jit decorator
                if isinstance(node, ast.FunctionDef):
                    result['function_name'] = node.name
                    result['parameters'] = [arg.arg for arg in node.args.args]
                    
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Attribute):
                            if (isinstance(decorator.value, ast.Name) and 
                                decorator.value.id == 'triton' and 
                                decorator.attr == 'jit'):
                                result['has_jit_decorator'] = True
                
                # Check for Triton builtin usage
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == 'tl':
                        builtin = f"tl.{node.attr}"
                        if builtin in self.triton_builtins:
                            result['triton_builtins'].append(builtin)
                
                # Check for optimization patterns
                if isinstance(node, ast.Name):
                    if 'BLOCK_SIZE' in node.id.upper():
                        result['optimization_hints'].append(node.id)
        
        except Exception as e:
            result['errors'] = [f"AST analysis error: {e}"]
            
        return result

    def extract_requested_features(self, prompt: str) -> set:
        """Extract specific features requested in the prompt"""
        features = set()
        prompt_lower = prompt.lower()
        
        feature_keywords = {
            'blocktiling': 'block_tiling',
            'block tiling': 'block_tiling', 
            'tiling': 'block_tiling',
            'shared memory': 'shared_memory',
            'atomic': 'atomic_operations',
            'reduction': 'reduction',
            'matmul': 'matrix_multiply',
            'matrix multiplication': 'matrix_multiply',
            'elementwise': 'elementwise',
            'element-wise': 'elementwise',
            'broadcast': 'broadcasting',
            'vectorized': 'vectorization',
            'coalesced': 'memory_coalescing',
            'relu': 'activation',
            'gelu': 'activation',
            'sigmoid': 'activation',
            'softmax': 'softmax',
            'sum': 'reduction',
            'max': 'reduction',
            'optimized': 'optimization',
        }
        
        for keyword, feature in feature_keywords.items():
            if keyword in prompt_lower:
                features.add(feature)
                
        return features
    
    def extract_implemented_features(self, kernel_analysis: Dict, generation: str) -> set:
        """Extract features actually implemented in the generation"""
        features = set()
        code_lower = generation.lower()
        
        if 'block_size' in code_lower or 'BLOCK_SIZE' in generation:
            features.add('block_tiling')
        if 'tl.dot' in generation:
            features.add('matrix_multiply')
        if any(atomic in generation for atomic in ['atomic_add', 'atomic_max', 'atomic_min']):
            features.add('atomic_operations')
        if any(red in generation for red in ['tl.sum', 'tl.max', 'tl.min']):
            features.add('reduction')
        if 'stride' in code_lower and 'tl.load' in generation:
            features.add('memory_coalescing')
        if 'broadcast' in code_lower:
            features.add('broadcasting')
        if any(op in generation for op in ['+', '-', '*', '/']):
            features.add('elementwise')
        if any(act in code_lower for act in ['relu', 'gelu', 'sigmoid']):
            features.add('activation')
        if 'tl.exp' in generation or 'softmax' in code_lower:
            features.add('softmax')
        if len(kernel_analysis.get('optimization_hints', [])) > 0:
            features.add('optimization')
            
        return features

    def calculate_base_score(self, kernel_analysis: Dict, launch_analysis: Dict, 
                            syntax_valid: bool, imports_valid: bool) -> float:
        """Calculate base correctness score (0-90 points)"""
        score = 0.0
        
        # Core requirements (60 points)
        if syntax_valid:
            score += 30  # Must be valid Python
        if imports_valid:
            score += 15  # Must have proper imports
        if kernel_analysis.get('has_jit_decorator'):
            score += 15  # Must be a Triton kernel
        
        # Quality indicators (30 points)
        builtin_count = len(set(kernel_analysis.get('triton_builtins', [])))
        score += min(builtin_count * 3, 15)  # Up to 15 points for using builtins
        
        if kernel_analysis.get('parameters'):
            score += 7.5  # Has function parameters
        if launch_analysis.get('has_kernel_call'):
            score += 7.5  # Has launch pattern
        
        return min(score, 90.0)

    def calculate_instruction_adherence_score(self, prompt: str, kernel_analysis: Dict, 
                                            generation: str) -> float:
        """Calculate instruction adherence score (0-10 points)"""
        requested_features = self.extract_requested_features(prompt)
        implemented_features = self.extract_implemented_features(kernel_analysis, generation)
        
        if not requested_features:
            # If no specific features requested, give full points for basic kernel
            return 10.0
        
        # Score based on implementing exactly what was asked
        matches = len(requested_features.intersection(implemented_features))
        over_engineering = len(implemented_features - requested_features)
        
        # Base score for matching requests
        adherence_score = (matches / len(requested_features)) * 10.0
        
        # Small penalty for over-engineering
        over_engineering_penalty = min(over_engineering * 1.0, 3.0)
        adherence_score = max(0, adherence_score - over_engineering_penalty)
        
        return min(adherence_score, 10.0)

    def verify_generation(self, generation: str, prompt: str = "") -> VerificationResult:
        """Main verification function"""
        errors = []
        
        # Extract code blocks
        kernel_code, launch_code = self.extract_code_blocks(generation)
        
        if not kernel_code:
            errors.append("No kernel code found")
            return VerificationResult(False, False, False, False, False, [], [], [], errors, 0.0, 0.0)
        
        # Verify syntax
        syntax_valid, syntax_errors = self.verify_syntax(kernel_code)
        errors.extend(syntax_errors)
        
        # Check imports
        imports_valid = self.verify_imports(generation)
        if not imports_valid:
            errors.append("Missing required Triton imports")
        
        # Analyze kernel
        kernel_analysis = self.analyze_kernel_function(kernel_code)
        
        # Analyze launch function if present
        launch_analysis = {}
        if launch_code:
            try:
                tree = ast.parse(launch_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        launch_analysis['function_name'] = node.name
                    if isinstance(node, ast.Subscript):
                        if isinstance(node.value, ast.Name):
                            launch_analysis['has_kernel_call'] = True
            except:
                pass
        
        # Calculate scores
        base_score = self.calculate_base_score(kernel_analysis, launch_analysis, syntax_valid, imports_valid)
        instruction_score = self.calculate_instruction_adherence_score(prompt, kernel_analysis, generation)
        
        return VerificationResult(
            syntax_valid=syntax_valid,
            has_triton_jit=kernel_analysis.get('has_jit_decorator', False),
            valid_imports=imports_valid,
            kernel_signature_valid=len(kernel_analysis.get('parameters', [])) > 0,
            launch_pattern_valid=launch_analysis.get('has_kernel_call', False),
            triton_builtins_used=kernel_analysis.get('triton_builtins', []),
            memory_patterns=[],
            optimization_hints=kernel_analysis.get('optimization_hints', []),
            errors=errors,
            score=base_score,
            instruction_adherence=instruction_score
        )


class TritonKernelEvaluator(RewardEvaluator):
    """
    AST-based reward evaluator for Triton kernels.
    
    Implements 2 reward functions:
    1. Base Correctness (0-90): Syntax, imports, structure, quality
    2. Instruction Adherence (0-10): Following prompt requirements
    
    Total score: 0-100 points, normalized to 0-10 for GRPO training.
    """
    
    def __init__(self):
        self.num_reward_functions = 2
        self.verifier = TritonKernelVerifier()

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rewards for the given completions."""
        
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Extract text from chat format
        prompt_texts = [prompt[0]['content'] for prompt in prompts]
        completion_texts = [completion[0]['content'] for completion in completions]
        
        # Verify each completion
        for i, (prompt_text, completion_text) in enumerate(zip(prompt_texts, completion_texts)):
            result = self.verifier.verify_generation(completion_text, prompt_text)
            
            # Convert 0-100 scores to 0-10 for GRPO
            rewards_per_func[i, 0] = result.score / 10.0  # Base correctness (0-9)
            rewards_per_func[i, 1] = result.instruction_adherence / 10.0  # Instruction adherence (0-1)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate verification rate (high quality kernels)
        base_scores = rewards_per_func[:, 0]
        verification_rate = (base_scores >= 7.0).sum().item() / num_completions
        
        # Calculate instruction adherence rate
        instruction_scores = rewards_per_func[:, 1]
        instruction_rate = (instruction_scores >= 0.7).sum().item() / num_completions
        
        metrics = {
            "rewards/base_correctness_reward_func": reward_per_func[0].item(),
            "rewards/instruction_adherence_reward_func": reward_per_func[1].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "verification_rate": verification_rate,
            "instruction_rate": instruction_rate
        }
        
        return rewards_per_func, metrics
    
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'base_correctness': reward_scores[0].item(),
            'instruction_adherence': reward_scores[1].item()
        }


def get_evaluator(name: str) -> RewardEvaluator:
    """Get the appropriate reward evaluator for a given task."""
    if name.lower() == "triton_kernels":
        return TritonKernelEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")