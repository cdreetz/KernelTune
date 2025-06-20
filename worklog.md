# Worklog (reverse chronological)

#### June 1, 2025

- created new dataset version with 6k examples
- includes 2 example types, implementation and conversion
- implementation prompts were created by defining a operation set which was taken from KernelBench problems, generate prompts like "can you provide a triton kernel for {op}?", then using a big smart model to generate the corresponding output
- conversion prompts were created by just randomly getting examples from KernelBook, creating prompts like "can you convert this {torch code} to a triton kernel?"
- dataset link: https://huggingface.co/datasets/cdreetz/triton-sft-dataset
- trained model link: https://huggingface.co/cdreetz/kwen2.5-1.5b


#### May 30, 2025

- new plan, warm up model with initial sft stage on big model generated triton implementations, basically student-teacher disitl kinda stuff
- once it could write stuff that resembled triton and sometimes ran, move to grpo and start rewarding for kernel compilation, correctness, performance


#### May 22, 2025

- with the recent grpo stuff was inspired to try to apply grpo/rlvr because kernel writing is indeed verifiable, and implementation preference is not opinionated, simply faster is better
- plan was to use kernelbench as prompts, generate output groups, primary reward for compilation (implies syntax correct), secondary reward for correctness compared to torch impl, extra rewards for using cuda optim techniques
- after running it for a bit noticed a lack of any occurance of primary or secondary reward, stepped back and manually tested the model (qwen2.5-1.5B-Instruct) on the prompts and realized it didnt even know what Triton was, kept trying to write weird half cuda half fake triton stuff
- initial grpo stuff: https://github.com/cdreetz/rlptx/tree/master/grpo


#### May 19, 2025

- initial idea was to just create an agentic environment and let agent (sonnet specifically) iterate on individual kernel implementations and just infitintely attempt to optimize the raw PTX
- thought process being it was a more direct route to kernels, can be optimized beyond just cuda kernels, and smaller possible instruction set than cuda. ptx -> bench -> reflect -> update -> bench -> reflect -> update
- impl: https://github.com/cdreetz/rlptx/blob/master/src/agent/agent_environment.py
