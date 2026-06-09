# How I arrived at semantic compression diffusion, and then found RLMs

A short note on convergent design. The longer technical writeup of the specific contributions is in a separate document.

## The problem I started on

I wanted to build something that would let LLMs handle conversations or contexts that exceed the model's context window without compaction. Compaction is the standard solution in production chat. When the conversation gets too long, summarize the older turns into shorter form, drop the originals.  Details the summarizer didn't think were important at the time are gone, even if a later turn reveals them to be relevant. The user has no easy way to know what was dropped.

This bothered me for a while before I tried to build anything. The information is *there*, on someone's disk somewhere, in the chat logs. Why does the model have to pretend the older turns don't exist as soon as the window fills? Why is the compaction decision a one-shot lossy operation? Why can't the system revisit the source if it later realizes something important was compressed away?

I started sketching architectures and chatting back and forth with agents.

## The path I took

My first attempt was a "parallel writers" architecture. A fixed-size shared memory of 16 slots, with 8 parallel agent processes writing diff-based contributions to the slots across multiple rounds per turn. The idea was that parallel writers would converge on a useful consolidation through repeated refinement, and the diff-based interface would prevent agents from clobbering each other's work. I built this on Qwen 2.5 models running locally via Ollama. It worked, sort of. I ran a 16-cell experiment (four model sizes from 0.5B to 7B, two test conditions, two memory conditions). The headline result looked clean: monotonic capacity scaling from 1/3 facts at 0.5B to 3/3 at 7B, with all eight memory-off controls failing.

Then I had the work audited by a skeptical reviewer. The audit found two problems that invalidated the headline. First, the test conversations were only ~6,600 tokens against a 32K context window which is well within the model's native capacity. So the "memory off, 4-turn window" control wasn't actually testing what I thought it was. The real baseline (full conversation in context, no memory) probably would have succeeded. Second, the compression layer used a regex hard-coded to vocabulary from earlier test runs, which meant the "monotonic capacity curve" was partly an artifact of which test needles happened to match the regex. The mechanism novelty was real, but the result was weaker than the methodology made it look.

That audit was the most valuable thing in the whole project. It clarified what I actually had (a sibling experiment in an established category) versus what I'd thought I had (a genuinely novel result). It also clarified what I'd been *trying* to build, which was something different.

## The architecture I actually wanted

What I originally meant, before the parallel-writers tangent, was a single growing output document refined across multiple passes, where each pass sees a different section of the source at a different compression level. Early passes see the whole conversation heavily compressed; later passes see narrow sections with no compression at all. Each pass edits the running output, adding or correcting detail. The skateboard-at-27 example: pass 1 sees the conversation compressed to "user mentioned learning skateboarding," pass 30 reads the specific paragraph verbatim and emits a `REPLACE` to surface "27." The compression hides detail; later passes uncover it.

Once I had this framing right, I noticed the diffusion analogy. The forward process noises by compressing. The reverse process denoises by refining. At maximum noise, the source has been replaced by a short summary; at zero noise, by verbatim chunks. The reverse process traverses the noise schedule, integrating progressively-revealed detail into the growing output state.

This led to a research proposal with three pieces:

1. **Context Diffusion** — the overall architecture (multi-pass refinement of an integration state with bounded model context and unbounded source-on-disk).
2. **DiSCo (Diffusion-based Semantic Compression)** — the choice of length-reducing semantic compression as the noise function, which structurally distinguishes the approach from masked diffusion and hierarchical-vocabulary diffusion because the noised sequence is shorter than the source rather than the same length.
3. **Pass-conditioned training** — explicit position conditioning during training, so a model native to the architecture learns to behave differently at different schedule positions.

I checked the diffusion language model literature carefully. Masked diffusion (LLaDA, Mercury, MDLM family) preserves sequence length under noise. Hierarchical diffusion (HDLM, NeurIPS 2025) replaces tokens with abstract ancestors but also preserves length. UltraLLaDA (October 2025) and LongLLaDA tried to extend diffusion LM context windows by scaling attention - same approach long-context autoregressive models use. None of them treated compression-of-length as a noise function. I thought I had a real gap.

## Then I found RLMs

[Recursive Language Models](https://arxiv.org/abs/2512.24601) by Alex Zhang, Tim Kraska, and Omar Khattab (MIT CSAIL), submitted to arxiv on December 31 2025 and revised January 28 2026, demonstrated empirically what I was trying to design toward. Their abstract: "we propose Recursive Language Models (RLMs), a general inference paradigm that treats long prompts as part of an external environment and allows the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt. RLMs can successfully process inputs up to two orders of magnitude beyond model context windows."

I read it carefully. The overlap is real:

- Both architectures treat long source as external state, not as context to load.
- Both use multiple inference calls instead of one big context window.
- Both preserve the original source verbatim and let later passes re-read it.
- Both propose specialized training (they post-trained RLM-Qwen3-8B; I proposed pass-conditioned training).
- Both demonstrate the result with Qwen-family models.

There are also real differences:

- RLM's decomposition is LLM-driven at inference time; Context Diffusion uses a predetermined schedule.
- RLM uses a programmatic REPL with recursion; Context Diffusion uses progressive semantic compression with linear pass schedule.
- RLM's training is implicit position-conditioning via trajectory imitation; Context Diffusion proposes explicit position-conditioning as an input signal.

I had been working on this independently. I found their paper after my proposal was largely written. The honest framing is convergent design: two groups arrived at similar overall approaches to the same problem from different angles. RLM landed first publicly, with empirical results, with frontier researchers and the institutional weight that comes with that. My contribution is narrower than what I'd planned to claim. It exists as a specific mechanism (diffusion-based semantic compression) and a specific training objective (pass-conditioning) that, as far as my searches show, are not in their work or in the surrounding literature.

## What I'm doing about it

This post and a companion technical post are for timestamping the contributions I think are novel: semantic compression diffusion as a noise function for arbitrary-length context, and pass-conditioned training for multi-pass refinement architectures — so they exist in the public record alongside RLMs rather than after them. The technical post is more detailed and is intended as the actual proposal document for anyone who wants to build on the ideas or audit them for prior art I missed.

I'm running small-scale viability experiments locally on consumer hardware. They are not validated. I expect that whatever interesting results emerge will come from the specific mechanism choices that distinguish this from RLMs.

I'd value collaboration from anyone working on related problems, particularly anyone who knows the diffusion-LM literature deeply enough to confirm or refute the novelty claim on the noise function side. I'll be honest about whether ideas survive contact with empirical reality.

## What I learned that was worth more than the original goal

Two things, both about how to do this kind of work.

First, the audit was the single most valuable thing that happened. I'd been ready to publish the parallel-writers result. The audit caught two structural problems that would have destroyed the work after publication. Running adversarial review *before* publication is the cheapest possible way to find out you're wrong. I'm going to keep doing it.

Second, the pattern of "I keep arriving at things that exist in the literature" turns out to be a feature, not a bug. It means the design intuitions are calibrated. RLMs being a close cousin of what I was building is *evidence the direction is right*, not evidence I'm behind. Frontier researchers and amateur tinkerers converging on similar architectures is what fields look like when an idea is becoming ripe. Showing up early to a ripe field with a slightly different angle is still useful work, even if you don't get to plant the headline flag.

The longer technical writeup lives at the README of this repository.
