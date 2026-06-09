# Semantic Compression Diffusion and Pass-Conditioned Training: A Proposal

A technical writeup of two contributions to multi-pass language model architectures for handling arbitrary-length context. Companion to the [short summary](./BACKGROUND.md) of how this work arrived at convergent design with Recursive Language Models.

This is a proposal, not a validated finding. Small scale viability experiments are in progress. Collaboration is welcome and encouraged, particularly on empirical validation and on prior-art that I may have missed.

## 1. The problem

Production chat systems and agent frameworks usually handle conversations or task histories that exceed the model's context window through *compaction*: irreversibly summarizing older content into shorter form. Compaction is lossy at the moment it's applied. Details the summarizer didn't think were important are gone, even if a later turn reveals them to be relevant. The user cannot reliably see what was lost. The model cannot recover it easily.

Existing alternatives have related limitations:

- **Long-context models** (Gemini, Claude with extended context) push the bound up but do not eliminate it. Attention degrades on middle content within large windows.
- **Memory-augmented architectures** (MemGPT, MemAgent, TiMem, HyMem, RAPTOR) bound input via hierarchical summaries or external memory with bounded routing. Late realization that an earlier chunk was important does not recover the chunk.
- **Diffusion language models** (LLaDA, Mercury, HDLM) are bound by their own context window because their noise functions preserve sequence length.
- **Durable-context agents** (InfiAgent, file-centric agents) externalize state to files but still pay context cost for every loaded fragment.
- **Recursive Language Models** ([RLMs](https://arxiv.org/abs/2512.24601), Zhang et al. 2025) demonstrate that arbitrary-length input is achievable via LLM-driven recursive decomposition. This is the closest existing work and an important reference point.

What none of these provide is a *deterministic, schedule-driven* refinement architecture where the noise function is *length-reducing semantic compression* and the model is *explicitly trained on its position within the schedule*. This proposal describes both pieces.

## 2. Three named pieces

I distinguish three separable contributions. They compose but each can be discussed and evaluated independently.

- **Context Diffusion** (architecture) - A multi-pass refinement system where a single growing integration state is iteratively built up across passes by editing operations against an external source preserved on disk.
- **DiSCo: Diffusion-based Semantic Compression** (noise function) - The specific choice of length-reducing semantic compression as the forward noise function in a diffusion-style refinement process, which structurally distinguishes the approach from masked diffusion and vocabulary-abstraction diffusion.
- **Pass-Conditioned Training** (training objective) - An explicit training paradigm in which a model is conditioned on its position within a multi-pass refinement schedule, learning to produce qualitatively different outputs at different schedule positions.

Context Diffusion as an architecture overlaps substantially with RLMs and I do not claim independent priority on the architectural shape. DiSCo and Pass-Conditioned Training, as far as my searches show, do not appear in the literature in the form proposed.

## 3. The architecture: Context Diffusion

At inference time:

1. A **pass schedule** determines, for each pass, what section of the source is read and at what compression level. Slice size and compression are coupled: narrow slice = light compression; whole source = heavy compression. Early passes see large compressed slices; late passes see small verbatim slices.

2. Each pass takes as input: the current **integration state** (the cumulative working representation built up by all prior passes), the **source slice** for this pass at this pass's compression level, the **pass position** within the schedule, and the user's task or query.

3. Each pass emits **edit operations** against the integration state: `ADD`, `REPLACE`, `REMOVE`, `NO_CHANGE`. These are applied to produce the next integration state.

4. After the schedule completes, the integration state is used by a final inference call to produce the output the task requires - a reply, a code block, an analysis, a tool call, an action plan. The architecture is agnostic with respect to output type.

### Key design properties

- **Original content preservation.** The source is immutable on disk. Late passes can re-read source content at full resolution if the integration state's current detail level is insufficient.
- **Bounded model context per pass.** No single inference call sees more than fits comfortably. The model never needs to ingest the full source at once.
- **Unbounded source length.** The total source can grow indefinitely because no single pass needs to fit it all.
- **Deterministic schedule.** Cost is predictable. The schedule is a parameter that can be tuned per use case.
- **Observable audit trail.** Every pass's input, output, and effect on the integration state is recorded to disk.

### Optional: confidence-based adaptive scheduling

A practical extension worth noting although not a primary contribution. After each pass, the model emits a confidence score about the adequacy of the current integration state for the user's task. If confidence is sufficiently high for enough passes (tunable), subsequent scheduled passes are skipped. This makes per-query cost adaptive. Simple queries terminate in less passes, complex queries run the full schedule.

This is not novel as a primitive (LLMs emitting confidence signals is well-established). What's notable is its interaction with the deterministic schedule: in LLM-driven loops (RLMs, ReAct, reasoning chains) the model is already deciding what to do next, and confidence just refines that. In Context Diffusion the schedule would otherwise execute fully, so confidence provides an orthogonal cost control without requiring the model to plan its own loop. Bolt-on, not headline, but worth describing because it's how the schedule-driven framing handles variable query difficulty.

## 4. The noise function: DiSCo

This is the first of the two contributions I believe are novel.

Standard masked diffusion language models (LLaDA, Mercury, the broader MDM family) use token masking as the noise function: tokens in the sequence are progressively replaced with `[MASK]` according to a noise schedule. The forward process noises by masking more tokens; the reverse process predicts masked tokens. The sequence stays at the same length throughout.

Hierarchical Diffusion Language Models ([HDLM](https://arxiv.org/abs/2505.19529), Zhou et al., NeurIPS 2025) extend this with vocabulary abstraction as the noise function: tokens are replaced with higher-level "ancestor" tokens in a hierarchical vocabulary. The forward process moves up the hierarchy; the reverse process predicts more detailed tokens. The sequence stays at the same length throughout.

Both of these noise functions are bounded by the model's context window. A diffusion LM trying to ingest a 100K-token source needs a 100K-token context window for any noised intermediate state, because the noised sequence is the same length as the source.

**DiSCo proposes length-reducing semantic compression as the noise function:**

```
x_T = Compress(x_0)
p_θ(x_{t-1} | x_t, t)
```

The forward process at maximum noise replaces the source `x_0` with its compression — a meaning-preserving summary that may be orders of magnitude shorter. The reverse process is a denoising distribution conditioned on the current noised state and the timestep. At maximum noise, `x_T = Compress(x_0)` is dramatically shorter than `x_0`. As noise decreases, the noised state lengthens, asymptotically approaching `x_0` itself at zero noise.

The contrast across noise functions:

| Noise function | At high noise | At low noise | Sequence length |
|---|---|---|---|
| Token masking (MDM, LLaDA) | most tokens replaced with `[MASK]` | most tokens visible | constant |
| Vocabulary abstraction (HDLM) | tokens replaced with coarse ancestors | original tokens restored | constant |
| **Length-reducing semantic compression (DiSCo)** | source compressed to short summary | source read verbatim | varies dramatically |

The structural consequence: diffusion processes using DiSCo as their noise function can ingest sources of arbitrary length, bounded only by what the compressor can summarize coherently. Token-masking and vocabulary-abstraction approaches cannot, because they require the noised sequence to fit in the model's context window.

This is the central architectural argument. *The choice of noise function determines whether the diffusion process is bounded by the model's context window or by the compressor's capacity.* DiSCo is bounded by the latter, which is a much weaker constraint.

### Why this isn't HDLM

The most likely "isn't this just HDLM?" objection deserves a precise answer. HDLM does semantic noise via vocabulary abstraction at fixed length. Tokens become abstract ancestors of themselves; sequence length is invariant. DiSCo compresses content across many tokens into fewer tokens at coarser granularity. The mechanisms operate on different axes (per-position semantic level vs. cross-position length reduction) and solve different problems (improved generation quality at bounded context vs. ingestion of unbounded-length context).

### Why this isn't cascaded image diffusion

Cascaded diffusion in images (Imagen) generates at low spatial resolution first and upscales. This is structurally analogous in the image domain: noise via spatial downscaling, refinement by upscaling. However it operates on the *output* side  - the image being generated changes resolution as denoising progresses - rather than on the input side. DiSCo addresses the input scaling problem, which is the harder one for text because text has no native multi-resolution representation analogous to image pyramids.

### Why this isn't just RLM

RLM treats the long source as a REPL variable that the LLM examines via code. The LLM decides what slices to look at, what to compress, what to recurse on. Compression in RLM is implicit and LLM-controlled. DiSCo has an explicit deterministic compression schedule where slice size and compression level are coupled by design. RLM is closer to "agent doing tool use over a long document"; DiSCo is closer to "diffusion process with compression as its noise medium." Both can handle arbitrary-length inputs; the mechanism for *how* is different.

## 5. The training objective: Pass-Conditioned Training

This is the second of the two contributions I claim as novel.

A model intended to operate inside the Context Diffusion architecture needs to behave differently at different schedule positions. At early passes, it should produce structural scaffolds. At middle passes, it should refine within established structure. At late passes, it should surface specific verbatim detail by reading source at full resolution.

Generalist instruction-tuned language models do this work poorly because they were not trained for it. They tend to over-contribute at every pass position, fail to integrate detail that late-pass verbatim reads surface, and emit malformed edit operations at high rates. (The parallel-writers experiment described in the [summary post](./BACKGROUND.md) confirmed this empirically: 1.5B Qwen produced ~50% malformed edit operations under prompt-based slot routing.)

**Pass-Conditioned Training** proposes training the model on data conditioned on explicit pass position. Each training example is a tuple:

```
(pass_position, schedule_total, input_view, integration_state, target_contribution)
```

Where:

- `pass_position` is the integer index of this pass within the schedule (1 to `schedule_total`).
- `schedule_total` is the total number of passes in the schedule.
- `input_view` is the model's compressed view of the source at this pass.
- `integration_state` is the cumulative output state going into this pass.
- `target_contribution` is the set of edit operations the model should emit.

The model is trained to predict `target_contribution` given the other four fields. Pass position and schedule total are explicit inputs - encoded as special tokens, embeddings, or auxiliary input fields. The model learns that different behaviors are expected at different positions, and the training loss rewards matching position-appropriate behavior.

### Data generation

The training data is generated synthetically. A strong teacher model (a larger LLM such as GPT-5 or Claude Opus) runs Context Diffusion against a corpus of long sources, recording each pass's contribution. Each pass becomes one training example for a smaller student model. Tens of thousands of long sources × ~30 passes per source yields hundreds of thousands of training pairs from existing text and a teacher.

### Relationship to closest prior work

[Progressive Thought Refinement](https://arxiv.org/abs/2410.04707) (PTR, Yang et al., ICLR 2025) is the closest existing approach. PTR trains language models on progressive refinement of their own reasoning chains via "weighted thought-mask fine-tuning." The training paradigm is in the same family. Two distinctions:

- **Application domain**: PTR refines self-generated thoughts; Pass-Conditioned Training refines the model's representation of external context.
- **Position conditioning**: PTR conditions implicitly through the refinement examples themselves; Pass-Conditioned Training proposes explicit position input as a separate signal.

Diffusion timestep conditioning (the general training paradigm where diffusion models learn to behave differently at different noise levels) is the mechanistic analog. DiSCo's pass position plays the role of diffusion timestep, and the model is trained to be position-aware in the same way. The novelty in Pass-Conditioned Training is applying this paradigm to multi-pass refinement of external context rather than to token-level denoising.

RLM-Qwen3-8B is post-trained on RLM trajectories — the closest existing example of training a model for multi-pass context processing. RLM does not use explicit position conditioning; the model learns position-appropriate behavior implicitly from trajectory diversity. The fact that this implicit conditioning produces 28.3% improvement over the base model suggests that explicit position conditioning (Pass-Conditioned Training) could provide additional gains, but this is unverified.

## 6. What I'm running and what I'd value help with

Small-scale viability experiments are in progress:

1. **Context Diffusion on Qwen 2.5 generalist models**, no fine-tuning, against synthetic conversations of 80K-150K tokens that exceed Qwen 2.5 0.5B's effective context. Tests three baselines (Context Diffusion, rolling compaction, no-memory) across multiple model sizes and difficulty tiers of query (lookup, single-turn detail, narrative arc, synthesis). Grading is two-axis: quantitative (per-query recall) and qualitative (output document coherence assessed via LLM-as-judge and human readers).

2. **Direct comparison against RLM-Qwen3-8B** where feasible. Their model is open at [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm). Running their approach on the same hardware as my experiments would enable comparative measurements.

3. **The Pass-Conditioned Training proposal is not currently being implemented** — it requires substantial compute for the synthetic data generation and fine-tuning. This is where I'd value collaboration from anyone with access to training infrastructure. If the Context Diffusion viability experiment produces positive results on generalist models, the natural follow-up is to train a small model with pass-conditioning and measure the improvement.

Beyond compute, I would value:

- Skeptical review of the prior-art claim. If DiSCo or Pass-Conditioned Training appears in any literature I missed, I want to know. The deep research I ran with both ChatGPT and Gemini surfaced no exact matches but I cannot guarantee comprehensiveness.
- Feedback on the mathematical formulation of DiSCo. The formulation I've given is informal; a proper treatment would specify the compression operator's properties (whether it's deterministic or stochastic, whether it admits a well-defined reverse-time process, how the noise schedule relates to compression ratio).
- Empirical comparisons against the obvious baselines (rolling compaction, RLM, HDLM at full sequence length) on standard long-context benchmarks (LOCOMO, LongMemEval, BABILong).
- Honest critique of the entire framing.

## 7. Honest scope

This proposal is unvalidated. The architecture is being tested at small scale; the training objective is purely a proposal. Specifically:

- I have not trained any model with the Pass-Conditioned Training objective.
- The current Context Diffusion implementation runs on generalist instruction-tuned models that were not designed for the edit-operation primitive.
- I have not directly compared against RLMs, HDLM, or the rolling-compaction baseline on standard benchmarks.
- The mathematical formulation of DiSCo is informal and the compression operator's formal properties are not characterized.
- Convergent design with RLMs means the architectural shape is no longer novel; only the noise-function and training-objective contributions are claimed as such.

These are real limitations. The point of stating them is to make clear what is and is not being claimed.

## 8. Acknowledgments and lineage

This proposal draws on three recent research lines:

- **Recursive Language Models** ([Zhang, Kraska, Khattab, 2025](https://arxiv.org/abs/2512.24601)) demonstrate empirically that arbitrary-length input is achievable through multi-pass external-state architectures. They are the closest existing work and the appropriate baseline for any future empirical comparison.
- **Diffusion language models** (LLaDA, Mercury, HDLM, MDLM line) provide the diffusion-process framing and timestep-conditioning training paradigm. DiSCo's contribution lives within this family as a noise-function variant.
- **Progressive Thought Refinement** ([Yang et al., ICLR 2025](https://arxiv.org/abs/2410.04707)) demonstrates that language models can be trained on progressive refinement directly. Pass-Conditioned Training is in the same family with explicit position conditioning added.

Methodological discipline (pre-registered evaluation, adversarial referee audit, honest disclosure of failure modes) is borrowed from small-scale rigorous research projects such as [Prizma](https://github.com/nazmiefearmutcu/Prizma) (Karmutcu, 2026), which demonstrate that disciplined methodology produces defensible findings without requiring frontier-lab resources.

## 9. License and contact

CC-BY-4.0 for the proposal documents in this repository. Code in `experiments/` is offered under MIT once present. If implemented and validated by anyone, credit should be shared between this proposal and the implementation; if implemented and refuted, the proposal stands as a hypothesis tested and falsified.

Contact: GitHub issues on this repository.

---

*Last updated: 2026-06-09. This document will be revised as the viability experiments produce data and as prior art is refined. Version history is in the repository commit log.*
