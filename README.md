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

- **Context Diffusion** (architecture) - A multi-pass refinement system where a single evolving integration state is iteratively built up across passes by edit operations against scheduled views of an external source — each view at a schedule-determined compression level, the source itself preserved verbatim on disk.

- **DiSCo: Diffusion-based Semantic Compression** (noise function) - The specific choice of length-reducing semantic compression as the forward noise function in a diffusion-style refinement process, which structurally distinguishes the approach from masked diffusion and vocabulary-abstraction diffusion.
- **Pass-Conditioned Training** (training objective) - An explicit training paradigm in which a model is conditioned on its position within a multi-pass refinement schedule, learning to produce qualitatively different outputs at different schedule positions.

Context Diffusion as an architecture overlaps substantially with RLMs and I do not claim independent priority on the architectural shape. DiSCo and Pass-Conditioned Training, as far as my searches show, do not appear in the literature in the form proposed.

A note on the name: diffusion processes are conventionally named for the variable that undergoes the forward process — image diffusion noises images, latent diffusion noises latents. Here the noised variable is the model's view of the context, hence *Context Diffusion*. The integration state refines but is never noised, which is why this is not called state diffusion or output diffusion.

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

A practical extension worth noting although not a primary contribution. After each pass, the model emits a confidence score about the adequacy of the current integration state for the user's task. If confidence is sufficiently high for enough passes (tunable), subsequent scheduled passes are skipped. This makes per-query cost adaptive. Simple queries terminate in fewer passes, complex queries run the full schedule.

This is not novel as a primitive (LLMs emitting confidence signals is well-established). What's notable is its interaction with the deterministic schedule: in LLM-driven loops (RLMs, ReAct, reasoning chains) the model is already deciding what to do next, and confidence just refines that. In Context Diffusion the schedule would otherwise execute fully, so confidence provides an orthogonal cost control without requiring the model to plan its own loop. Bolt-on, not headline, but worth describing because it's how the schedule-driven framing handles variable query difficulty.

## 4. The noise function: DiSCo

This is the first of the two contributions I believe are novel.

Standard masked diffusion language models (LLaDA, Mercury, the broader MDM family) use token masking as the noise function: tokens in the sequence are progressively replaced with `[MASK]` according to a noise schedule. The forward process noises by masking more tokens; the reverse process predicts masked tokens. The sequence stays at the same length throughout.

Hierarchical Diffusion Language Models ([HDLM](https://arxiv.org/abs/2505.19529), Zhou et al., NeurIPS 2025) extend this with vocabulary abstraction as the noise function: tokens are replaced with higher-level "ancestor" tokens in a hierarchical vocabulary. The forward process moves up the hierarchy; the reverse process predicts more detailed tokens. The sequence stays at the same length throughout.

Both of these noise functions are bounded by the model's context window. A diffusion LM trying to ingest a 100K-token source needs a 100K-token context window for any noised intermediate state, because the noised sequence is the same length as the source.

**DiSCo proposes length-reducing semantic compression as the noise function.**

A sketch of the process, kept informal but with the objects separated properly:

```
x          source (immutable, on disk; arbitrary length)
q          the task or query
K          total passes in the schedule
k          pass index, 1..K
sigma_k    compression level at pass k; sigma_1 = max, decreasing to sigma_K = 0
r_k        source region read at pass k; broad early, narrow late (coupled to sigma_k)
v_k        the view at pass k:  v_k = C(slice(x, r_k), sigma_k)
s_k        integration state after pass k; s_0 = empty
e_k        edit operations emitted at pass k

e_k ~ p_theta( e | s_{k-1}, v_k, k, K, q )
s_k = apply(s_{k-1}, e_k)
y  ~ p( y | s_K, q )
```

The forward (noising) process is the compression family `C(x, sigma)`: as `sigma` increases, the representation of `x` shortens, terminating in the maximum-noise view `C(x, sigma_max)` — the `x_T` of this process — a global summary that may be orders of magnitude shorter than `x`. The pass schedule traverses the forward process in reverse: each successive view sits at a lower noise level, until late passes read narrow regions of the source verbatim.

There are two objects here with two distinct roles, and conflating them produces most of the confusion this kind of proposal invites. The **view sequence** is the noised variable: it is what the forward process degrades and what the schedule progressively reveals. It is produced by the schedule, not by the model. The **integration state** is the model's running estimate of the clean signal — the analog of the per-step clean-data estimate (the `x̂_0` prediction) that a standard diffusion model maintains at every denoising step. The state is refined; it is never noised.

Two honest deviations from textbook diffusion:

1. **The estimation target is not `x`.** The state converges not toward the source itself but toward `s*`, the task-conditioned distillation of `x` sufficient to handle `q`. This is what keeps the state bounded even when the source is not (see "Bounding the integration state" below).
2. **The reverse process is grounded, not generative.** Each step is conditioned on progressively revealed ground truth — the views — so the process reconstructs by integration rather than imagining detail from a learned prior. This is what makes it a reading architecture rather than a generator.

The maximum-noise view is informative about `x`, unlike Gaussian diffusion where `x_T` is independent of the data. Cold Diffusion (treated below) establishes that this is acceptable within the diffusion family: deterministic degradations with informative endpoints support the same training recipe.

One unification worth making explicit: the conditional `p_theta(e | s_{k-1}, v_k, k, K, q)` is exactly the Pass-Conditioned Training tuple of §5 — `(pass_position, schedule_total, input_view, integration_state, target_contribution)`. The training objective is maximum likelihood on this conditional. The architecture and the training proposal are the same object viewed from inference time and from training time.

The contrast across noise functions:

| Noise function | At high noise | At low noise | Sequence length |
|---|---|---|---|
| Token masking (MDM, LLaDA) | most tokens replaced with `[MASK]` | most tokens visible | constant |
| Vocabulary abstraction (HDLM) | tokens replaced with coarse ancestors | original tokens restored | constant |
| **Length-reducing semantic compression (DiSCo)** | source compressed to short summary | source read verbatim | varies dramatically |

The structural consequence: diffusion processes using DiSCo as their noise function can ingest sources of arbitrary length, bounded only by what the compressor can summarize coherently. Token-masking and vocabulary-abstraction approaches cannot, because they require the noised sequence to fit in the model's context window.

This is the central architectural argument. *The choice of noise function determines whether the diffusion process is bounded by the model's context window or by the compressor's capacity.* DiSCo is bounded by the latter, which is a much weaker constraint.

One caveat keeps this claim honest. For sources beyond what the compressor can ingest in a single call, `C` must itself be applied hierarchically — chunked, map-reduce-style summarization — so the maximum-noise view is the root of a summary tree. This quietly reintroduces a RAPTOR-shaped structure. The difference is the role it plays: in RAPTOR-style memory systems the tree *is* the load-bearing store, and what the tree loses is lost; here the tree only manufactures views, ground truth remains the flat source on disk, and late passes read it directly, so compression loss at any level of the tree is recoverable rather than permanent. "Bounded by the compressor" therefore means bounded by what hierarchical compression can summarize coherently — a much weaker constraint than the model window, but not a free one.

### Why this isn't HDLM

The most likely "isn't this just HDLM?" objection deserves a precise answer. HDLM does semantic noise via vocabulary abstraction at fixed length. Tokens become abstract ancestors of themselves; sequence length is invariant. DiSCo compresses content across many tokens into fewer tokens at coarser granularity. The mechanisms operate on different axes (per-position semantic level vs. cross-position length reduction) and solve different problems (improved generation quality at bounded context vs. ingestion of unbounded-length context).

### Why this isn't cascaded image diffusion

Cascaded diffusion in images (Imagen) generates at low spatial resolution first and upscales. This is structurally analogous in the image domain: noise via spatial downscaling, refinement by upscaling. However it operates on the *output* side  - the image being generated changes resolution as denoising progresses - rather than on the input side. DiSCo addresses the input scaling problem, which is the harder one for text because text has no native multi-resolution representation analogous to image pyramids.

### Why this isn't Cold Diffusion

[Cold Diffusion](https://arxiv.org/abs/2208.09392) (Bansal et al., 2022; NeurIPS 2023) showed that the diffusion training recipe survives replacing Gaussian noise with arbitrary deterministic degradations - blur, masking, downsampling - including degradations whose endpoints remain informative about the data. (Soft Diffusion, Daras et al. 2022, generalizes score matching to the same family.) It cuts both ways for this proposal, so it gets a direct treatment.

What it gives DiSCo: legitimacy for the operator class. "Noise function" does not have to mean stochastic corruption toward an uninformative terminal state; the diffusion family already admits deterministic degradations with structured endpoints. `C(x, sigma_max)` being a meaningful summary rather than noise is not a category violation.

What it threatens: downsampling is a resolution-reducing degradation, which makes cold diffusion the nearest prior art for "the degraded representation is smaller than the data." Three distinctions. First, cold diffusion's restoration network *inverts* the degradation: its job is to reconstruct `x_0` in full, and full-resolution `x_0` is materialized at the end. DiSCo never reconstructs `x`; the source stays on disk, and the reverse traversal integrates into a bounded, task-conditioned state. Second, as with cascaded generation above, the degradation operates on the generation target - the output side. DiSCo degrades the *conditioning input*; the output is the integration state, which has no full-resolution form to be restored to. Third, image downsampling moves within a fixed-dimensionality array family, and the model still processes full-size tensors at restoration time, so it does not deliver the property that is DiSCo's entire point: no stage of DiSCo ever materializes `x` at full length inside a model call.

The same input-side/output-side distinction covers compression-aware diffusion in image coding ([for example](https://arxiv.org/abs/2505.08281)), where denoising is initialized from a compressed latent rather than from pure noise — further evidence that the informative-endpoint family is established, again with constant latent dimensionality and an output-side restoration target.

### Why this isn't RAPTOR

[RAPTOR](https://arxiv.org/abs/2401.18059) (Sarthi et al., ICLR 2024) recursively embeds, clusters, and summarizes chunks bottom-up into a tree whose levels span abstraction — structurally, the same multi-resolution representation of the source that hierarchical compression produces here (see the caveat above). The convergence on the artifact is real and worth conceding plainly. The divergence is everything that happens *with* it.

RAPTOR is a retrieval system: at inference time, nodes are selected by embedding similarity to the query — possibly across levels — packed into a single context, and read once. There is no schedule, no integration state, no passes, and no revision: whether a verbatim leaf is ever seen depends on a similarity score, and there are no intermediate conclusions to revisit when detail elsewhere contradicts them. That is exactly the late-realization failure described in §1. Context Diffusion buys the opposite guarantee at the opposite price: a deterministic schedule visits the source coarse-to-fine regardless of what any query embedding scores, integrating into a state that later passes can edit when verbatim reads surface what the compression hid. The tree, where present, is a view factory; in RAPTOR it is the retrieval index itself.

The two claimed-novel contributions don't touch RAPTOR at all: it has no noise-function framing — the tree is an index, not a forward process with a trained reverse traversal — and it proposes no training objective; the retriever and reader are off the shelf.

### Why this isn't just RLM

RLM treats the long source as a REPL variable that the LLM examines via code. The LLM decides what slices to look at, what to compress, what to recurse on. Compression in RLM is implicit and LLM-controlled. DiSCo has an explicit deterministic compression schedule where slice size and compression level are coupled by design. RLM is closer to "agent doing tool use over a long document"; DiSCo is closer to "diffusion process with compression as its noise medium." Both can handle arbitrary-length inputs; the mechanism for *how* is different.

### Bounding the integration state

Two things govern the state's size; two things govern termination. They are different questions and both deserve explicit answers.

**Size.** The per-pass context invariant is `|s_{k-1}| + |v_k| + instruction overhead <= W` for every pass, where `W` is the model's window. The schedule controls `|v_k|` by construction. `|s_k|` is controlled by the model's edit mix: as the state approaches budget, expected behavior shifts from net-additive (`ADD`-heavy) toward net-neutral or net-reductive (`REPLACE`/`REMOVE`-heavy). This is itself position- and budget-dependent behavior — squarely what Pass-Conditioned Training targets — and a state-budget signal is a natural third conditioning input alongside pass position (an extension, not a core claim).

The deeper reason the state stays bounded is the estimation target. The state estimates `s*`, the task-conditioned distillation of the source, not the source itself. The architecture therefore carries an assumption worth stating plainly: *the task-relevant content of an arbitrarily long source is compressible into a bounded state.* Tasks whose faithful output is proportional to source length — verbatim transcription of everything, say — violate the assumption and are out of scope, as they are for any bounded-window system.

**Termination.** The schedule is finite and known in advance. In the default schedule, every region of `x` has been visited at `sigma = 0` by the end — cumulatively across the late passes; no single pass reads the full source verbatim, which would itself exceed the window for large sources. The model receives `k` and `K` explicitly, so "this is the final pass" is an input, not an inference, and a pass-conditioned model can finalize accordingly. Confidence-based early exit (§3) can terminate the schedule sooner when several consecutive passes agree the state is adequate.

**Cost honesty.** A schedule that guarantees every region is eventually read verbatim has a pass count linear in source length (roughly `|x| /` verbatim-slice-size). That is the fair price: no system can guarantee that no detail is unrecoverable without, in the worst case, being able to look at everything once. Schedules can also be sublinear — skipping verbatim coverage of low-relevance regions — trading the guarantee for cost; because the source remains on disk, what a sublinear schedule skips is *unread*, not *lost*, and is recoverable by extending the schedule. The win over a single long-context read is bounded per-call context and an auditable, recoverable process, not sublinear reading cost. The ~30-pass schedules in §6 are calibrated to the 80–150K-token experimental sources, not a universal constant.

## 5. The training objective: Pass-Conditioned Training

This is the second of the two contributions I claim as novel.

A model intended to operate inside the Context Diffusion architecture needs to behave differently at different schedule positions. At early passes, it should produce structural scaffolds. At middle passes, it should refine within established structure. At late passes, it should surface specific verbatim detail by reading source at full resolution.

Small generalist instruction-tuned models do this work poorly. (The parallel-writers experiment described in the [summary post](https://github.com/dev-boz/diffusive-semantic-compression/blob/master/BACKGROUND.md) confirmed this empirically: 1.5B Qwen produced ~50% malformed edit operations under prompt-based slot routing.) Larger frontier models follow the edit-operation format reliably — RLM demonstrates strong generalists driving multi-pass loops — but format compliance is not position-appropriate behavior. An untrained model has no signal for *when* to scaffold versus refine versus surface verbatim detail beyond whatever the prompt asserts; in practice it over-contributes at every position and under-integrates the detail that late-pass verbatim reads surface. Pass-Conditioned Training targets that gap directly, with the second goal of pushing the capability down into small, cheap models, where prompt-based control demonstrably fails.

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

[SCoRe](https://arxiv.org/abs/2409.12917) (Kumar et al., Google DeepMind, ICLR 2025) is also adjacent: it trains multi-turn self-correction with reinforcement learning such that the model's second attempt behaves differently from, and improves on, its first. That is position-dependent behavior across passes, learned implicitly through the multi-turn structure. The same two distinctions as PTR apply: SCoRe refines the model's own prior output rather than a representation of external context, and position is carried implicitly by the context structure rather than supplied as an explicit input signal.

Diffusion timestep conditioning (the general training paradigm where diffusion models learn to behave differently at different noise levels) is the mechanistic analog. DiSCo's pass position plays the role of diffusion timestep, and the model is trained to be position-aware in the same way. The novelty in Pass-Conditioned Training is applying this paradigm to multi-pass refinement of external context rather than to token-level denoising.

RLM-Qwen3-8B is post-trained on RLM trajectories — the closest existing example of training a model for multi-pass context processing. RLM does not use explicit position conditioning; the model learns position-appropriate behavior implicitly from trajectory diversity. The fact that this implicit conditioning produces 28.3% improvement over the base model suggests that explicit position conditioning (Pass-Conditioned Training) could provide additional gains, but this is unverified.

## 6. What I'm running and what I'd value help with

Small-scale viability experiments are in progress:

1. **Context Diffusion on Qwen 2.5 generalist models**, no fine-tuning, against synthetic conversations of 80K-150K tokens that exceed Qwen 2.5 0.5B's effective context. Tests three conditions (Context Diffusion, rolling compaction, no-memory) across multiple model sizes and difficulty tiers of query (lookup, single-turn detail, narrative arc, synthesis). Grading is two-axis: quantitative (per-query recall) and qualitative (output document coherence assessed via LLM-as-judge and human readers).

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
- The architecture assumes the task-relevant content of the source is compressible into a bounded integration state even when the source is not. Tasks whose faithful output scales with source length are out of scope.
- Schedules that guarantee eventual verbatim coverage of the whole source have pass counts linear in source length. Sublinear schedules trade away that guarantee - recoverably, since the source remains on disk.

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

*Last updated: 2026-06-10. This document will be revised as the viability experiments produce data and as prior art is refined. Version history is in the repository commit log.*
