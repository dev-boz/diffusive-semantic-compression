[Cascaded image diffusion](https://arxiv.org/abs/2106.15282)
Cascaded diffusion in images (Imagen) works on the output, the image getting upscaled. DiSCo works on the input, the source getting compressed.

[Cold Diffusion](https://arxiv.org/abs/2208.09392)
Rebuilds the original in full at the end. DiSCo never rebuilds the source at all-it just stays on disk, never pulled into a model call at full length. 

[Compression-aware diffusion in image coding](https://arxiv.org/abs/2505.08281)
Same pattern as Cold Diffusion, fixed-size, restores on the output. For DiSCo nothing ever gets restored to full size on either end. 

[RAPTOR](https://arxiv.org/abs/2401.18059)
Structurally, the same multi-resolution representation of the source with hierarchical compression. But RAPTOR is a retrieval system, DiSCo is a processing system.

[LCM](https://arxiv.org/abs/2605.04050)
An extension to RLM with the same spirit but slight difference. Same hierarchical summarization idea. 
LCM's Compaction is engine-triggered, recovery is model-contingent. Context Diffusion is deterministic, the pass schedule governs recovery, no engine has to fire for any slice to be read at full resolution. LCM has no model training and the engine works with off-the-shelf models. Pass-Conditioned Training is a proposal for exactly that.

[Recursive Language Models](https://arxiv.org/abs/2512.24601)
The model decides what slices to look at, what to compress, what to recurse on. DiSCo has an explicit deterministic compression system. RLM is closer to "agent doing tool use over a long document"; DiSCo is closer to "diffusion process with compression as its noise medium.”

[Progressive Thought Refinement](https://arxiv.org/abs/2410.04707) 
PTR refines the model's thoughts; Pass-Conditioned Training refines the model's external context & how it treats each pass. PTR conditions implicitly; Pass-Conditioned Training has explicit position input.

[SCoRe](https://arxiv.org/abs/2409.12917)
SCoRe focuses on training the model for self correction, PCT is training for a specific architecture.
SCoRe position training is implicit via context structure. PCT has an explicit input signal.

[Diffusion timestep conditioning] (the general training paradigm where diffusion models learn to behave differently at different noise levels)
This is the mechanistic analog. DiSCo's pass position functions as the diffusion timestep, and the model is trained to be position-aware in the same way. The novelty in PCT is applying this to multi-pass reading rather than token-level denoising.

[Diffusion language models](LLaDA, Mercury, HDLM, MDLM line)
These provide the diffusion-process framing and timestep-conditioning training. DiSCo is an alternative noise-function.

| Noise function | At high noise | At low noise | Sequence length |
|---|---|---|---|
| Gaussian noise (standard image diffusion) | random static | clean image | constant (pixels) |
| Token masking (MDM, LLaDA) | most tokens replaced with `[MASK]` | most tokens visible | constant |
| Vocabulary abstraction (HDLM) | tokens replaced with coarse ancestors | original tokens restored | constant |
| **Length-reducing semantic compression (DiSCo)** | source compressed to short summary | source read verbatim | varies dramatically |

Context diffusion overlaps with so many other successful systems, blending them together, whilst keeping its own uniqueness. It gives me confidence in its viability.
