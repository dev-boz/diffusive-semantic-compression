# Context Diffusion

Semantically compress massive context and process through a diffusion-like architecture and a custom trained LLM

## The problem

When using an AI to process complex ideas, write long storylines, develop characters, refactor codebases or develop business strategies, the session often grows very large, especially if it's a continued session over several days or weeks that you keep coming back to. These conversations are often highly productive, or entertaining, or valuable, because they are so detailed and are rich with content. But then once they get too large for the context window (or even sooner for hallucinations) then you need to compact or summarise. This is inherently lossy and for the most part out of the user's control. There are systems to make this process better (memory, rag etc) but in my experience I have yet to come across a solution that can retain the one thing that I think is so valuable in really large sessions.

I lack a single descriptive word, so I will use a few: sentiment, vibes, affect, intent, subtext, implication. Subtle nuance that the ai might pick up on if given the whole view, but is hidden if only given partial or fragmented information. I'll refer to it mostly as nuance.

My personal annoyance with this comes from large conversations spanning days or weeks brainstorming ideas. But at a certain point in the conversation I don't know if the LLM is still incorporating my previous detailed information or if it's been compacted away.

## My search for the solution

I have a background in 3d graphics, so my visual model was basically how very complex scenes are handled by renderers. One of the best solutions is Interactive Photorealistic Rendering (IPR) where the scene is first displayed as blurry/coarse then progressively gets more detailed. I started investigating if this could be applied to a text LLM for large context the way IPR is applied to large 3D scenes, since my understanding was that the limiting factors for LLM are both needing to hold output in cache before finalising and also needing to prefill everything at the same time. What I found was that there are many systems that do this, but not quite in the exact way I had pictured.

The other area that ties in is the explosion of context compression schemes that are being developed to deal with token costs. I see new projects coming out all the time such as RTK, headroom, caveman etc. many minds working on the problem of compression without losing meaning. My goal became much clearer - apply that same thinking to compress a massive conversation. But none of these are universally applicable, some handle code and terminal outputs, others like caveman handle semantic compression. But they are all still inherently lossy on the large-conversational scale.

The solution is an intersection of these ideas. Use a mixture of compression schemes combined with a coarse-to-fine input system and progressive output do disk. Work within the limitations of the models context window input/output by keeping most of the context and output outside of cache. I realised that the compression can map to the diffusion idea by way of compressed slices/passes. First pass is the whole session highly compressed, Second 2 passes are the conversation split in 2 with slightly less compression. The final passes are verbatim text. Each pass has partial output to disk, influenced by the amount of compression. This was my solution to preserving overall nuance & fine detail at the same time. There was always an inherent risk that I would be introducing a new lossy factor, which is the loss incurred by the steps from coarse to fine and the combined output. I believe the best solution to this problem would be a model that is trained to treat each slice/pass differently.

I’m a visual thinker. Everything here is visualised in abstract first. As such I couldn’t help but create a visualisation of -how the system works-

## The core concepts:

Diffusion-based Semantic Compression (DiSCo) - the compression-as-noise idea. 

Pass-conditioned reading - making sure the model efficiently reads the passes. First passes should be scaffolding to prepare for final details from the later passes.

Context diffusion - to sum up the overall combined system. We diffuse/noise the context with semantic compression. Then we de-noise with progressive refinement.

Caveat: there is a very obvious tradeoff, as context size grows, token use and time per output increases linearly. This would not be ideal for everyone. But for a few niche use cases the tradeoff is worth it when you really need to capture the whole conversation with both nuance and fine details. That said, I have a few ideas for features that can possibly alleviate this burden slightly. Read more here.

This project has iterated through some failed attempts before landing here. I also got part way through when I found RLM (Zhang) in a sweep for prior art. Of all the other similar systems RLM is the closest in both functionality and intent. I actually had to refine my original idea since it was too similar. I think it's always a good sign that you're on the right path when your design converges with existing (and very good) ideas. What remains, which I believe is still novel, is using compressed slices as noise. I’ve done a pretty thorough look at other similar prior art which I’ll list the similarities and differences -here-

## Current status

I’m currently doing viability testing using small untrained models (mostly Qwen family). The current status is “minimum viability" which means many parts working in isolation and a handful of successful end-to-end runs. It is not yet fully conclusive whether the issues are due to lack of model training, improper testing method or architecture failure. This is still very early testing but there are very early signs that a custom trained model will perform significantly better. Testing repo -here-

## Contributing

If you find this idea interesting I would love help with developing it further, constructive critique (prior art checks, skeptical review), help and expertise with model training or compute for model training and testing. I don't have training compute or formal experience writing for academic venues.


## License and contact


CC-BY-4.0 for the proposal documents in this repository. Code in experiments/ is offered under MIT once present. If implemented and validated by anyone, credit should be shared between this proposal and the implementation; if implemented and refuted, the proposal stands as a hypothesis tested and falsified.

Contact: GitHub issues on this repository.
