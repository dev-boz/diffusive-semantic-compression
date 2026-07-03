Here is a list of possible extra features which could make for a highly tuneable system:

-Selective reading: the early slices rate which verbatim slices are most important. Only important verbatim slices are read

-Guided reading: early slices leave messages for later slices (“this detail is important”)

-Confidence based early stoppage. Each pass outputs a confidence score, if enough of the early outputs align then output is finalised early (if the output is short yes/no or a tool call)

-Concurrent prefill: prefill the slice+prompt to multiple LLM threads at the same time before and during output to save time

-Slice overlapping control: Allow the compressed slices to overlap or not, high overlapping = less lossy but high cost+time.

-Reverse reading: a later pass suggests re-reading an earlier pass to get structure

-Pass skipping: Skip n slices, more lossy but lower cost+time.
