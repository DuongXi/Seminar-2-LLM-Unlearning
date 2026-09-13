# TSV reference material (archived, not integrated)
Source: [`deeplearning-wisc/tsv`](https://github.com/deeplearning-wisc/tsv) --
code for *Steer LLM Latents for Hallucination Detection* (Park, Du, Yeh,
Wang, Li; ICML 2025), arXiv:2503.01917. Apache-2.0 licensed -- `LICENSE` in
this folder is copied verbatim from that repo, per the license's
redistribution terms.

```bibtex
@inproceedings{park2025steer,
  title={Steer {LLM} Latents for Hallucination Detection},
  author={Seongheon Park and Xuefeng Du and Min-Hsuan Yeh and Haobo Wang and Yixuan Li},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```

**Status: pure archival.** Nothing here has been modified, adapted, or wired
into `pkg_halluc`. It's not imported by anything, has no tests, and isn't
part of the installed package. This is reference material for whoever
implements `pkg_halluc/methods/steering.py` later.

## Why only these two files

TSV's own method (a trainable vector learned via MLE on a von-Mises-Fisher
class model, semi-supervised with Optimal-Transport pseudo-labeling -- see
the paper's Section 4) solves a different problem than what the proposal
plans: TSV needs pseudo-labeling machinery because *ground-truth labels are
expensive* in its setting (is this QA answer truthful? -- needs a human or
GPT-4o judge). Package hallucination doesn't have that problem: whether a
package name is real is a **free, deterministic PyPI/npm lookup** --
literally what `build_valid_packages.py` / `package_detection.py` in this
project already do. There's no labeled-data scarcity to work around, so
there's no reason to reproduce TSV's semi-supervised training loop.

What *is* reusable, independent of which statistical method computes the
steering vector, is the **mechanism for adding a vector to a model's hidden
state at a chosen layer during generation** -- that's generic activation-
engineering infrastructure, not specific to TSV's training objective. That's
`llm_layers.py`, kept in full:

| Piece | What it does |
| --- | --- |
| `get_layers`, `get_layers_path`, `find_longest_modulelist`, `find_module`, `get_nested_attr`/`set_nested_attr` | Architecture-agnostic: find a model's decoder layer list and named submodules (`mlp`, `self_attn`, ...) without hardcoding a specific model class. |
| `TSVLayer` | The actual intervention: `x -> x + lam * v` (Eq. 2 in the paper: `h^(l) <- h^(l) + lambda*v`). Generic -- works with *any* vector `v`, not just one trained via TSV's objective. This is what the proposal's "Intervention: add the (optionally confidence-scaled) steering vector to the hidden state at generation time" needs. |
| `add_tsv_layers`, `LlamaDecoderLayerWrapper` | Monkey-patches a loaded model to insert `TSVLayer` at a chosen layer index, in one of three places: residual stream (`component='res'`), MLP output (`'mlp'`), or attention output (`'attn'`). Matches the proposal's ablation plan ("multi-layer steering") and the paper's own ablation (Fig. 3a: residual stream in early-middle layers works best). |

`hidden_state_utils.py::get_last_non_padded_token_rep` (extracted from
`train_utils.py`) is the small utility for pulling out "the last real
token's hidden state" from a padded batch -- needed for both RQ1 (probing:
extract activations at the package-name generation position) and steering
vector construction (mean-difference needs per-example activation vectors to
average).

**Left out:**

- `sinkhorn_knopp.py`, and everything else in `train_utils.py` (OT
  pseudo-labeling, centroid EMA updates) -- TSV-specific semi-supervised
  training, not needed (see above).
- `tsv_main.py` -- the full original script. Not copied, but worth reading
  directly from `tsv-main.zip` if you want to see `add_tsv_layers` /
  `TSVLayer` used end-to-end (the `test_model` function there is the
  clearest example of inference-time usage).
- `cache_utils.py` -- a frozen copy of an old internal `transformers.Cache`
  class, kept only for a type hint. Not needed; `transformers==4.57.6`
  (this project's pinned version) has its own.
- `data_indices/*.npy`, `gen.sh`, `gt.sh`, `train.sh`, `tsv.yml`,
  `requirements.txt` -- tied to TSV's own QA benchmarks (TruthfulQA /
  TriviaQA / SciQ / NQ Open) and a Python 3.8.15 env. Not applicable.

## Known issues / adaptation needed before use

Nothing below has been fixed -- this is a list for whoever picks this up
next, not a promise that the code works as-is:

- **Hardcoded dtype.** `TSVLayer.forward` calls `.half()` unconditionally.
  This project resolves dtype per-GPU (`config.py::resolve_dtype` --
  bfloat16 when supported, float16 otherwise); `llm_layers.py` needs to
  respect that instead of assuming fp16.
- **Hardcoded `.cuda()` / device handling in the wider TSV codebase** (not
  in the two files kept here, but worth knowing if you go back to
  `tsv_main.py` for reference) -- this project generally uses
  `device_map="auto"`.
- **`transformers` version drift.** `LlamaDecoderLayerWrapper.forward` calls
  `self.llama_decoder_layer.self_attn(...)` with a hand-written argument
  list that matches whatever `transformers` version TSV was built against
  (old enough to pin Python 3.8.15). `transformers==4.57.6` (this project's
  pinned version) may have a different `self_attn` call signature --
  **this needs to be checked against the actual installed version before
  use**, not assumed to work.
- **Untested on Qwen2.5.** TSV's `LlamaDecoderLayerWrapper` has a special
  case for `model_name == 'qwen2.5-7B'` (skips passing `position_embeddings`
  to `self_attn`), implying it *was* made to work with a Qwen2.5 model at
  some point -- but that was presumably against whatever older
  `transformers` TSV pinned, not this project's. Re-verify against
  `Qwen/Qwen2.5-0.5B-Instruct` (this project's default model) specifically.
- **Padding side.** `get_last_non_padded_token_rep` assumes **right**-padded
  sequences (it indexes `hidden_states[i, lengths[i]-1, :]`, which only
  lands on the last real token if padding comes *after* it). Worth noting:
  this project's own eval pipeline has already surfaced a `transformers`
  warning about right-padding being detected on a decoder-only model (see
  `Package_Hallucination_Testing/generate_code.py`'s tokenizer setup, via
  the vendored AU code) -- so this function's assumption may or may not
  match whatever tokenizer config the steering implementation ends up
  using. Check `padding_side` explicitly rather than assuming.

## How this maps to the proposal's Representation Steering plan

From `Proposal.docx`, Methods > Representation Steering:

| Proposal step | Where to start |
| --- | --- |
| "Probing: extract hidden states at the package-name generation position across all layers" | `hidden_state_utils.get_last_non_padded_token_rep` + `output_hidden_states=True` on a normal HF forward pass (see `tsv_main.py::get_ex_data` in the original zip for the pattern -- not copied here, TSV-specific glue around it). |
| "Steering vector construction: mean-difference vector between real- and hallucinated-package activation clusters" | New code, not in TSV at all (TSV computes a *trained* vector via MLE, not a closed-form mean-difference / ITI-style vector). Labels come for free from this project's own PyPI lookup, per above -- no pseudo-labeling needed. |
| "Intervention: add the ... steering vector to the hidden state at generation time" | `llm_layers.add_tsv_layers` + `TSVLayer`, after the fixes listed above. |
| "Variants ... multi-layer steering; adaptive steering strength scaled by probe confidence" | `add_tsv_layers` already supports picking a single `str_layer` + `component`; multi-layer would mean calling it (or a generalized version) at several layers at once -- not implemented in the original either. |

See also `pkg_halluc/methods/steering.py`'s module docstring for how
this slots into the pipeline's method registry once someone starts on it.
