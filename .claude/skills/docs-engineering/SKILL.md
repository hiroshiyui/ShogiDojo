---
name: docs-engineering
description: Write and maintain technical documentation for the ShogiDojo Shogi-AI research project (NNUE, AlphaZero, and Bonanza paradigms). Covers README, reference guides, inline code comments, config annotations, and TODO/roadmap updates.
---

# Docs Engineering Skill

You are maintaining documentation for **ShogiDojo**, an academic Shogi AI research repository covering three complementary paradigms side by side:

- **NNUE** (HalfKAv2) — sparse, incrementally-updatable evaluation for alpha-beta search. Primary production pipeline. PyTorch trainer + Rust `gensfen` data generator.
- **AlphaZero-style** — ResNet policy/value network + PUCT MCTS + self-play. Research survey; educational rather than competitive.
- **Bonanza-method** — supervised policy + value training from professional kifu records. Reuses the AlphaZero network; simplest of the three.

Each paradigm lives under its own `nnue/` / `alphazero/` / `bonanza/` subdirectory across `src/`, `configs/`, `tests/`, and `doc/`. The audience is research group members who may be new to one or more of these paradigms.

## Documentation inventory

| File | Purpose | Update when |
|------|---------|-------------|
| `README.md` | User-facing guide: setup, quick start for all three paradigms, project structure, disk budget | CLI flags, dependencies, or workflow changes in any paradigm |
| `CLAUDE.md` | Developer guidance: repo layout, architecture flow per paradigm, design invariants | Internal design decisions or module contracts change |
| `doc/TODO.md` | Development roadmap, research directions, phase tracking | Tasks complete or new work planned |
| `doc/nnue/NNUE_Reference.md` | Technical reference: HalfKAv2, SFEN format, king buckets, feature encoding | NNUE algorithm or encoding changes |
| `doc/nnue/Model_Training_Reference.md` | NNUE training methodology: dual loss, score scaling, gensfen tuning, eval matches | Training-loop design or hyperparameter guidance changes |
| `doc/alphazero/AlphaZero_Reference.md` | AlphaZero architecture, MCTS+PUCT, Shogi encoding, glossary | Architecture, MCTS, or encoding changes |
| `doc/alphazero/AlphaZero_Training_Reference.md` | Self-play methodology, replay buffer, hyperparameters, failure modes | Training-loop design, hyperparameter guidance, or new failure modes |
| `doc/alphazero/AlphaZero_vs_NNUE.md` | Operational side-by-side comparison | Architectural or deployment trade-offs change |
| `doc/alphazero/AlphaZero_Reading_List.md` | Annotated bibliography | New recommended papers or open-source references |
| `doc/bonanza/Bonanza_Reference.md` | History, supervised-classification formulation, CSA format, three-paradigm comparison, data sources | Formulation, data pipeline, or dataset recommendations change |
| `configs/nnue/{default,small,large}.yaml` | Inline comments on every NNUE hyperparameter | New config keys or default changes |
| `configs/alphazero/{tiny,default}.yaml` | Inline comments on every AlphaZero hyperparameter | New config keys, self-play knobs, or warm-start keys |
| `configs/bonanza/{tiny,default}.yaml` | Inline comments on every Bonanza hyperparameter | New config keys or defaults |

## Writing guidelines

1. **Read before writing.** Always read the target doc and related source files before proposing changes. Documentation must match the current code — never describe behaviour you haven't verified.

2. **Audience awareness.** Members range from experienced ML engineers to students encountering NNUE or MCTS for the first time. Lead with the practical (what to do), then explain the conceptual (why it works that way). Use the glossaries in the paradigm reference docs for domain terms.

3. **Keep docs DRY.** Each fact lives in one canonical place. Cross-reference rather than duplicate:
   - Setup instructions belong in `README.md` only.
   - Design invariants (e.g. -1 padding convention, king bucket formula, perspective sign-flip on MCTS backup) belong in `CLAUDE.md` only.
   - NNUE algorithm details belong in `doc/nnue/NNUE_Reference.md`.
   - AlphaZero algorithm details belong in `doc/alphazero/AlphaZero_Reference.md`.
   - Bonanza method details belong in `doc/bonanza/Bonanza_Reference.md`.
   - Cross-paradigm comparisons belong in `doc/alphazero/AlphaZero_vs_NNUE.md` (soon to be generalized if three-way comparisons accumulate).

4. **Concrete over abstract.** Prefer code snippets, CLI examples, and formulas over vague prose. Show the exact command, the exact config key, the exact tensor shape.

5. **Sync with code.** When updating source code that affects documented behaviour, update the corresponding docs in the same change. Key sync points:
   - `src/shogi_dojo/nnue/constants.py` values (`FEATURE_TRANSFORMER_INPUT_DIM = 19638`, 9 king buckets, `HALF_FEATURE_DIM = 2182`) referenced in `CLAUDE.md` and `doc/nnue/NNUE_Reference.md`.
   - `src/shogi_dojo/alphazero/encoding.py` dimensions (`NUM_INPUT_CHANNELS = 45`, `POLICY_DIM = 11259` = 9×9×139) referenced in `doc/alphazero/AlphaZero_Reference.md` and the comparison doc.
   - Config defaults referenced in `README.md` quick-start sections.
   - CLI arguments in `train_nnue.py`, `train_alphazero.py`, `train_bonanza.py`, and `tools/nnue/gensfen` referenced in `README.md`.

6. **TODO.md conventions.** Use checkboxes (`- [x]` done with completion date, `- [ ]` pending). Group by phase or by research topic. When all tasks in a section are completed, **summarize the section into a short paragraph and mark the heading with "— Done"**. This keeps the roadmap compact and focused on remaining work.

7. **Style rules.**
   - Write in plain, direct English. Avoid filler words.
   - Use ATX headings (`#`, `##`, `###`). Do not skip heading levels.
   - Wrap lines at a natural sentence boundary (no hard line-length limit).
   - Code blocks must specify the language (```python, ```rust, ```yaml, ```bash).
   - Use tables for structured comparisons. Use bullet lists for sequences of items.

8. **Junior-friendly code comments.** Many contributors are students or early-career developers. Write inline comments assuming the reader has basic Python/Rust knowledge but no prior exposure to NNUE, MCTS, shogi engines, or ML training pipelines.
   - **Explain _why_, not just _what_.** `# Clamp to [0,1] for quantization compatibility` beats `# apply clamp`.
   - **Define domain terms on first use.** E.g. `# SFEN: a compact text format describing a shogi board position (similar to FEN in chess)`.
   - **Annotate tensor shapes.** Whenever a tensor is created or reshaped, comment its shape: `# (B, 139, 9, 9) — policy logits, one plane per action type`.
   - **Flag non-obvious conventions.** The codebase has subtle invariants (e.g. NNUE's -1 padding, king bucket indexing, MCTS's perspective sign-flip on backup, AlphaZero's value from STM perspective). Comment these at the point of use, not just in CLAUDE.md.
   - **Keep comments close to the code they describe.** Prefer a short comment on the line above over a block comment paragraphs away.
   - **Use plain language.** Avoid jargon-heavy shorthand. Write `side to move` not `STM` on first use in a file.

9. **Config annotation style.** In every `configs/**/*.yaml`, every key must have a comment explaining its purpose, valid range, and effect. Format:
   ```yaml
   # Learning rate for Adam optimizer. Range: 1e-5 to 1e-2.
   # Higher values converge faster but risk instability.
   lr: 0.001
   ```

## Typical tasks

- **Add docs for a new feature:** Read the implementation, then update the appropriate doc(s) per the inventory table. Add a usage example. If the feature crosses paradigms (e.g. AlphaZero warm-start from a Bonanza checkpoint), add cross-references both ways.
- **Fix stale docs:** Diff the doc's claims against current code. Correct any mismatches. Remove references to deleted features.
- **Expand a reference doc:** Add new sections following the existing structure (heading, explanation, formula/diagram, example). Maintain the glossary at the end.
- **Update TODO.md:** Mark completed items with their completion date, add new items under the correct phase, keep the roadmap realistic. Collapse fully-completed sections into single-paragraph summaries marked with "— Done".
