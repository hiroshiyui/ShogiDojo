---
name: docs-engineering
description: Write and maintain technical documentation for the ShogiDojo NNUE training project. Covers README, reference guides, inline code comments, config annotations, and TODO/roadmap updates.
---

# Docs Engineering Skill

You are maintaining documentation for **ShogiDojo**, an academic Shogi NNUE training framework (PyTorch + Rust gensfen). The audience is research group members who may be new to NNUE or the codebase.

## Documentation inventory

| File | Purpose | Update when |
|------|---------|-------------|
| `README.md` | User-facing guide: setup, quick start, CLI usage, disk budget | CLI flags, dependencies, or workflow changes |
| `CLAUDE.md` | Developer guidance: architecture flow, design invariants, conventions | Internal design decisions or module contracts change |
| `doc/NNUE_Reference.md` | Deep technical reference on NNUE, HalfKAv2, SFEN format | Algorithm or encoding changes |
| `doc/TODO.md` | Development roadmap with phase tracking | Tasks are completed or new work is planned |
| `configs/default.yaml` | Inline comments on every hyperparameter | New config keys are added or defaults change |

## Writing guidelines

1. **Read before writing.** Always read the target doc and related source files before proposing changes. Documentation must match the current code — never describe behaviour you haven't verified.

2. **Audience awareness.** Members range from experienced ML engineers to students encountering NNUE for the first time. Lead with the practical (what to do), then explain the conceptual (why it works that way). Use the glossary in `doc/NNUE_Reference.md` for domain terms.

3. **Keep docs DRY.** Each fact lives in one canonical place. Cross-reference rather than duplicate:
   - Setup instructions belong in `README.md` only.
   - Design invariants (e.g. -1 padding convention, king bucket formula) belong in `CLAUDE.md` only.
   - Algorithm details belong in `doc/NNUE_Reference.md` only.

4. **Concrete over abstract.** Prefer code snippets, CLI examples, and formulas over vague prose. Show the exact command, the exact config key, the exact tensor shape.

5. **Sync with code.** When updating source code that affects documented behaviour, update the corresponding docs in the same change. Key sync points:
   - `constants.py` values (feature count 19,638; king buckets 9) referenced in `CLAUDE.md` and `doc/NNUE_Reference.md`
   - `configs/default.yaml` defaults referenced in `README.md`
   - CLI arguments in `train.py` and `tools/gensfen` referenced in `README.md`

6. **TODO.md conventions.** Use checkboxes (`- [x]` done, `- [ ]` pending). Group by phase. When marking a task complete, add the date in parentheses. When adding new tasks, place them under the appropriate phase heading. **When all tasks in a section are completed, summarize the section into a short paragraph and mark the heading with "— Done".** This keeps the roadmap compact and focused on remaining work.

7. **Style rules.**
   - Write in plain, direct English. Avoid filler words.
   - Use ATX headings (`#`, `##`, `###`). Do not skip heading levels.
   - Wrap lines at a natural sentence boundary (no hard line-length limit).
   - Code blocks must specify the language (```python, ```rust, ```yaml, ```bash).
   - Use tables for structured comparisons. Use bullet lists for sequences of items.

8. **Junior-friendly code comments.** Many contributors are students or early-career developers. Write inline comments assuming the reader has basic Python/Rust knowledge but no prior exposure to NNUE, shogi engines, or ML training pipelines.
   - **Explain _why_, not just _what_.** `# Clamp to [0,1] for quantization compatibility` beats `# apply clamp`.
   - **Define domain terms on first use.** E.g. `# SFEN: a compact text format describing a shogi board position (similar to FEN in chess)`.
   - **Annotate tensor shapes.** Whenever a tensor is created or reshaped, comment its shape: `# (batch, 2, max_features) — two perspectives: STM and NSTM`.
   - **Flag non-obvious conventions.** The codebase has subtle invariants (e.g. -1 padding, king bucket indexing). Comment these at the point of use, not just in CLAUDE.md.
   - **Keep comments close to the code they describe.** Prefer a short comment on the line above over a block comment paragraphs away.
   - **Use plain language.** Avoid jargon-heavy shorthand. Write `side to move` not `stm` the first time it appears in a file.

9. **Config annotation style.** In `configs/default.yaml`, every key must have a comment explaining its purpose, valid range, and effect. Format:
   ```yaml
   # Learning rate for Adam optimizer. Range: 1e-5 to 1e-2.
   # Higher values converge faster but risk instability.
   lr: 0.001
   ```

## Typical tasks

- **Add docs for a new feature:** Read the implementation, then update the appropriate doc(s) per the inventory table. Add a usage example.
- **Fix stale docs:** Diff the doc's claims against current code. Correct any mismatches. Remove references to deleted features.
- **Expand NNUE_Reference.md:** Add new sections following the existing structure (heading, explanation, formula/diagram, example). Maintain the glossary at the end.
- **Update TODO.md:** Mark completed items, add new items under the correct phase, keep the roadmap realistic.
