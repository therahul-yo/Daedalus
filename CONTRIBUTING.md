# Contributing to Daedalus

Thanks for your interest in Daedalus — a thermally-governed MLX inference
engine built for fanless Apple Silicon. Contributions of all sizes are
welcome: bug reports, benchmark results from your hardware, docs fixes, and
code.

By participating you agree to our [Code of Conduct](CODE_OF_CONDUCT.md).
Security issues go through [SECURITY.md](SECURITY.md) (private advisories),
not public issues.

## Development setup

You need a Mac with Python 3.12+ and [uv](https://docs.astral.sh/uv/). Apple
Silicon is required to actually run models; the unit test suite itself uses
fake engines and never loads real weights, so it runs fine on machines that
can't sustain Metal inference.

```bash
git clone https://github.com/therahul-yo/Daedalus.git
cd Daedalus
uv sync --extra dev
```

## Before you open a PR

Run the same checks CI runs:

```bash
uv run --extra dev pytest -q       # full suite, must be green
uv run ruff check daedalus tests   # lint, must be clean
uv lock --check                    # if you touched pyproject.toml
```

Every behavior change needs a test. The suite is fast (a few seconds) because
server and cache tests run against fake engines — follow that pattern; a test
that downloads a model or needs Metal does not belong in `tests/`. Real-model
verification lives in `bench/` and the manual `real-model-verify` workflow
instead.

Keep PRs focused: one concern per PR, with the *why* in the description. The
PR template will ask what you changed, why, and how you verified it.

## Performance changes are special

Daedalus makes measured performance claims on thermally-constrained hardware,
so a speed change is held to a higher bar than a bug fix:

- State which path it improves: cold prefill, RAM cache hit, disk cache hit
  (restart), or decode — and show the others didn't regress.
- Attach fingerprinted benchmark artifacts from `bench/bench.py` (run with
  `--require-nominal` on a cooled machine). `bench/regression.py` compares
  two artifacts and refuses mismatched configurations — compare like with
  like.
- A shorter wall clock that pushes the machine into sustained thermal
  throttling is not an improvement. Report thermal state before and after.

Hosted CI runners cannot sustain Metal inference, so real-hardware numbers
from contributors (especially non-Air machines) are genuinely valuable — an
issue with a benchmark artifact attached is a great contribution on its own.

## Code style

- `ruff check` must pass; match the style of the surrounding code.
- Comments explain constraints the code can't ("deepcopy is required here
  because the cache is mutated during decode"), not what the next line does.
- Public behavior changes should update the relevant doc (`README.md`,
  `docs/`, or `MULTIMODEL_DESIGN.md`).

## Reporting bugs and requesting features

Use the issue templates — the bug template asks for your macOS version and
hardware because on this project "works on my machine" often literally depends
on the machine's temperature. Performance regressions have their own template;
include benchmark artifacts if you can.

## Questions

Check the [documentation](https://daedalus-mlx.vercel.app/docs) first, then
open a GitHub issue. There is no chat server; the issue tracker is the source
of truth.
