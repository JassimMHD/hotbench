# Contributing to Hotbench

Thanks for helping. Hotbench is a live test bench for local models: run them,
convert them, and measure what they cost. Most of its value comes from the
adapters, converters, recipes and benchmark results that people add. This page tells you how to get a change merged quickly.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to help

| You have… | Do this |
|---|---|
| 5 minutes | try the quick start and open an issue for anything that was unclear |
| a model that doesn't work | open a **Bug report** with `hotbench doctor` output |
| a model you want supported | open an **Adapter request**, or write the adapter (below) |
| a model family you know how to export (ONNX, OpenVINO, GGUF, …) | open an **Adapter or converter request** (choose *converter*), or write the converter |
| numbers from your machine | `hotbench bench --share` and send the run file to `benchmarks/` |
| an idea that changes the API, CLI or run-file format | open an **RFC** issue first; don't start with a PR |

Issues labelled [`good first issue`](../../labels/good%20first%20issue) are
small and come with pointers to the code involved.

## Development setup

> **Pre-alpha:** until v0.1 there is no `pyproject.toml`, test suite or
> `hotbench` command yet. Set up the prototype as the
> [README's quick start](README.md#quick-start) shows, and run it with
> `python app.py`. The setup below is how it will work from v0.1.

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/JassimMHD/hotbench && cd hotbench
uv sync --extra dev --extra vision      # add --extra speech / llm as needed
uv run pre-commit install
uv run pytest                           # fast tests: a tiny model, no downloads
uv run hotbench run tests/assets/tiny.onnx --source tests/assets/clip.mp4
```

Slow tests that download real models are marked `slow` and run nightly:
`uv run pytest -m slow`.

## Making a change

1. For anything bigger than a small fix, comment on the issue first so two
   people don't build the same thing.
2. Branch from `main`: `feat/openvino-adapter`, `fix/rtsp-reconnect`.
3. Keep a PR to one change. Several small PRs are reviewed faster than one big one.
4. Add or update tests. A bug fix comes with a test that failed before it.
5. Add a line under **Unreleased** in `CHANGELOG.md`.
6. Run `uv run ruff check . && uv run ruff format --check . && uv run pytest`.
7. Sign off every commit (see DCO below), push, and open the PR using the template.

A maintainer will reply within a few days. If you hear nothing for a week,
leave a comment on the PR.

### Commit messages

[Conventional Commits](https://www.conventionalcommits.org):
`feat(vision): add OpenVINO adapter`, `fix(sources): reconnect after RTSP EOF`,
`docs: …`, `perf: …`, `test: …`, `chore: …`.

### Developer Certificate of Origin (DCO)

We use the [DCO](https://developercertificate.org) instead of a CLA. Sign off
each commit to confirm that you wrote the change, or have the right to submit it:

```bash
git commit -s -m "feat(llm): add Ollama adapter"
# adds: Signed-off-by: Your Name <you@example.com>
```

Forgot? `git commit --amend -s`, or `git rebase --signoff main` for several commits.

## Writing an adapter

An adapter teaches Hotbench to run one kind of model. It doesn't decide where
input comes from or how results are shown. The core handles sources, threads,
metrics and the UI.

```python
from hotbench import Adapter, ModelSpec, register
from hotbench.types import Image, Labels

@register("timm", tasks=["classify"], files=["timm:*"], extra="vision")
class Timm(Adapter):
    def load(self, spec: ModelSpec, device: str, threads: int) -> None:
        import timm, torch                     # import heavy libraries here, not at module top
        torch.set_num_threads(threads)
        self.model = timm.create_model(spec.name, pretrained=True).eval().to(device)

    def warmup(self) -> None:
        ...                                    # one dummy call; never measured

    def __call__(self, item: Image) -> Labels:
        ...                                    # return a typed result, never a raw tensor
```

**Checklist for an adapter PR:**

- [ ] Heavy imports happen inside `load()`, so `import hotbench` stays fast
      without the extra installed.
- [ ] It respects `threads` and `device`. Unbounded thread pools make the cost
      numbers wrong (ONNX Runtime starts one thread per logical CPU unless told
      otherwise).
- [ ] It returns a type from `hotbench.types`. If none fits, propose a new
      type in an RFC.
- [ ] It never prints or logs secrets (URLs with passwords, API keys).
- [ ] A test with a tiny or mocked model that runs without network access.
- [ ] A recipe in `recipes/` and a docs page with the install command and a
      licence note for the library and its weights.
- [ ] The library's licence is compatible with Apache-2.0, or it is an optional
      extra and the docs say so (Ultralytics is AGPL-3.0, for example).

**Where it goes:** core adapters (widely used formats) live in
`src/hotbench/adapters/`. Model-specific or niche ones go in `contrib/`, where
you are listed as their owner in `CODEOWNERS`, or in your own
`hotbench-<name>` package registered through the `hotbench.adapters` entry
point. All three are equally welcome. Ask in the issue if you are unsure which fits.

## Benchmark results

Run files go in `benchmarks/<task>/<model>/<hardware-slug>.json`. Make them with
`hotbench bench --share`, which strips paths and URLs and adds the hardware
fingerprint. Don't edit the numbers by hand. One PR may add many files.

## Docs

Docs live in `docs/` (MkDocs Material). Preview them with `uv run mkdocs serve`.
Short sentences, one idea each, and a runnable command wherever possible.

## Reporting security issues

Don't open a public issue. See [SECURITY.md](SECURITY.md).
