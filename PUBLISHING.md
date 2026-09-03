# Publishing to GitHub

Deliverable (1) is a GitHub URL. This tree is ready to push as-is.

## One time

```bash
cd GEM-heterogeneous-macro

git init                       # if this is not already a git repo
git add -A
git commit -m "GEM heterogeneous FPGA-macro simulation (DSP48E2 / CARRY4 / SRLC32E)"

# create an EMPTY repo on github.com first (no README/licence), then:
git remote add origin https://github.com/<your-org>/<your-repo>.git
git branch -M main
git push -u origin main
```

## What is and isn't committed

`.gitignore` already excludes build output (`target/`, `build/`,
`verify_logs/`, `submission-results/`, `*.gv`, `*.gemparts`, `*.log`,
`__pycache__/`). Everything else — source, scripts, docs, PDFs, fixtures,
recorded `benchmark-results/` — is tracked and should be pushed.

The vendored Rust crates in `eda-infra-rs/` are a normal part of the tree
(not a submodule) — they are committed directly so the build is self-contained.

## Large files

`docs/*.pdf` (~200 KB each) and `benchmark-results/part_b_v2.ncu-rep` (~1–2 MB)
are committed directly; they are well under GitHub's 100 MB limit, so Git LFS is
not required.

## Before you submit the URL

1. `git clone` your pushed repo into a fresh directory.
2. Run `bash compile.sh --quick` (or `compile.bat --quick` on Windows) in the
   clone to confirm it builds and the functional gates pass from a clean state.
3. Check `README.md` renders correctly on the GitHub web page.
