#!/usr/bin/env bash
# ===========================================================================
#  GEM heterogeneous-macro  --  install every build/verify/benchmark dependency
#  on a fresh Linux or WSL2 (Ubuntu/Debian) machine.
#
#    bash scripts/install_deps.sh            install everything that is missing
#    bash scripts/install_deps.sh --check    only report what is present / missing
#
#  Installs:  build-essential, git, curl, Python 3 + reportlab,
#             Rust (rustup, stable),
#             Yosys 0.68 + Icarus Verilog + yosys-slang  (via OSS CAD Suite),
#  Checks / guides:  the CUDA Toolkit + NVIDIA driver (vendor download).
#
#  Safe to re-run.  Uses sudo only for apt and for /opt.
# ===========================================================================
set -uo pipefail
cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1
OSS_PREFIX=${OSS_CAD_PREFIX:-/opt/oss-cad-suite}
YOSYS_MIN="0.68"

say()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
ver()  { "$1" --version 2>&1 | head -1; }

report() {
    say "dependency status"
    for t in gcc git curl python3 pip3 rustc cargo yosys iverilog vvp nvcc nvidia-smi; do
        if have "$t"; then printf '  \033[1;32m OK \033[0m %-10s %s\n' "$t" "$(ver "$t")"
        else               printf '  \033[1;31mMISS\033[0m %-10s\n' "$t"; fi
    done
    if have yosys; then
        y=$(yosys --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
        [[ -n "$y" ]] && awk -v a="$y" -v b="$YOSYS_MIN" 'BEGIN{exit !(a+0 < b+0)}' \
            && printf '  \033[1;33mNOTE\033[0m yosys %s < %s (the PS asks for %s)\n' "$y" "$YOSYS_MIN" "$YOSYS_MIN"
    fi
}

[[ $CHECK_ONLY == 1 ]] && { report; exit 0; }

# ---------------------------------------------------------------- 1. apt base
if have apt-get; then
    say "apt: build tools, git, curl, python3"
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates xz-utils \
        python3 python3-pip libssl-dev pkg-config
else
    echo "WARNING: no apt-get. Install build-essential / git / curl / python3 with your package manager, then re-run."
fi

# --------------------------------------------------------- 2. Python: reportlab
say "pip: reportlab (only needed to regenerate the PDFs)"
python3 -m pip install --user --quiet reportlab || echo "  (reportlab optional — skipped)"

# ------------------------------------------------------------------- 3. Rust
if have cargo; then
    echo "rust already present: $(ver cargo)"
else
    say "rustup: stable Rust toolchain"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env" 2>/dev/null || true
    echo "  -> restart your shell, or 'source \$HOME/.cargo/env', before building"
fi

# --------------------------------------- 4. Yosys 0.68 + Icarus + slang (OSS CAD)
need_oss=1
if have yosys && have iverilog; then
    y=$(yosys --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [[ -n "$y" ]] && awk -v a="$y" -v b="$YOSYS_MIN" 'BEGIN{exit !(a+0 >= b+0)}'; then
        echo "yosys $y and iverilog already present — skipping OSS CAD Suite"
        need_oss=0
    fi
fi
if [[ $need_oss == 1 ]]; then
    say "OSS CAD Suite (bundles a current Yosys + yosys-slang + Icarus Verilog)"
    arch=$(uname -m)
    case "$arch" in
        x86_64)  key="linux-x64" ;;
        aarch64) key="linux-arm64" ;;
        *) echo "  unsupported arch $arch — install Yosys 0.68 + iverilog manually"; key="" ;;
    esac
    if [[ -n "$key" ]]; then
        api="https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases/latest"
        url=$(curl -sL "$api" | grep -oE "https://[^\"]*oss-cad-suite-${key}-[0-9]+\.tgz" | head -1)
        if [[ -z "$url" ]]; then
            echo "  could not resolve the latest release URL; set it by hand:"
            echo "    https://github.com/YosysHQ/oss-cad-suite-build/releases"
        else
            echo "  downloading $url"
            tmp=$(mktemp -d)
            curl -sL "$url" -o "$tmp/oss.tgz"
            sudo mkdir -p "$(dirname "$OSS_PREFIX")"
            sudo rm -rf "$OSS_PREFIX"
            sudo tar -xzf "$tmp/oss.tgz" -C "$(dirname "$OSS_PREFIX")"
            sudo mv "$(dirname "$OSS_PREFIX")/oss-cad-suite" "$OSS_PREFIX" 2>/dev/null || true
            rm -rf "$tmp"
            line="source \"$OSS_PREFIX/environment\""
            for rc in "$HOME/.bashrc" "$HOME/.profile"; do
                grep -qF "$line" "$rc" 2>/dev/null || echo "$line" >> "$rc"
            done
            echo "  installed to $OSS_PREFIX ; added 'source $OSS_PREFIX/environment' to your shell rc"
            # shellcheck disable=SC1091
            source "$OSS_PREFIX/environment" 2>/dev/null || true
        fi
    fi
fi

# ------------------------------------------------------------------- 5. CUDA
say "CUDA Toolkit + NVIDIA driver"
if have nvcc && have nvidia-smi; then
    echo "  present: $(ver nvcc)"
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
else
    cat <<'EOF'
  nvcc / nvidia-smi not found. The CUDA Toolkit is a vendor download and is not
  installed by this script. On WSL2 / Ubuntu:

    # 1. Windows side: install a recent NVIDIA game/studio driver (gives WSL a GPU)
    # 2. Ubuntu side  (CUDA >= 12.8 recommended for native RTX 50-series / Blackwell):
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc

  Native Linux: pick your distro at https://developer.nvidia.com/cuda-downloads
EOF
fi

# ------------------------------------------------------------------- 6. verify
report
say "next"
echo "  1. open a NEW shell (so PATH picks up Rust / OSS CAD Suite / CUDA)"
echo "  2. from the repo root:   ./compile.sh          (or compile.bat on Windows)"
