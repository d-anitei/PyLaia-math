#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo 'Usage: bash scripts/install_lm_tools.sh' >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$ROOT"
JOBS="${LM_BUILD_JOBS:-2}"
KALDI_COMMIT=71f38e62cad01c3078555bfe78d0f3a527422d75
KENLM_COMMIT=4cb443e60b7bf2c0ddf3c745378f76cb59e254e5
OPENFST_VERSION=1.7.2
OPENBLAS_VERSION=0.3.13

for tool in git cmake make gcc g++ gfortran wget tar gzip perl gawk bc autoconf automake libtoolize patch unzip bzip2 sox svn; do
    if ! command -v "$tool" >/dev/null; then
        echo "Required build tool is missing: $tool. See the PyLaia README." >&2
        exit 1
    fi
done

mkdir -p utils/logs
LOG="$ROOT/utils/logs/install.log"
trap 'echo "Installation stopped. See $LOG for details." >&2' ERR
echo "Build log is at $LOG"

echo '[1/5] Downloading the pinned official Kaldi source...'
if [[ ! -d utils/kaldi ]]; then
    git clone https://github.com/kaldi-asr/kaldi.git utils/kaldi >> "$LOG" 2>&1
fi
git -C utils/kaldi checkout --detach "$KALDI_COMMIT" >> "$LOG" 2>&1

echo '[2/5] Building OpenFst...'
(
    cd utils/kaldi/tools
    # Kaldi's official recipe downloads and configures its supported OpenFst.
    make -j "$JOBS" openfst OPENFST_VERSION="$OPENFST_VERSION"
) >> "$LOG" 2>&1

echo '[3/5] Building OpenBLAS and the Kaldi decoding tools...'
if [[ ! -f utils/kaldi/tools/OpenBLAS/install/lib/libopenblas.a ]]; then
    if [[ ! -d utils/kaldi/tools/OpenBLAS ]]; then
        wget -O utils/kaldi/tools/openblas-source.tar.gz \
            "https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v${OPENBLAS_VERSION}.tar.gz" >> "$LOG" 2>&1
        tar -xzf utils/kaldi/tools/openblas-source.tar.gz -C utils/kaldi/tools >> "$LOG" 2>&1
        mv "utils/kaldi/tools/OpenBLAS-${OPENBLAS_VERSION}" utils/kaldi/tools/OpenBLAS
    fi
    # A fixed CPU target avoids failed CPU detection in this older OpenBLAS.
    make -C utils/kaldi/tools/OpenBLAS -j "$JOBS" TARGET=NEHALEM USE_LOCKING=1 USE_THREAD=0 \
        PREFIX="$ROOT/utils/kaldi/tools/OpenBLAS/install" all install >> "$LOG" 2>&1
fi
(
    cd utils/kaldi/src
    ./configure --shared --use-cuda=no --with-cudadecoder=no \
        --mathlib=OPENBLAS --openblas-root="$ROOT/utils/kaldi/tools/OpenBLAS/install"
    make depend -j "$JOBS"
    make -j "$JOBS" bin featbin fstbin gmmbin latbin lmbin
) >> "$LOG" 2>&1

echo '[4/5] Building KenLM from the pinned official source...'
if [[ ! -d utils/kenlm/.git ]]; then
    git clone https://github.com/kpu/kenlm.git utils/kenlm >> "$LOG" 2>&1
fi
git -C utils/kenlm checkout --detach "$KENLM_COMMIT" >> "$LOG" 2>&1
cmake -S utils/kenlm -B utils/kenlm/build -DCMAKE_BUILD_TYPE=Release \
    -DKENLM_MAX_ORDER=16 >> "$LOG" 2>&1
cmake --build utils/kenlm/build --target lmplz --parallel "$JOBS" >> "$LOG" 2>&1

echo '[5/5] Checking that the required tools were built...'
KALDI_ROOT="$ROOT/utils/kaldi"
KENLM_ROOT="$ROOT/utils/kenlm"
source "$SCRIPT_DIR/path_lm.sh"
for tool in feat-to-dim latgen-faster-mapped \
    lattice-add-penalty lattice-best-path arpa2fst fstcompile fstarcsort \
    fstprint fstrmepsilon fstconvert fstaddselfloops fstisstochastic fsttablecompose \
    fstdeterminizestar fstminimizeencoded fstpushspecial fstcomposecontext \
    make-h-transducer fstrmsymbols fstrmepslocal add-self-loops am-info \
    gmm-init-mono gmm-copy copy-tree gmm-info tree-info lmplz; do
    TOOL_PATH="$(command -v "$tool" || true)"
    if [[ "$TOOL_PATH" != "$ROOT/utils/"* ]]; then
        echo "Installation incomplete: $tool is missing." >&2
        exit 1
    fi
done

echo
echo 'Language-model tools installed successfully.'
echo '  Software: utils/kaldi and utils/kenlm'
echo '  Build log: utils/logs/install.log'
echo '  The PyLaia decoder loads these tools automatically.'
