#!/usr/bin/env bash

LM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$LM_DIR/bin:$PATH"

if [[ -z "${KALDI_ROOT:-}" && ( -x "$LM_DIR/../utils/kaldi/src/latbin/latgen-faster-mapped" || -x "$LM_DIR/../utils/kaldi/src/bin/latgen-faster-mapped" ) ]]; then
    KALDI_ROOT="$LM_DIR/../utils/kaldi"
fi
if [[ -n "${KALDI_ROOT:-}" ]]; then
    KALDI_ROOT="$(cd -- "$KALDI_ROOT" && pwd)" || return 1
    for dir in bin featbin fstbin gmmbin latbin lmbin; do
        export PATH="$KALDI_ROOT/src/$dir:$PATH"
    done
    export PATH="$KALDI_ROOT/tools/openfst/bin:$PATH"
    export LD_LIBRARY_PATH="$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst/lib:$KALDI_ROOT/tools/OpenBLAS/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ -z "${KENLM_ROOT:-}" && -x "$LM_DIR/../utils/kenlm/build/bin/lmplz" ]]; then
    KENLM_ROOT="$LM_DIR/../utils/kenlm"
fi
if [[ -n "${KENLM_ROOT:-}" ]]; then
    KENLM_ROOT="$(cd -- "$KENLM_ROOT" && pwd)" || return 1
    export PATH="$KENLM_ROOT/build/bin:$PATH"
fi
