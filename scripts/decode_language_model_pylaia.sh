#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C.UTF-8

if (( $# < 2 || $# > 3 )); then
    echo "Usage: ${0##*/} <ngram-order> <score-matrix.ark> [training-transcripts.txt]" >&2
    exit 1
fi

order="$1"
matrix="$(realpath -e -- "$2")"
name="$(basename -- "$matrix" .ark)"
name="${name#ConfMats_example_}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
utils_dir="$script_dir/../utils"
training_transcripts=("$script_dir/../examples/tr.txt" "$script_dir/../examples/va.txt")
if (( $# == 3 )); then
    training_transcripts=("$(realpath -e -- "$3")")
fi
source "$script_dir/path_lm.sh"

# Use the acoustic scale and insertion penalty selected on validation data.
read -r asf wip < <(awk '
    $1 == "Optimum" && $2 == "command:" { asf = $4; wip = $5 }
    END { if (asf != "" && wip != "") print asf, wip }
' "$utils_dir/simplex_out")

output_dir="$PWD/EXP/LM"
mkdir -p -- "$output_dir"
result="$output_dir/$name-lm-$order"
work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT
cd -- "$work_dir"

blank='<ctc>'
dummy='<DUMMY>'
mkdir -p data/train HMM-dir
sort -k2,2n "$utils_dir/symb_mixt.txt" > symbols.txt
prepare_lang_cl-ds.sh data/train symbols.txt "$blank" '' "$dummy"

# Each row starts with an image ID. Keep the model tokens, including LaTeX backslashes.
awk '{$1 = ""; sub(/^ /, ""); if (NF) print}' "${training_transcripts[@]}" > corpus.txt
bash "$script_dir/bin/build_character_lm.sh" "$order" corpus.txt LM.arpa
prepare_lang_test-ds.sh LM.arpa data/train/lang data/train/lang_test "$dummy"

feat_dim="$(feat-to-dim "ark:$matrix" -)"
mapfile -t phones < data/train/lang/phones/nonsilence.int
dummy_id="$(awk -v symbol="$dummy" '$1 == symbol {print $2}' data/train/lang/phones.txt)"
blank_id="$(awk -v symbol="$blank" '$1 == symbol {print $2}' data/train/lang/pdf_blank.txt)"
create_proto_rnn-ds.sh "$feat_dim" 0.5 0.5 HMM-dir "$dummy_id" "$blank_id" "${phones[@]}"
mv HMM-dir/new.mdl HMM-dir/final.mdl
mv HMM-dir/new.tree HMM-dir/tree
mkgraph.sh --transition-scale 1.0 --self-loop-scale 1.0 data/train/lang_test HMM-dir graph

latgen-faster-mapped --acoustic-scale="$asf" --allow-partial=true \
    --max-active=500000000 --beam=100 --word-determinize=false --lattice-beam=25 HMM-dir/final.mdl \
    graph/HCLG.fst "ark:$matrix" 'ark:|gzip -c > lat.gz'
lattice-add-penalty --word-ins-penalty="$wip" 'ark:gunzip -c lat.gz |' ark:- |
    lattice-best-path ark:- ark,t:hyp
int2sym.pl -f 2- data/train/lang/words.txt < hyp > transcripts.txt

cp -- transcripts.txt "$result.txt"
cp -- lat.gz "$result.lat.gz"
cp -- LM.arpa "$result.arpa"
cp -- data/train/lang/words.txt "$result.words.txt"
printf 'Transcripts: %s\nLattice: %s\nLanguage model: %s\nLattice symbols: %s\n' \
    "$result.txt" "$result.lat.gz" "$result.arpa" "$result.words.txt"
