#!/usr/bin/env bash
set -euo pipefail

ORDER="$1"
CORPUS="$2"
ARPA="$3"

if [[ "$ORDER" -eq 0 ]]; then
    # Preserve the uniform vocabulary-only baseline, including the end token.
    awk '
        {
            for (i = 1; i <= NF; i++)
                if ($i != "<s>" && $i != "</s>" && $i != "<unk>")
                    vocab[$i] = 1
        }
        END {
            probability = -log(length(vocab) + 1) / log(10)
            printf "\n\\data\\\nngram 1=%d\n\n\\1-grams:\n", length(vocab) + 2
            printf "%.7g\t</s>\n-99\t<s>\n", probability
            for (token in vocab)
                printf "%.7g\t%s\n", probability, token
            print "\n\\end\\"
        }
    ' "$CORPUS" > "$ARPA"
else
    # Small training samples may lack the counts needed to estimate discounts.
    lmplz -o "$ORDER" -S "${KENLM_MEMORY:-512M}" --discount_fallback \
        < "$CORPUS" > "$ARPA"
fi
