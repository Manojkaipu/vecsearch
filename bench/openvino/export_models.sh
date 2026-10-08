#!/usr/bin/env bash
# Export the two models to OpenVINO IR at FP16, INT8 and INT4 weight precision with optimum-intel.
#
#   bench/openvino/export_models.sh models            # both models, all precisions
#   bench/openvino/export_models.sh models minilm     # just the embedding model
#   bench/openvino/export_models.sh models qwen       # just the LLM
#
# Needs: pip install "optimum-intel[openvino,nncf]" torch sentence-transformers. The LLM export
# holds the model in memory twice, so give it about 8 GB. Each export prints its wall time; the
# sizes are written to <out>/sizes.txt.
set -euo pipefail

OUT=${1:-models}
WHICH=${2:-all}
MINILM=sentence-transformers/all-MiniLM-L6-v2
QWEN=Qwen/Qwen2.5-1.5B-Instruct
mkdir -p "$OUT"

for w in fp16 int8 int4; do
  if [[ $WHICH == all || $WHICH == minilm ]] && [[ ! -f $OUT/minilm-$w/openvino_model.xml ]]; then
    echo "== MiniLM $w"
    time optimum-cli export openvino --model "$MINILM" --task feature-extraction --weight-format "$w" "$OUT/minilm-$w"
  fi
  if [[ $WHICH == all || $WHICH == qwen ]] && [[ ! -f $OUT/qwen2.5-1.5b-$w/openvino_model.xml ]]; then
    echo "== Qwen2.5-1.5B-Instruct $w"
    time optimum-cli export openvino --model "$QWEN" --task text-generation-with-past --weight-format "$w" "$OUT/qwen2.5-1.5b-$w"
  fi
done

du -sb "$OUT"/*/openvino_model.bin 2>/dev/null | sed 's#/openvino_model.bin##' | tee "$OUT/sizes.txt"
