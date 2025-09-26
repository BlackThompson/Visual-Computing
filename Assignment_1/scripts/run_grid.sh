#!/usr/bin/env bash
set -euo pipefail

BIN="/mnt/d/Code/Course/Visual Computing/Assignment_1/build/panorama"
IMG1="$1"; IMG2="$2"
SET_ID="${3:-setA}"; PAIR_ID="${4:-img01_to_img02}"

DETS=(sift orb akaze)
RATIOS=(0.7 0.8)
THS=(1 3 5 10)
BLENDS=(overlay feather)

for det in "${DETS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    for th in "${THS[@]}"; do
      for blend in "${BLENDS[@]}"; do
        echo "Running det=$det ratio=$ratio th=$th blend=$blend"
        "$BIN" --set "$SET_ID" --pair "$PAIR_ID" --det "$det" --blend "$blend" \
          --ratio "$ratio" --ransac 2000 --th "$th" --debug "$IMG1" "$IMG2"
      done
    done
  done
done


