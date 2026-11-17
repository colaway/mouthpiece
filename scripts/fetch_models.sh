#!/usr/bin/env bash
# Download the pretrained assets into models/ (git-ignored).
#   - Auto-AVSR LRS3 visual-only VSR model, WER 19.1  (~937 MB zip, 1.0 GB unpacked)
#   - LRS3 subword language model                     (~200 MB zip)
#   - Auto-AVSR 68-point mean face used for alignment (1 KB)
#   - MediaPipe Face Landmarker model                 (3.7 MB)
#   - LRS3 SentencePiece unigram5000 tokenizer + units (0.4 MB). The units file
#     (ESPnet token list) ships with the VSR repo's code; the .model with auto_avsr.
# Sources are the authors' official links. Every file is checked against a pinned
# SHA-256 before it is used, so a changed or tampered download stops the script.
# Licenses: model weights are for non-commercial use
# (see mpc001/Visual_Speech_Recognition_for_Multiple_Languages).
set -euo pipefail
cd "$(dirname "$0")/.."

DL=models/_download
GDOWN="gdown@6.5.1"   # pinned so the downloader itself doesn't change under us
mkdir -p "$DL" models/vsr models/lm models/preprocess

sha_ok() {  # file sha256
  [[ -f "$1" ]] && [[ "$(shasum -a 256 "$1" | cut -d' ' -f1)" == "$2" ]]
}

verify() {  # file sha256
  if ! sha_ok "$1" "$2"; then
    echo "Checksum mismatch for $1, deleting it. Re-run to try again." >&2
    rm -f "$1"
    exit 1
  fi
}

fetch_gdrive() {  # id out sha256
  if sha_ok "$2" "$3"; then echo "ok  $2"; return; fi
  uvx "$GDOWN" "$1" -O "$2"
  verify "$2" "$3"
}

fetch_url() {  # url out sha256
  if sha_ok "$2" "$3"; then echo "ok  $2"; return; fi
  curl -fsSL -o "$2" "$1"
  verify "$2" "$3"
}

fetch_gdrive 1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6 "$DL/vsr.zip" \
  bbf052e7b7aa390d491c5eeac9015cfb70884216bd4bf3982f06af2487988e58
fetch_gdrive 1g31HGxJnnOwYl17b70ObFQZ1TSnPvRQv "$DL/lm.zip" \
  40154e6f2fa068380ac82a3eb3c6d5039ec83706773f89482d7bca39cc0b1060
unzip -joq "$DL/vsr.zip" -d models/vsr
unzip -joq "$DL/lm.zip"  -d models/lm

fetch_url https://raw.githubusercontent.com/mpc001/auto_avsr/main/preparation/detectors/retinaface/20words_mean_face.npy \
  models/preprocess/20words_mean_face.npy \
  dbf68b2044171e1160716df7c53e8bbfaa0ee8c61fb41171d04cb6092bb81422
fetch_url https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  models/preprocess/face_landmarker.task \
  64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
fetch_url https://raw.githubusercontent.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/master/pipelines/tokens/unigram5000_units.txt \
  models/vsr/unigram5000_units.txt \
  29001d967ad290e28f6f1d7d47e07e314354bf67d3b1f494e936fa2fed9a9122
fetch_url https://raw.githubusercontent.com/mpc001/auto_avsr/main/spm/unigram/unigram5000.model \
  models/vsr/unigram5000.model \
  2c1d648ccf5fdce6612ecbea2ffbd1cab5aabc90458781186175c3911c4bdb1e

echo "Done. You can delete $DL to free ~1.1 GB."
