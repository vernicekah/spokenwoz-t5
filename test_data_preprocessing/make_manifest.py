import json
import soundfile as sf
import torchaudio
from pathlib import Path
from tqdm import tqdm

# --- Config ---
ROOT_DIR = Path("data")  # relative symlink
AUDIO_DIR = ROOT_DIR / "processed_audio" / "audio_5700_test_16k"
TEXT_JSON = ROOT_DIR / "raw_data" / "text_5700_test" / "data.json"
OUTPUT_MANIFEST = ROOT_DIR / "processed_data" / "root_test_manifest_NeMo.json"
SEGMENTS_DIR = ROOT_DIR / "processed_audio" / "audio_5700_test_16k_segments"


def main():
    """Combines 16kHz audio and text into JSON NeMo manifest format
    Segments audio based on word-level timestamps"""

    # print(f"Loading text annotations from {TEXT_JSON}")
    with open(TEXT_JSON, "r", encoding="utf-8") as f:
        text_data = json.load(f)

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True) # directory for audio segments

    manifest = []

    # print(f"Building manifest from {AUDIO_DIR}")
    # iterate over all utterance IDs from the JSON
    for utt_id, record in tqdm(text_data.items()):
        audio_path = AUDIO_DIR / f"{utt_id}.wav"
        if not audio_path.exists():
            continue

        waveform, sr = torchaudio.load(audio_path)

        # iterate over dialogue turns
        for i, turn in enumerate(record.get("log", [])):
            if "words" not in turn or len(turn["words"]) == 0:
                continue

            # start and end time from the first and last word
            start = turn["words"][0]["BeginTime"] / 1000.0
            end = turn["words"][-1]["EndTime"] / 1000.0

            # convert times to audio sample indices
            start_frame = int(start * sr)
            end_frame = int(end * sr)

            chunk = waveform[:, start_frame:end_frame] # extract slice from waveform
            duration = (end_frame - start_frame) / sr # each segment duration

            if chunk.shape[1] == 0: # skip empty segments
                continue

            # If stereo → downmix to mono
            if chunk.shape[0] > 1:
                chunk = chunk.mean(dim=0, keepdim=True)

            # save chunk as a new wav -> save using soudfile
            out_name = f"{utt_id}_turn{i+1}.wav" # each segment file name 
            out_path = SEGMENTS_DIR / out_name
            # if not out_path.parent.exists():   
            sf.write(out_path, chunk.squeeze().numpy().astype("float32"), sr)

            # Extract main dialog_act key (if dict)
            raw_act = turn.get("dialog_act") or turn.get("act")
            if isinstance(raw_act, dict) and len(raw_act) > 0:
                dialog_act = list(raw_act.keys())[0]
            else:
                dialog_act = "unknown"
            # manifest entry
            entry = {
                "audio_filepath": str(out_path.resolve()),
                "duration": duration,
                "text": turn["text"].strip(),
                "dialog_act": dialog_act
            }
            manifest.append(entry)

    # filter out unknown intents before saving
    manifest = [entry for entry in manifest if entry["dialog_act"] != "unknown"]
    
    # print(f"Writing manifest to {OUTPUT_MANIFEST}")
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"{len(manifest)} entries written to {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()


# Total: - samples in the manifest
# remove unknown intent sample
# remaining - samples