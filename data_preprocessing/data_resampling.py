import os
from pathlib import Path
import torchaudio

def resample_audio(input_path, output_path, target_sr=16000):
    """
    Resample a single audio file if needed.
    - input_path: path to original audio file
    - output_path: where to save resampled audio
    - target_sr: target sampling rate (default = 16kHz)
    """
    waveform, sample_rate = torchaudio.load(input_path)

    # Only resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        print(f"Resampled {os.path.basename(input_path)} from {sample_rate} Hz → {target_sr} Hz")
    else:
        print(f"{os.path.basename(input_path)} already {target_sr} Hz")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, waveform, target_sr)


def process_audio_folder(input_dir, output_dir, target_sr=16000):
    """
    Go through a directory of audio files and resample if needed.
    Keeps directory structure intact.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for f in files:
            if not f.endswith(".wav"):
                continue

            input_path = Path(root) / f
            rel_path = input_path.relative_to(input_dir)
            output_path = output_dir / rel_path

            resample_audio(input_path, output_path, target_sr=target_sr)


def main():
    # Example paths — change to your actual directories
    input_dir = "data/raw_data/audio_5700_train_dev"
    output_dir = "data/processed_audio/audio_5700_train_dev_16k"
    target_sr = 16000

    process_audio_folder(input_dir, output_dir, target_sr)
    print("\n All files processed and saved to:", output_dir)


if __name__ == "__main__":
    main()
