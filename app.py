# Debate Speech Analyzer API with auto-baseline slurred speech scoring
# pip install flask praat-parselmouth pydub librosa numpy

import os
import tempfile
import uuid
import urllib.request
from flask import Flask, request, jsonify
from pydub import AudioSegment
import parselmouth
import mimetypes
import numpy as np
import librosa

app = Flask(__name__)

def download_file1(url):
    """Download an audio file, detect type, convert to MP3 if needed."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        uniq = uuid.uuid4().hex
        tmp_dir = tempfile.gettempdir()

        with urllib.request.urlopen(req, timeout=60) as response:
            content_type = response.getheader("Content-Type")
            ext = mimetypes.guess_extension(content_type.split(";")[0]) if content_type else ".bin"
            output_filename = os.path.join(tmp_dir, f"audio_{uniq}{ext}")

            with open(output_filename, "wb") as file:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    file.write(chunk)

        audio_types = ["audio/webm", "audio/ogg", "audio/m4a", "audio/x-m4a", "audio/wav", "audio/x-wav"]
        if content_type and content_type.split(";")[0] in audio_types and not output_filename.lower().endswith(".mp3"):
            mp3_filename = os.path.splitext(output_filename)[0] + ".mp3"
            sound = AudioSegment.from_file(output_filename)
            sound.export(mp3_filename, format="mp3")
            os.remove(output_filename)
            return mp3_filename

        return output_filename

    except Exception as e:
        print(f"Error downloading file: {e}")
        return None


def download_file(url, suffix):
    """Low-level download using urllib."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                tmp.write(chunk)
        tmp.flush()
        return tmp.name
    except Exception as e:
        tmp.close()
        os.remove(tmp.name)
        raise Exception(f"Failed to download file from {url}: {e}")


def update_baseline(speaker, raw_score):
    """Update or create baseline for speaker, keep last MAX_BASELINE_REFS."""
    if speaker not in speaker_baselines:
        speaker_baselines[speaker] = []
    speaker_baselines[speaker].append(raw_score)
    if len(speaker_baselines[speaker]) > MAX_BASELINE_REFS:
        speaker_baselines[speaker] = speaker_baselines[speaker][-MAX_BASELINE_REFS:]
    return np.mean(speaker_baselines[speaker])


def compute_slurred_score_auto_baseline(mfcc, speaker="default"):
    """Compute adaptive slurred speech score with automatic baseline."""
    mfcc_std = np.std(mfcc, axis=1)
    avg_std = np.mean(mfcc_std)
    mfcc_diff = np.mean(np.abs(np.diff(mfcc, axis=1)))
    raw_score = avg_std + mfcc_diff

    baseline = update_baseline(speaker, raw_score)
    relative = (raw_score - baseline) / baseline
    score = np.clip(relative * 10, 0, 10)
    return round(score, 1)


def analyze_debate_speech(audio_mp3, transcript, speaker="default",
                          short_pause_threshold=0.2, long_pause_threshold=0.3, target_wpm=140):
    audio_wav = audio_mp3.replace(".mp3", ".wav")
    AudioSegment.from_file(audio_mp3, format="mp3").export(audio_wav, format="wav")

    snd = parselmouth.Sound(audio_wav)
    words = transcript.split()
    word_count = len(words)
    total_duration = snd.duration
    speech_rate_wpm = word_count / total_duration * 60 if total_duration > 0 else 0

    intensity = snd.to_intensity()
    times = intensity.xs()
    values = intensity.values[0]
    below_idx = np.where(values < 40.0)[0]
    pause_durations_list = []
    if below_idx.size > 0:
        groups = np.split(below_idx, np.where(np.diff(below_idx) != 1)[0] + 1)
        for g in groups:
            pause_durations_list.append(times[g[-1]] - times[g[0]])

    short_pauses = [d for d in pause_durations_list if d >= short_pause_threshold]
    long_pauses = [d for d in pause_durations_list if d >= long_pause_threshold]

    fillers = ["um", "uh", "like", "you know"]
    filler_count = sum(transcript.lower().count(f) for f in fillers)
    filler_rate_per_100 = (filler_count / word_count) * 100 if word_count > 0 else 0

    pitch = snd.to_pitch()
    mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
    stdev_pitch = parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, "Hertz")
    min_pitch = parselmouth.praat.call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    max_pitch = parselmouth.praat.call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    y, sr = librosa.load(audio_wav, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    slurred_speech_score = compute_slurred_score_auto_baseline(mfcc, speaker=speaker)

    formants = snd.to_formant_burg()
    times_f = np.linspace(0, snd.duration, num=10)
    f1_vals, f2_vals = [], []
    for t in times_f:
        f1 = formants.get_value_at_time(1, t)
        f2 = formants.get_value_at_time(2, t)
        if f1 > 0 and f2 > 0:
            f1_vals.append(f1)
            f2_vals.append(f2)
    f1_mean = np.mean(f1_vals) if f1_vals else 0
    f2_mean = np.mean(f2_vals) if f2_vals else 0

    metrics = {
        "word_count": word_count,
        "speech_duration_s": round(total_duration, 2),
        "speech_rate_wpm": round(speech_rate_wpm, 1),
        "target_wpm": target_wpm,
        "short_pauses_count": len(short_pauses),
        "short_pauses_durations_s": [round(d, 2) for d in short_pauses],
        "long_pauses_count": len(long_pauses),
        "long_pauses_durations_s": [round(d, 2) for d in long_pauses],
        "filler_count": filler_count,
        "filler_rate_per_100_words": round(filler_rate_per_100, 1),
        "pitch_mean_Hz": round(mean_pitch, 1),
        "pitch_std_Hz": round(stdev_pitch, 1),
        "pitch_min_Hz": round(min_pitch, 1),
        "pitch_max_Hz": round(max_pitch, 1),
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_std": mfcc_std.tolist(),
        "slurred_speech_score": slurred_speech_score,
        "formant_F1_mean_Hz": round(f1_mean, 1),
        "formant_F2_mean_Hz": round(f2_mean, 1)
    }

    os.remove(audio_wav)
    return metrics


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    audio_url = data.get("audio_url")
    transcript_url = data.get("transcript_url")
    speaker = data.get("speaker", "default")  # optional speaker ID

    if not audio_url or not transcript_url:
        return jsonify({"error": "Missing audio_url or transcript_url"}), 400

    try:
        target_audio = download_file1(audio_url)
        if not target_audio:
            return jsonify({"error": "Failed to download or convert audio"}), 500

        tmp_transcript = download_file(transcript_url, ".txt")
        with open(tmp_transcript, "r") as f:
            transcript = f.read().strip()

        metrics = analyze_debate_speech(target_audio, transcript, speaker=speaker)

        os.remove(target_audio)
        os.remove(tmp_transcript)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(metrics)

if __name__ == "__main__":
    app.run()