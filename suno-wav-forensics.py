#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, chirp
import pyloudnorm as pyln

# =========================
# CONFIG
# =========================
MP3_PATH = os.path.expanduser("~/Downloads/suno-test.mp3")
WAV_PATH = os.path.expanduser("~/Downloads/suno-test.wav")
WORKDIR = "suno_forensics_output"
os.makedirs(WORKDIR, exist_ok=True)

N_FFT = 4096
HOP_LENGTH = 1024

# =========================
# UTILS
# =========================
def ensure_exists(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found at: {path}")

def save_waveform(y, sr, outpath, title="", zoom=None):
    """
    zoom: (start_sec, end_sec) or None
    """
    plt.figure(figsize=(12, 4))
    if zoom is not None:
        start = int(zoom[0] * sr)
        end = int(zoom[1] * sr)
        y_plot = y[start:end]
        times = np.linspace(zoom[0], zoom[1], len(y_plot))
        plt.plot(times, y_plot)
        plt.xlim(zoom[0], zoom[1])
    else:
        times = np.linspace(0, len(y) / sr, num=len(y))
        plt.plot(times, y)
        plt.xlim(0, len(y) / sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_spectrogram(y, sr, outpath, title="", cmap="magma", log_freq=True):
    plt.figure(figsize=(12, 4))
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    if log_freq:
        librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="log",
            cmap=cmap,
        )
    else:
        librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="linear",
            cmap=cmap,
        )
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def lowpass_filter(y, sr, cutoff=16000.0, order=10):
    sos = butter(order, cutoff / (sr / 2.0), btype="lowpass", output="sos")
    return sosfilt(sos, y)

def highband_mask(sr, n_fft, low=16000.0, high=None):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    if high is None:
        high = sr / 2.0
    return (freqs >= low) & (freqs <= high)

def compute_rms(y):
    return float(np.sqrt(np.mean(y**2) + 1e-15))

def compute_peak(y):
    return float(np.max(np.abs(y)) + 1e-15)

def compute_crest_factor(y):
    rms = compute_rms(y)
    peak = compute_peak(y)
    return 20.0 * np.log10(peak / rms)

def compute_lufs(y, sr):
    meter = pyln.Meter(sr)  # EBU R128
    return float(meter.integrated_loudness(y))

def align_signals(a, b):
    """
    Naive alignment: trim both to min length.
    """
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]

def compute_band_energies(y, sr, bands):
    """
    bands: list of (low_freq, high_freq, label)
    returns dict label -> energy
    """
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    energies = {}
    for low, high, label in bands:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            energies[label] = 0.0
        else:
            band_energy = float(np.mean(S[mask, :]))
            energies[label] = band_energy
    return energies

def phase_correlation(a, b):
    """
    Simple normalized correlation between two aligned signals.
    """
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-15
    return float(np.sum(a * b) / denom)

def save_histogram(y, outpath, title="", bins=100, range_=(-1, 1)):
    plt.figure(figsize=(8, 4))
    plt.hist(y, bins=bins, range=range_, density=True)
    plt.title(title)
    plt.xlabel("Amplitude")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def run_ffmpeg_decode_to_wav(src_path, dst_path, target_sr=None, stereo=True):
    cmd = ["ffmpeg", "-y", "-i", src_path]
    if target_sr is not None:
        cmd += ["-ar", str(target_sr)]
    if stereo:
        cmd += ["-ac", "2"]
    cmd += [dst_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# =========================
# MAIN LOGIC
# =========================
def main():
    ensure_exists(MP3_PATH, "MP3 file")
    ensure_exists(WAV_PATH, "WAV file")

    print("Loading audio...")

    # Use librosa to load, then unify SR
    mp3_y, mp3_sr = librosa.load(MP3_PATH, sr=None, mono=True)
    wav_y, wav_sr = librosa.load(WAV_PATH, sr=None, mono=True)

    print(f"MP3 SR: {mp3_sr}, WAV SR: {wav_sr}")

    # Resample so both share the same SR
    target_sr = max(mp3_sr, wav_sr)
    if mp3_sr != target_sr:
        mp3_y = librosa.resample(mp3_y, orig_sr=mp3_sr, target_sr=target_sr)
        mp3_sr = target_sr
    if wav_sr != target_sr:
        wav_y = librosa.resample(wav_y, orig_sr=wav_sr, target_sr=target_sr)
        wav_sr = target_sr

    sr = target_sr
    print(f"Common SR: {sr}")

    # -------------------------
    # BASELINE: MP3 → WAV roundtrip
    # -------------------------
    print("Creating baseline MP3-derived WAV...")
    baseline_mp3_wav_path = os.path.join(WORKDIR, "baseline_mp3_derived.wav")
    run_ffmpeg_decode_to_wav(MP3_PATH, baseline_mp3_wav_path, target_sr=sr, stereo=False)
    baseline_y, baseline_sr = librosa.load(baseline_mp3_wav_path, sr=None, mono=True)

    # Align everything
    mp3_y, wav_y = align_signals(mp3_y, wav_y)
    mp3_y, baseline_y = align_signals(mp3_y, baseline_y)
    # Align all to same min length
    min_len = min(len(mp3_y), len(wav_y), len(baseline_y))
    mp3_y = mp3_y[:min_len]
    wav_y = wav_y[:min_len]
    baseline_y = baseline_y[:min_len]

    duration_sec = min_len / sr
    print(f"Aligned duration: {duration_sec:.2f} s")

    # -------------------------
    # BASIC METRICS
    # -------------------------
    def basic_metrics(label, y):
        return {
            "label": label,
            "rms": compute_rms(y),
            "peak": compute_peak(y),
            "crest_db": compute_crest_factor(y),
            "lufs": compute_lufs(y, sr),
        }

    print("Computing loudness and dynamics...")
    metrics_mp3 = basic_metrics("MP3", mp3_y)
    metrics_wav = basic_metrics("WAV", wav_y)
    metrics_baseline = basic_metrics("Baseline_MP3_Derived", baseline_y)

    # -------------------------
    # NULL TESTS
    # -------------------------
    print("Running null tests...")

    residual_unknown = wav_y - mp3_y
    residual_baseline = baseline_y - mp3_y

    residual_unknown_path = os.path.join(WORKDIR, "residual_unknown_full.wav")
    residual_baseline_path = os.path.join(WORKDIR, "residual_baseline_full.wav")
    sf.write(residual_unknown_path, residual_unknown, sr)
    sf.write(residual_baseline_path, residual_baseline, sr)

    # Band-limited (below 16kHz)
    mp3_lp = lowpass_filter(mp3_y, sr, cutoff=16000.0)
    wav_lp = lowpass_filter(wav_y, sr, cutoff=16000.0)
    baseline_lp = lowpass_filter(baseline_y, sr, cutoff=16000.0)

    residual_unknown_lp = wav_lp - mp3_lp
    residual_baseline_lp = baseline_lp - mp3_lp

    residual_unknown_lp_path = os.path.join(WORKDIR, "residual_unknown_lowpassed.wav")
    residual_baseline_lp_path = os.path.join(WORKDIR, "residual_baseline_lowpassed.wav")
    sf.write(residual_unknown_lp_path, residual_unknown_lp, sr)
    sf.write(residual_baseline_lp_path, residual_baseline_lp, sr)

    # Residual stats
    rms_mp3 = compute_rms(mp3_y)
    rms_res_unknown = compute_rms(residual_unknown)
    rms_res_baseline = compute_rms(residual_baseline)
    rms_res_unknown_lp = compute_rms(residual_unknown_lp)
    rms_res_baseline_lp = compute_rms(residual_baseline_lp)

    residual_ratio_unknown = rms_res_unknown / (rms_mp3 + 1e-15)
    residual_ratio_baseline = rms_res_baseline / (rms_mp3 + 1e-15)
    residual_ratio_unknown_lp = rms_res_unknown_lp / (rms_mp3 + 1e-15)
    residual_ratio_baseline_lp = rms_res_baseline_lp / (rms_mp3 + 1e-15)

    # -------------------------
    # PHASE CORRELATION
    # -------------------------
    print("Computing phase correlations...")
    corr_wav_mp3 = phase_correlation(wav_y, mp3_y)
    corr_baseline_mp3 = phase_correlation(baseline_y, mp3_y)

    # -------------------------
    # BAND ENERGIES
    # -------------------------
    print("Computing band energies...")
    bands = [
        (0, 5000, "0–5k"),
        (5000, 10000, "5–10k"),
        (10000, 16000, "10–16k"),
        (16000, sr / 2.0, "16k–Nyq"),
    ]

    band_mp3 = compute_band_energies(mp3_y, sr, bands)
    band_wav = compute_band_energies(wav_y, sr, bands)
    band_baseline = compute_band_energies(baseline_y, sr, bands)

    # Relative high-band energy
    def highband_ratio(bands_dict):
        total = sum(bands_dict.values()) + 1e-15
        return bands_dict["16k–Nyq"] / total

    highband_mp3 = highband_ratio(band_mp3)
    highband_wav = highband_ratio(band_wav)
    highband_baseline = highband_ratio(band_baseline)

    # -------------------------
    # HISTOGRAMS
    # -------------------------
    print("Rendering histograms...")
    save_histogram(mp3_y, os.path.join(WORKDIR, "hist_mp3.png"), "Amplitude Histogram - MP3")
    save_histogram(wav_y, os.path.join(WORKDIR, "hist_wav.png"), "Amplitude Histogram - WAV")
    save_histogram(
        residual_unknown,
        os.path.join(WORKDIR, "hist_residual_unknown.png"),
        "Amplitude Histogram - Residual (WAV - MP3)",
        bins=200,
        range_=(-0.5, 0.5),
    )
    save_histogram(
        residual_baseline,
        os.path.join(WORKDIR, "hist_residual_baseline.png"),
        "Amplitude Histogram - Residual (Baseline - MP3)",
        bins=200,
        range_=(-0.5, 0.5),
    )

    # -------------------------
    # WAVEFORMS
    # -------------------------
    print("Rendering waveforms...")
    save_waveform(mp3_y, sr, os.path.join(WORKDIR, "wave_mp3_full.png"), "Waveform - MP3 (Full)")
    save_waveform(wav_y, sr, os.path.join(WORKDIR, "wave_wav_full.png"), "Waveform - WAV (Full)")
    save_waveform(
        baseline_y,
        sr,
        os.path.join(WORKDIR, "wave_baseline_full.png"),
        "Waveform - Baseline MP3-derived (Full)",
    )
    save_waveform(
        residual_unknown,
        sr,
        os.path.join(WORKDIR, "wave_residual_unknown_full.png"),
        "Residual (WAV - MP3) Waveform (Full)",
    )
    save_waveform(
        residual_baseline,
        sr,
        os.path.join(WORKDIR, "wave_residual_baseline_full.png"),
        "Residual (Baseline - MP3) Waveform (Full)",
    )

    # Zoom on first 5 seconds for detail
    for (sig, name) in [
        (mp3_y, "mp3"),
        (wav_y, "wav"),
        (baseline_y, "baseline"),
        (residual_unknown, "residual_unknown"),
        (residual_baseline, "residual_baseline"),
    ]:
        save_waveform(
            sig,
            sr,
            os.path.join(WORKDIR, f"wave_{name}_zoom.png"),
            f"Waveform - {name} (0–5s)",
            zoom=(0, min(5, duration_sec)),
        )

    # -------------------------
    # SPECTROGRAMS (multi-style)
    # -------------------------
    print("Rendering spectrograms...")
    cmaps = ["magma", "viridis", "plasma"]

    def render_all_specs(prefix, y, label):
        for cmap in cmaps:
            save_spectrogram(
                y,
                sr,
                os.path.join(WORKDIR, f"spec_{prefix}_{cmap}_log.png"),
                f"{label} Spectrogram (log, {cmap})",
                cmap=cmap,
                log_freq=True,
            )
            save_spectrogram(
                y,
                sr,
                os.path.join(WORKDIR, f"spec_{prefix}_{cmap}_linear.png"),
                f"{label} Spectrogram (linear, {cmap})",
                cmap=cmap,
                log_freq=False,
            )

    render_all_specs("mp3", mp3_y, "MP3")
    render_all_specs("wav", wav_y, "WAV")
    render_all_specs("baseline", baseline_y, "Baseline MP3-derived")
    render_all_specs("residual_unknown", residual_unknown, "Residual (WAV - MP3)")
    render_all_specs("residual_baseline", residual_baseline, "Residual (Baseline - MP3)")

    # -------------------------
    # SYNTHETIC CHIRP REFERENCE
    # -------------------------
    print("Generating synthetic CHIRP reference...")
    chirp_duration = 3.0
    t = np.linspace(0, chirp_duration, int(sr * chirp_duration), endpoint=False)
    chirp_sig = chirp(
        t,
        f0=20.0,
        f1=sr / 2.1,
        t1=chirp_duration,
        method="logarithmic",
    )
    chirp_sig = chirp_sig * 0.9  # headroom

    chirp_wav_path = os.path.join(WORKDIR, "chirp_wideband.wav")
    sf.write(chirp_wav_path, chirp_sig, sr)

    chirp_mp3_path = os.path.join(WORKDIR, "chirp_wideband.mp3")
    # encode chirp to MP3 using ffmpeg, then decode back
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            chirp_wav_path,
            "-b:a",
            "192k",
            chirp_mp3_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    chirp_mp3_decoded_path = os.path.join(WORKDIR, "chirp_wideband_mp3_decoded.wav")
    run_ffmpeg_decode_to_wav(chirp_mp3_path, chirp_mp3_decoded_path, target_sr=sr, stereo=False)

    chirp_decoded, _ = librosa.load(chirp_mp3_decoded_path, sr=None, mono=True)
    chirp_sig, chirp_decoded = align_signals(chirp_sig, chirp_decoded)

    # Spectrograms of chirp original vs mp3-decoded
    save_spectrogram(
        chirp_sig,
        sr,
        os.path.join(WORKDIR, "spec_chirp_original_log.png"),
        "Synthetic CHIRP (Original, log)",
        cmap="magma",
        log_freq=True,
    )
    save_spectrogram(
        chirp_decoded,
        sr,
        os.path.join(WORKDIR, "spec_chirp_decoded_log.png"),
        "Synthetic CHIRP (MP3-decoded, log)",
        cmap="magma",
        log_freq=True,
    )

    # -------------------------
    # ENHANCED HEURISTIC LIKELIHOOD SCORE
    # -------------------------
    print("Computing enhanced likelihood score...")

    # ---- Basic similarity component (loudness / dynamics) ----
    lufs_delta_wav = abs(metrics_wav["lufs"] - metrics_mp3["lufs"])
    crest_delta_wav = abs(metrics_wav["crest_db"] - metrics_mp3["crest_db"])
    rms_ratio_wav = metrics_wav["rms"] / (metrics_mp3["rms"] + 1e-15)

    def similarity_from_delta(delta, max_delta):
        # 0 delta => 1.0 score, >= max_delta => 0
        return max(0.0, min(1.0, 1.0 - (delta / max_delta)))

    sim_lufs = similarity_from_delta(lufs_delta_wav, 3.0)           # 0–3 dB window
    sim_crest = similarity_from_delta(crest_delta_wav, 6.0)         # 0–6 dB window
    sim_rms = similarity_from_delta(abs(1.0 - rms_ratio_wav), 0.5)  # ±50% RMS

    basic_similarity = float(0.4 * sim_lufs + 0.3 * sim_crest + 0.3 * sim_rms)

    # ---- Residual-based components ----
    def residual_likelihood_component(unknown, baseline):
        # If unknown <= baseline → strongest evidence (perfect repack)
        if unknown <= baseline:
            return 1.0
        # Otherwise decay with ratio
        return max(0.0, min(1.0, baseline / (unknown + 1e-15)))

    # ---- Correlation component ----
    def corr_likelihood_component(corr_unknown, corr_baseline):
        diff = abs(corr_unknown - corr_baseline)
        # diff 0 -> 1, diff >= 0.2 -> ~0
        return max(0.0, min(1.0, 1.0 - diff / 0.2))

    # ---- High-band pattern component ----
    def highband_likelihood_component(hb_mp3, hb_wav, hb_baseline):
        if hb_baseline < 1e-12:
            hb_baseline = 1e-12
        factor_wav = hb_wav / hb_baseline
        # If WAV highband is close to baseline → similar lossy footprint
        if factor_wav <= 1.5:
            return 1.0
        elif factor_wav >= 6.0:
            return 0.0
        else:
            return max(0.0, min(1.0, 1.0 - (factor_wav - 1.5) / (6.0 - 1.5)))

    comp_residual_full = residual_likelihood_component(residual_ratio_unknown, residual_ratio_baseline)
    comp_residual_lp = residual_likelihood_component(residual_ratio_unknown_lp, residual_ratio_baseline_lp)
    comp_corr = corr_likelihood_component(corr_wav_mp3, corr_baseline_mp3)
    comp_highband = highband_likelihood_component(highband_mp3, highband_wav, highband_baseline)

    # Pack components (0–1)
    likelihood_components = {
        "residual_full": comp_residual_full,
        "residual_lowpassed": comp_residual_lp,
        "correlation": comp_corr,
        "highband_pattern": comp_highband,
        "basic_similarity": basic_similarity,
    }

    # Component confidence labels (for HTML)
    def component_label(name, score):
        if score >= 0.85:
            level = "HIGH"
        elif score >= 0.65:
            level = "MEDIUM"
        elif score >= 0.45:
            level = "LOW"
        else:
            level = "VERY LOW"

        if score >= 0.65:
            trend = "Supports MP3-derived hypothesis"
        elif score <= 0.35:
            trend = "Points away from pure MP3-derived"
        else:
            trend = "Inconclusive on its own"

        return f"{level} – {trend}"

    # Weighting: residuals heavy, correlation + similarity medium, highband moderate
    likelihood_score_raw = (
        0.30 * comp_residual_full
        + 0.20 * comp_residual_lp
        + 0.20 * comp_corr
        + 0.15 * comp_highband
        + 0.15 * basic_similarity
    ) * 100.0

    # Detection-biased calibration: nudge toward MP3-derived in ambiguous regimes
    likelihood_score = likelihood_score_raw * 0.85 + 10.0
    likelihood_score = float(max(0.0, min(100.0, likelihood_score)))

    # Textual classification
    if likelihood_score > 90:
        verdict_text = "Extremely likely WAV is MP3-derived (or very tightly coupled to the MP3)."
    elif likelihood_score > 75:
        verdict_text = "Highly likely WAV is MP3-derived."
    elif likelihood_score > 55:
        verdict_text = "Moderately likely WAV is MP3-derived."
    elif likelihood_score > 35:
        verdict_text = "Inconclusive / mixed evidence."
    elif likelihood_score > 15:
        verdict_text = "Unlikely WAV is purely MP3-derived."
    else:
        verdict_text = "Very unlikely WAV is MP3-derived using this MP3 as source."

    # -------------------------
    # HTML REPORT
    # -------------------------
    print("Writing HTML report...")
    html_path = os.path.join(WORKDIR, "report.html")

    def metrics_table_row(m):
        return f"""
<tr>
  <td>{m['label']}</td>
  <td>{m['rms']:.6f}</td>
  <td>{m['peak']:.6f}</td>
  <td>{m['crest_db']:.2f} dB</td>
  <td>{m['lufs']:.2f} LUFS</td>
</tr>
"""

    def band_table_rows(label, bands_dict):
        rows = ""
        for (low, high, band_label) in bands:
            val = bands_dict[band_label]
            rows += f"<tr><td>{label}</td><td>{band_label}</td><td>{val:.6e}</td></tr>\n"
        return rows

    with open(html_path, "w") as f:
        f.write(f"""
<html>
<head>
  <meta charset="UTF-8">
  <title>Suno WAV Forensics Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 30px;
      background-color: #f5f5f5;
      color: #222;
    }}
    h1, h2, h3 {{
      margin-top: 1.5em;
    }}
    .score {{
      font-size: 2em;
      font-weight: bold;
      padding: 10px 0;
    }}
    table {{
      border-collapse: collapse;
      margin: 15px 0;
      width: 100%;
      background: white;
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 6px 8px;
      text-align: left;
      font-size: 0.9em;
    }}
    th {{
      background: #eee;
    }}
    img {{
      max-width: 100%;
      margin: 10px 0 30px 0;
      border: 1px solid #ddd;
      background: white;
    }}
    .section {{
      background: #ffffff;
      padding: 20px;
      margin-bottom: 20px;
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    code {{
      background: #eee;
      padding: 2px 4px;
      border-radius: 3px;
    }}
  </style>
</head>
<body>

<h1>Suno WAV Forensic Analysis</h1>

<div class="section">
  <h2>Headline Verdict</h2>
  <p class="score">Estimated likelihood WAV is MP3-derived: <b>{likelihood_score:.2f}%</b></p>
  <p><b>Interpretation:</b> {verdict_text}</p>

  <h3>Component Scores (0–1 scale)</h3>
  <table>
    <tr><th>Component</th><th>Score</th><th>Meaning</th><th>Confidence / Interpretation</th></tr>
    <tr>
      <td>Residual (Full-band)</td>
      <td>{likelihood_components['residual_full']:.3f}</td>
      <td>How closely WAV cancels against MP3 across the full spectrum compared to a known MP3-derived baseline.</td>
      <td>{component_label('residual_full', likelihood_components['residual_full'])}</td>
    </tr>
    <tr>
      <td>Residual (&lt;16 kHz)</td>
      <td>{likelihood_components['residual_lowpassed']:.3f}</td>
      <td>How closely WAV cancels against MP3 when focusing on frequencies below ~16 kHz.</td>
      <td>{component_label('residual_lowpassed', likelihood_components['residual_lowpassed'])}</td>
    </tr>
    <tr>
      <td>Phase Correlation</td>
      <td>{likelihood_components['correlation']:.3f}</td>
      <td>How similar the temporal / phase structure of WAV vs MP3 is, relative to baseline.</td>
      <td>{component_label('correlation', likelihood_components['correlation'])}</td>
    </tr>
    <tr>
      <td>High-band Pattern</td>
      <td>{likelihood_components['highband_pattern']:.3f}</td>
      <td>Whether the WAV high-frequency band behaves more like “MP3-derived” or “new independent content”.</td>
      <td>{component_label('highband_pattern', likelihood_components['highband_pattern'])}</td>
    </tr>
    <tr>
      <td>Basic Similarity (Loudness / Dynamics)</td>
      <td>{likelihood_components['basic_similarity']:.3f}</td>
      <td>How closely WAV matches MP3 in overall level, loudness, and crest factor.</td>
      <td>{component_label('basic_similarity', likelihood_components['basic_similarity'])}</td>
    </tr>
  </table>
</div>

<div class="section">
  <h2>Basic Signal Metrics</h2>
  <p>Duration analyzed: <b>{duration_sec:.2f} s</b> at <b>{sr} Hz</b></p>
  <table>
    <tr>
      <th>Signal</th><th>RMS</th><th>Peak</th><th>Crest Factor</th><th>Integrated LUFS</th>
    </tr>
    {metrics_table_row(metrics_mp3)}
    {metrics_table_row(metrics_wav)}
    {metrics_table_row(metrics_baseline)}
  </table>
</div>

<div class="section">
  <h2>Residual Analysis (Null Tests)</h2>
  <h3>Residual RMS Ratios (lower = closer to MP3)</h3>
  <table>
    <tr><th>Type</th><th>Residual RMS / MP3 RMS</th></tr>
    <tr><td>Unknown WAV vs MP3 (full-band)</td><td>{residual_ratio_unknown:.6f}</td></tr>
    <tr><td>Baseline MP3-derived vs MP3 (full-band)</td><td>{residual_ratio_baseline:.6f}</td></tr>
    <tr><td>Unknown WAV vs MP3 (&lt;16 kHz)</td><td>{residual_ratio_unknown_lp:.6f}</td></tr>
    <tr><td>Baseline MP3-derived vs MP3 (&lt;16 kHz)</td><td>{residual_ratio_baseline_lp:.6f}</td></tr>
  </table>

  <p>In a true “WAV is just a repackaged MP3” scenario, the unknown WAV residual ratios should be similar to the baseline
  MP3-derived residual ratios. The closer they are, the more the WAV behaves like a decoded MP3.</p>

  <h3>Residual Waveforms</h3>
  <img src="wave_residual_unknown_full.png" alt="Residual Unknown Full">
  <img src="wave_residual_baseline_full.png" alt="Residual Baseline Full">
  <img src="wave_residual_unknown_zoom.png" alt="Residual Unknown Zoom">
  <img src="wave_residual_baseline_zoom.png" alt="Residual Baseline Zoom">

  <h3>Residual Histograms</h3>
  <img src="hist_residual_unknown.png" alt="Residual Unknown Histogram">
  <img src="hist_residual_baseline.png" alt="Residual Baseline Histogram">
</div>

<div class="section">
  <h2>Phase Correlation</h2>
  <table>
    <tr><th>Comparison</th><th>Correlation</th></tr>
    <tr><td>Unknown WAV vs MP3</td><td>{corr_wav_mp3:.4f}</td></tr>
    <tr><td>Baseline MP3-derived vs MP3</td><td>{corr_baseline_mp3:.4f}</td></tr>
  </table>
  <p>Values near <b>1.0</b> indicate almost perfect time/phase alignment. If unknown WAV correlation is very close
  to the baseline MP3-derived correlation, that’s additional evidence that it is sourced from the same MP3 data.</p>
</div>

<div class="section">
  <h2>Band Energy Distribution</h2>
  <h3>Energy per Band</h3>
  <table>
    <tr><th>Signal</th><th>Band</th><th>Average Energy</th></tr>
    {band_table_rows("MP3", band_mp3)}
    {band_table_rows("WAV", band_wav)}
    {band_table_rows("Baseline", band_baseline)}
  </table>

  <h3>High-band Ratios (16k–Nyquist as fraction of total)</h3>
  <table>
    <tr><th>Signal</th><th>High-band Ratio</th></tr>
    <tr><td>MP3</td><td>{highband_mp3:.6e}</td></tr>
    <tr><td>WAV</td><td>{highband_wav:.6e}</td></tr>
    <tr><td>Baseline</td><td>{highband_baseline:.6e}</td></tr>
  </table>

  <p>If the MP3 shows a strong cutoff near 16 kHz (typical lossy behavior) but the WAV suddenly has a lot of energy in the
  16k–Nyquist band, this may indicate some sort of “air reconstruction” or exciter. If the WAV high-band ratio is similar
  to the baseline decoded MP3, that suggests it hasn’t gained real new high-frequency information.</p>
</div>

<div class="section">
  <h2>Core Visuals: MP3 vs WAV vs Baseline</h2>

  <h3>Waveforms (Full)</h3>
  <img src="wave_mp3_full.png" alt="MP3 Waveform Full">
  <img src="wave_wav_full.png" alt="WAV Waveform Full">
  <img src="wave_baseline_full.png" alt="Baseline Waveform Full">

  <h3>Waveforms (0–5s Zoom)</h3>
  <img src="wave_mp3_zoom.png" alt="MP3 Waveform Zoom">
  <img src="wave_wav_zoom.png" alt="WAV Waveform Zoom">
  <img src="wave_baseline_zoom.png" alt="Baseline Waveform Zoom">

  <h3>Amplitude Histograms</h3>
  <img src="hist_mp3.png" alt="MP3 Histogram">
  <img src="hist_wav.png" alt="WAV Histogram">
</div>

<div class="section">
  <h2>Spectrogram Gallery</h2>
  <p>For each of MP3 / WAV / Baseline / Residual, multiple color maps and scales are rendered. Look for:</p>
  <ul>
    <li>Sharp cutoff around ~16 kHz (MP3 typical behavior).</li>
    <li>"Air band" activity in WAV not present in MP3 (could be exciter / upscaler).</li>
    <li>Residual spectrogram showing what’s actually different between signals.</li>
  </ul>

  <h3>MP3</h3>
  <img src="spec_mp3_magma_log.png">
  <img src="spec_mp3_viridis_log.png">
  <img src="spec_mp3_plasma_log.png">

  <h3>WAV</h3>
  <img src="spec_wav_magma_log.png">
  <img src="spec_wav_viridis_log.png">
  <img src="spec_wav_plasma_log.png">

  <h3>Baseline MP3-derived</h3>
  <img src="spec_baseline_magma_log.png">
  <img src="spec_baseline_viridis_log.png">
  <img src="spec_baseline_plasma_log.png">

  <h3>Residuals (Unknown vs MP3)</h3>
  <img src="spec_residual_unknown_magma_log.png">
  <img src="spec_residual_unknown_viridis_log.png">
  <img src="spec_residual_unknown_plasma_log.png">

  <h3>Residuals (Baseline vs MP3)</h3>
  <img src="spec_residual_baseline_magma_log.png">
  <img src="spec_residual_baseline_viridis_log.png">
  <img src="spec_residual_baseline_plasma_log.png">
</div>

<div class="section">
  <h2>Synthetic CHIRP Reference (Ground Truth of MP3 Damage)</h2>
  <p>This section generates a synthetic wideband CHIRP (20 Hz → ~Nyquist), encodes it to MP3, and decodes back.
  That gives you a “clean lab specimen” of exactly how MP3 behaves on full-band content.</p>

  <h3>Spectrograms</h3>
  <img src="spec_chirp_original_log.png" alt="Chirp Original Spectrogram">
  <img src="spec_chirp_decoded_log.png" alt="Chirp MP3-decoded Spectrogram">

  <p>Use this as a visual reference when comparing your Suno spectrograms to see if their
  high-frequency behavior resembles standard MP3 artifacts, an excited/upscaled version, or something else.</p>
</div>

<div class="section">
  <h2>How to Read This Report</h2>
  <ul>
    <li><b>If the unknown WAV nulls against the MP3 about as well as the baseline does</b> (both full-band and &lt;16kHz), and
    phase correlations are close, that’s strong evidence it’s MP3-derived.</li>
    <li><b>If the WAV has large residuals below 16 kHz</b> and a lot of genuinely different structure, it acts more like a
    separate render / master.</li>
    <li><b>If high-band (16k–Nyquist) energy appears only in WAV</b> and not MP3, especially with a smeared / “fake air”
    texture in the spectrogram, that’s consistent with upscaling, exciters, or reconstruction algorithms rather than true
    raw model output.</li>
  </ul>
</div>

</body>
</html>
""")

    print("\nDone.")
    print(f"Report written to: {html_path}")
    print("Open it in your browser to inspect all metrics and visuals.")


if __name__ == "__main__":
    main()
