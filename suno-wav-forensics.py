#!/usr/bin/env python3
"""
suno_wav_forensics.py

Usage (from macOS Terminal):

    python3 suno_wav_forensics.py \
        --mp3 ~/Downloads/suno-test.mp3 \
        --wav ~/Downloads/suno-test.wav \
        --out ~/Downloads/suno_report.html
"""

import argparse
import os
import math
import json
from dataclasses import dataclass, asdict

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# -----------------------------
# Utility & loading
# -----------------------------

def load_audio(path, sr=44100, mono=True):
    y, fs = librosa.load(path, sr=sr, mono=mono)
    return y.astype(np.float32), fs

def rms(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x**2) + 1e-12))

def normalize(x):
    peak = np.max(np.abs(x))
    if peak < 1e-9:
        return x
    return x / peak

def bandpass_fft(x, sr, f_low, f_high):
    """
    Zero everything outside [f_low, f_high] in the FFT domain.
    Return time-domain signal.
    """
    x = np.asarray(x, dtype=np.float32)
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/sr)

    mask = (freqs >= f_low) & (freqs <= f_high)
    X_filtered = np.zeros_like(X)
    X_filtered[mask] = X[mask]

    y = np.fft.irfft(X_filtered, n=N)
    return y.astype(np.float32)

def highband_energy_ratio(x, sr, split_hz=16000.0, top_hz=22050.0):
    """
    Energy in [split_hz, top_hz] / total energy.
    """
    x = np.asarray(x, dtype=np.float32)
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/sr)

    total = np.sum(np.abs(X)**2) + 1e-12
    mask_hi = (freqs >= split_hz) & (freqs <= top_hz)
    hi = np.sum(np.abs(X[mask_hi])**2)
    return float(hi / total)

def pearson_corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(len(a), len(b))
    if n < 10:
        return 0.0
    a = a[:n]
    b = b[:n]
    a = a - np.mean(a)
    b = b - np.mean(b)
    num = float(np.sum(a * b))
    den = float(np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-12)
    return num / den

# -----------------------------
# Analysis dataclasses
# -----------------------------

@dataclass
class MetricResult:
    name: str
    value: float
    explanation: str
    confidence: str  # "Low" | "Medium" | "High"

@dataclass
class AnalysisResults:
    correlation: MetricResult
    residual_full_ratio: MetricResult
    residual_low_ratio: MetricResult
    hf_energy_mp3: MetricResult
    hf_energy_wav: MetricResult
    hf_ratio_wav_vs_mp3: MetricResult
    probability_mp3_derived: float
    probability_explanation: str

# -----------------------------
# Metric interpretation helpers
# -----------------------------

def label_confidence(value, low_thr, high_thr, invert=False):
    """
    Simple 3-way confidence label.
    If invert=False:
        value < low_thr  -> Low
        low_thr..high_thr -> Medium
        > high_thr       -> High
    If invert=True, flip the logic (useful when "lower is stronger").
    """
    if invert:
        # lower = stronger
        if value > high_thr:
            return "Low"
        elif value > low_thr:
            return "Medium"
        else:
            return "High"
    else:
        # higher = stronger
        if value < low_thr:
            return "Low"
        elif value < high_thr:
            return "Medium"
        else:
            return "High"

# -----------------------------
# Core analysis
# -----------------------------

def run_analysis(mp3_path, wav_path, out_dir, sr=44100):
    os.makedirs(out_dir, exist_ok=True)

    # Load & normalize
    mp3, fs1 = load_audio(mp3_path, sr=sr)
    wav, fs2 = load_audio(wav_path, sr=sr)

    mp3 = normalize(mp3)
    wav = normalize(wav)

    n = min(len(mp3), len(wav))
    mp3 = mp3[:n]
    wav = wav[:n]

    # --- Correlation ---
    corr = pearson_corr(wav, mp3)

    corr_expl = (
        f"Pearson correlation between WAV and MP3 is {corr:.6f}. "
        "Values above ~0.99 usually indicate the same underlying performance and "
        "very similar mix; above ~0.999 suggests extremely tight similarity."
    )
    corr_conf = label_confidence(corr, low_thr=0.98, high_thr=0.995, invert=False)
    corr_metric = MetricResult(
        name="Full-band Pearson correlation",
        value=corr,
        explanation=corr_expl,
        confidence=corr_conf,
    )

    # --- Residual / null test (full band) ---
    resid = wav - mp3
    resid_rms = rms(resid)
    mp3_rms = rms(mp3)
    resid_ratio_full = resid_rms / (mp3_rms + 1e-12)

    resid_full_expl = (
        f"Residual RMS / MP3 RMS (full band) = {resid_ratio_full:.6f}. "
        "Values near 0 mean almost perfect cancellation (identical signals). "
        "Values below ~0.15 indicate the signals are very similar but not identical."
    )
    resid_full_conf = label_confidence(resid_ratio_full, low_thr=0.25, high_thr=0.15, invert=True)
    resid_full_metric = MetricResult(
        name="Residual RMS ratio (full band)",
        value=resid_ratio_full,
        explanation=resid_full_expl,
        confidence=resid_full_conf,
    )

    # --- Residual / null test (<16 kHz band) ---
    low_mp3 = bandpass_fft(mp3, sr, 20.0, 16000.0)
    low_wav = bandpass_fft(wav, sr, 20.0, 16000.0)
    low_resid = low_wav - low_mp3
    low_resid_ratio = rms(low_resid) / (rms(low_mp3) + 1e-12)

    resid_low_expl = (
        f"Residual RMS / MP3 RMS (<16 kHz) = {low_resid_ratio:.6f}. "
        "If this closely matches the full-band residual ratio, it suggests the "
        "difference is broadband processing, not only in the extreme high end."
    )
    # Compare similarity of low-band and full-band
    similarity = 1.0 - abs(low_resid_ratio - resid_ratio_full) / max(resid_ratio_full, 1e-6)
    # similarity close to 1 => strong evidence of broadband similarity
    resid_low_conf = label_confidence(similarity, low_thr=0.6, high_thr=0.85, invert=False)
    resid_low_metric = MetricResult(
        name="Residual RMS ratio (<16 kHz)",
        value=low_resid_ratio,
        explanation=resid_low_expl,
        confidence=resid_low_conf,
    )

    # --- High-frequency energy ratios ---
    hf_mp3 = highband_energy_ratio(mp3, sr, split_hz=16000.0, top_hz=sr/2)
    hf_wav = highband_energy_ratio(wav, sr, split_hz=16000.0, top_hz=sr/2)

    hf_ratio = (hf_wav + 1e-9) / (hf_mp3 + 1e-9)

    hf_mp3_expl = (
        f"MP3 high-band (≥16 kHz) energy fraction = {hf_mp3:.6e}. "
        "Very low values here are consistent with standard lossy roll-off."
    )
    hf_wav_expl = (
        f"WAV high-band (≥16 kHz) energy fraction = {hf_wav:.6e}. "
        "Larger values than MP3 often indicate reconstructed or synthetic 'air'."
    )
    hf_ratio_expl = (
        f"WAV high-band / MP3 high-band energy ratio = {hf_ratio:.2f}. "
        "Very large ratios (e.g., > 5–10×) suggest that the WAV has extra "
        "high-frequency energy on top of an MP3-like base, not original detail."
    )

    hf_mp3_conf = label_confidence(hf_mp3, low_thr=1e-3, high_thr=1e-2, invert=True)
    hf_wav_conf = label_confidence(hf_wav, low_thr=1e-3, high_thr=1e-2, invert=False)
    hf_ratio_conf = label_confidence(hf_ratio, low_thr=3.0, high_thr=10.0, invert=False)

    hf_mp3_metric = MetricResult(
        name="MP3 high-band energy fraction (≥16 kHz)",
        value=hf_mp3,
        explanation=hf_mp3_expl,
        confidence=hf_mp3_conf,
    )
    hf_wav_metric = MetricResult(
        name="WAV high-band energy fraction (≥16 kHz)",
        value=hf_wav,
        explanation=hf_wav_expl,
        confidence=hf_wav_conf,
    )
    hf_ratio_metric = MetricResult(
        name="High-band energy ratio (WAV / MP3)",
        value=hf_ratio,
        explanation=hf_ratio_expl,
        confidence=hf_ratio_conf,
    )

    # -----------------------------
    # Heuristic probability scoring
    # -----------------------------
    # This is explicitly an inference, not a "fact".

    score = 0.0
    explanations = []

    # 1. Correlation: 0.99–0.995–1.0
    if corr > 0.995:
        score += 35
        explanations.append("Very high correlation (>0.995) suggests same underlying mix/source.")
    elif corr > 0.99:
        score += 20
        explanations.append("High correlation (>0.99) suggests closely related audio.")
    else:
        explanations.append("Correlation not extremely high; weakens MP3-derived hypothesis.")

    # 2. Residual full-band: low residual = high similarity
    if resid_ratio_full < 0.05:
        score += 25
        explanations.append("Residual ratio <0.05 (very strong similarity; near-null).")
    elif resid_ratio_full < 0.15:
        score += 18
        explanations.append("Residual ratio <0.15 (strong similarity, consistent with MP3-derived + processing).")
    elif resid_ratio_full < 0.3:
        score += 8
        explanations.append("Residual ratio <0.3 (moderate similarity).")
    else:
        explanations.append("Residual ratio is large; weakens MP3-derived hypothesis.")

    # 3. Broadband similarity: low-band residual close to full-band residual
    diff = abs(low_resid_ratio - resid_ratio_full)
    rel = diff / max(resid_ratio_full, 1e-6)
    if rel < 0.1:
        score += 15
        explanations.append("Low-band and full-band residuals match closely (broadband processing).")
    elif rel < 0.25:
        score += 8
        explanations.append("Low-band vs full-band residuals are somewhat similar.")
    else:
        explanations.append("Residual pattern differs between bands; weaker evidence.")

    # 4. HF shelf / reconstruction: MP3 HF very low, WAV HF higher
    if hf_mp3 < 1e-3 and hf_ratio > 5.0:
        score += 20
        explanations.append(
            "MP3 high-band energy extremely low, but WAV has much more HF energy "
            "(strongly suggests synthetic/reconstructed 'air' on top of a lossy base)."
        )
    elif hf_mp3 < 5e-3 and hf_ratio > 3.0:
        score += 10
        explanations.append(
            "MP3 high-band relatively low and WAV HF noticeably higher (suggestive but not decisive)."
        )
    else:
        explanations.append("HF behavior not strongly indicative of MP3-derived reconstruction.")

    # Clip score into [0, 100]
    score = max(0.0, min(100.0, score))

    prob_expl = (
        "This probability is a heuristic estimate of how likely the WAV is derived from "
        "the same lossy MP3-like source, possibly with additional DSP or upscaling. "
        "It is not a mathematical proof, but aggregates correlation, null-test residuals, "
        "and high-frequency behavior into a single interpretable number."
    )

    results = AnalysisResults(
        correlation=corr_metric,
        residual_full_ratio=resid_full_metric,
        residual_low_ratio=resid_low_metric,
        hf_energy_mp3=hf_mp3_metric,
        hf_energy_wav=hf_wav_metric,
        hf_ratio_wav_vs_mp3=hf_ratio_metric,
        probability_mp3_derived=score,
        probability_explanation=prob_expl + " " + " ".join(explanations),
    )

    # -----------------------------
    # Visualization
    # -----------------------------
    def save_waveform(y, sr, title, filename):
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()

    def save_spectrogram(y, sr, title, filename, log_scale=True):
        plt.figure(figsize=(10, 4))
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
        S_db = librosa.power_to_db(S, ref=np.max)
        if log_scale:
            librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='log')
            plt.ylabel("Log frequency (Hz)")
        else:
            librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='linear')
            plt.ylabel("Frequency (Hz)")
        plt.title(title)
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()

    def save_residual_spectrogram(resid, sr, title, filename):
        save_spectrogram(resid, sr, title, filename, log_scale=True)

    # Waveforms
    save_waveform(mp3, sr, "MP3 Waveform", "mp3_waveform.png")
    save_waveform(wav, sr, "WAV Waveform", "wav_waveform.png")

    # Spectrograms
    save_spectrogram(mp3, sr, "MP3 Spectrogram (log freq)", "mp3_spec_log.png")
    save_spectrogram(wav, sr, "WAV Spectrogram (log freq)", "wav_spec_log.png")
    save_spectrogram(mp3, sr, "MP3 Spectrogram (linear freq)", "mp3_spec_lin.png")
    save_spectrogram(wav, sr, "WAV Spectrogram (linear freq)", "wav_spec_lin.png")

    # Residual spectrogram
    save_residual_spectrogram(resid, sr, "Residual Spectrogram (WAV - MP3)", "residual_spec_log.png")

    # -----------------------------
    # HTML Report
    # -----------------------------
    return results

def generate_html(results: AnalysisResults, mp3_path, wav_path, out_html_path, out_dir):
    def metric_row(m: MetricResult):
        return f"""
            <tr>
                <td>{m.name}</td>
                <td>{m.value:.6g}</td>
                <td>{m.confidence}</td>
                <td>{m.explanation}</td>
            </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Suno WAV vs MP3 Forensic Report</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    background: #0b0b0f;
    color: #f4f4f5;
    padding: 20px;
}}
h1, h2, h3 {{
    color: #f9fafb;
}}
a {{
    color: #38bdf8;
}}
.section {{
    margin-bottom: 32px;
    padding: 16px;
    border-radius: 8px;
    background: #111827;
    border: 1px solid #1f2937;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
}}
th, td {{
    border: 1px solid #1f2937;
    padding: 8px;
    text-align: left;
    vertical-align: top;
}}
th {{
    background: #020617;
}}
.conf-High {{
    color: #22c55e;
    font-weight: 600;
}}
.conf-Medium {{
    color: #eab308;
    font-weight: 600;
}}
.conf-Low {{
    color: #f97316;
    font-weight: 600;
}}
.metric-table td:nth-child(3) {{
    text-align: center;
}}
img {{
    max-width: 100%;
    border-radius: 6px;
    margin: 8px 0;
}}
.badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.85rem;
    margin-right: 8px;
}}
.badge-primary {{
    background: #1d4ed8;
    color: white;
}}
.badge-soft {{
    background: #111827;
    color: #e5e7eb;
    border: 1px solid #374151;
}}
</style>
</head>
<body>

<h1>Suno WAV vs MP3 Forensic Report</h1>

<div class="section">
  <h2>Overview</h2>
  <p><span class="badge badge-primary">Heuristic Inference</span>
     <span class="badge badge-soft">Not a cryptographic proof</span></p>
  <p>This report compares two files:</p>
  <ul>
    <li><b>MP3</b>: {mp3_path}</li>
    <li><b>WAV</b>: {wav_path}</li>
  </ul>
  <p>
    It runs correlation, residual (null test), and spectral analysis to estimate how likely it is
    that the WAV is derived from the same lossy MP3-like source, possibly with extra DSP
    (e.g. exciter, EQ, upscaling).
  </p>
</div>

<div class="section">
  <h2>Headline Result</h2>
  <h3>Estimated likelihood WAV is MP3-derived / MP3-processed: <b>{results.probability_mp3_derived:.2f}%</b></h3>
  <p>{results.probability_explanation}</p>
</div>

<div class="section">
  <h2>Metrics & Confidence Levels</h2>
  <table class="metric-table">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Value</th>
        <th>Confidence</th>
        <th>Interpretation</th>
      </tr>
    </thead>
    <tbody>
      {metric_row(results.correlation)}
      {metric_row(results.residual_full_ratio)}
      {metric_row(results.residual_low_ratio)}
      {metric_row(results.hf_energy_mp3)}
      {metric_row(results.hf_energy_wav)}
      {metric_row(results.hf_ratio_wav_vs_mp3)}
    </tbody>
  </table>
</div>

<div class="section">
  <h2>Waveforms</h2>
  <h3>MP3 Waveform</h3>
  <img src="mp3_waveform.png" alt="MP3 waveform" />
  <h3>WAV Waveform</h3>
  <img src="wav_waveform.png" alt="WAV waveform" />
</div>

<div class="section">
  <h2>Spectrograms (Log Frequency)</h2>
  <h3>MP3</h3>
  <img src="mp3_spec_log.png" alt="MP3 spectrogram (log)" />
  <h3>WAV</h3>
  <img src="wav_spec_log.png" alt="WAV spectrogram (log)" />
</div>

<div class="section">
  <h2>Spectrograms (Linear Frequency)</h2>
  <h3>MP3</h3>
  <img src="mp3_spec_lin.png" alt="MP3 spectrogram (linear)" />
  <h3>WAV</h3>
  <img src="wav_spec_lin.png" alt="WAV spectrogram (linear)" />
</div>

<div class="section">
  <h2>Residual Spectrogram (WAV − MP3)</h2>
  <img src="residual_spec_log.png" alt="Residual spectrogram (WAV - MP3)" />
  <p>
    This shows what remains after subtracting the MP3 from the WAV. Broadband, low-level
    energy is consistent with mild processing on top of the same lossy core; strong, structured
    residuals would indicate a very different source.
  </p>
</div>

<div class="section">
  <h2>Notes</h2>
  <ul>
    <li>All numerical metrics are computed after time-aligning and normalizing both signals.</li>
    <li>Probability is a heuristic aggregation of multiple metrics; it is not a legal or cryptographic proof.</li>
    <li>Confidence labels describe how strongly each metric alone supports the MP3-derived hypothesis.</li>
  </ul>
</div>

</body>
</html>
"""

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Forensic comparison of Suno WAV vs MP3.")
    parser.add_argument("--mp3", required=True, help="Path to reference MP3 file.")
    parser.add_argument("--wav", required=True, help="Path to WAV file to test.")
    parser.add_argument("--out", required=True, help="Path to HTML report output.")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate for analysis (default: 44100).")

    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    results = run_analysis(args.mp3, args.wav, out_dir, sr=args.sr)
    generate_html(results, args.mp3, args.wav, args.out, out_dir)

    print(f"[+] Report written to: {args.out}")
    print(f"[+] Heuristic MP3-derived likelihood: {results.probability_mp3_derived:.2f}%")

if __name__ == "__main__":
    main()
