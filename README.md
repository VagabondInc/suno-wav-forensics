# Suno WAV Forensic Analyzer  
A reproducible audio-forensics toolkit for verifying whether Suno-exported WAV files are truly lossless originals or reconstructed/upscaled versions of MP3-quality audio.

This tool performs correlation, null testing, spectral comparison, HF-energy analysis, generates visual evidence (waveforms + spectrograms), and outputs a full HTML report that includes confidence labels and a heuristic probability score indicating MP3-derived likelihood.

---

## Features

- **Full-band Pearson correlation analysis**
- **Residual (null-test) RMS comparison**
- **Low-band (<16 kHz) vs full-band residual matching**
- **High-frequency energy evaluation (16–22 kHz)**
- **WAV/MP3 high-band ratio analysis**
- **Waveform plots**
- **Spectrograms (log + linear frequency)**
- **Residual spectrogram (WAV − MP3)**
- **Confidence labels for every metric**
- **Heuristic MP3-derived probability score (0–100%)**
- **Automatic HTML report generation**

All results are local and fully reproducible.

---

## Requirements

- Python 3.9+
- macOS (recommended; Linux works, Windows may need minor path tweaks)
- Python packages:
  - `librosa`
  - `numpy`
  - `matplotlib`
  - `soundfile`

Install dependencies:

```bash
pip install numpy librosa matplotlib soundfile
```

⸻

Installation

Clone the repository:

```
git clone https://github.com/yourusername/suno-wav-forensics.git
cd suno-wav-forensics
```

Make the script executable (optional):

```
chmod +x suno_wav_forensics.py
```

⸻

Usage

Place your comparison files here:
```
	•	~/Downloads/suno-test.mp3
	•	~/Downloads/suno-test.wav
```

Run the tool:

```
python3 suno_wav_forensics.py \
  --mp3 ~/Downloads/suno-test.mp3 \
  --wav ~/Downloads/suno-test.wav \
  --out ~/Downloads/suno_report.html
```

After running, open the report:

```
open ~/Downloads/suno_report.html
```

The output directory will contain:
	•	Spectrogram images
	•	Waveform images
	•	Residual spectrogram
	•	A full HTML report

⸻

Output: What You’ll See

The generated HTML report includes:

1. Summary Probability Score

A single heuristic number (0–100%) estimating how likely the provided WAV is derived from MP3-quality audio. This score is computed from weighted evidence:
	•	Correlation strength
	•	Residual similarity
	•	Low-band vs full-band behavior
	•	High-frequency energy reconstruction

2. Detailed Metrics w/ Confidence Labels

Each metric includes:
	•	Raw numeric value
	•	Interpretation
	•	“Low”, “Medium”, or “High” confidence indicating how strongly it supports the MP3-derived hypothesis

3. Visual Evidence
	•	MP3 waveform
	•	WAV waveform
	•	MP3 spectrogram (log + linear)
	•	WAV spectrogram (log + linear)
	•	Residual spectrogram (WAV − MP3)

These images visually reveal HF roll-offs, reconstruction artifacts, broadband residual patterns, and shared underlying structure.

⸻

Methodology

This tool uses standard audio forensics techniques:
	1.	Normalization
Ensures amplitude differences do not pollute correlation or null tests.
	2.	Pearson correlation
Detects shared timing and sample-by-sample similarity.
	3.	Null testing (residual analysis)
Subtracting WAV − MP3 exposes broadband or HF-band differences.
	4.	Band-limited analysis (<16 kHz)
Determines whether changes are only in reconstructed “air” or across the whole band.
	5.	High-frequency energy ratio
MP3 files typically roll-off steeply at ~16 kHz; reconstructed WAVs often show synthetic high-band energy.
	6.	Scoring system
A weighted heuristic model combines all evidence into a single probability score.

This tool does not claim cryptographic certainty; it provides reproducible statistical evidence.

⸻

Limitations
	•	The probability score is heuristic, not a mathematical proof.
	•	Perfect MP3 time-alignment is assumed after normalization; severely misaligned files require pre-processing.
	•	Spectral reconstruction algorithms vary; extremely advanced models may require updated heuristics.
	•	Analysis is done at a fixed sample rate (default 44.1 kHz).

⸻

Contributing

Pull requests are welcome for:
	•	New forensic metrics
	•	Better scoring models
	•	Support for multi-channel audio
	•	More visualization styles
	•	Plugin-based DSP reconstruction detectors

Open issues if you find corner cases or want new features.

⸻

License

MIT License. Feel free to fork, modify, and integrate into your workflows.

⸻

Contact

For questions or contributions, open an issue on GitHub.
