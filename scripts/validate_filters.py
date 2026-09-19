from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from eegpipe.preprocessing.filters import design_bandpass_fir, freq_response


def main():
    fs = 256.0
    # 1.0Hz transition width forces ~931 taps. (Passband is effectively ~1 to 44.5 Hz at -6dB)
    h = design_bandpass_fir(fs, 0.5, 45.0, 60.0, 1.0)
    w, mag_db = freq_response(h, fs)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(w, mag_db, 'b')
    ax1.set_ylabel('Magnitude (dB)', color='b')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylim([-90, 5])
    ax1.grid(True)

    w, h_resp = signal.freqz(h, worN=8192, fs=fs)
    phase = np.unwrap(np.angle(h_resp))
    ax2 = ax1.twinx()
    ax2.plot(w, phase, 'g')
    ax2.set_ylabel('Phase (radians)', color='g')

    plt.title('Kaiser FIR Band-pass (0.5-45 Hz)')
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "A_fir_response.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Filter validation plot saved to {out_path}")

if __name__ == "__main__":
    main()
