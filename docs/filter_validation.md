# Preprocessing Filter Validation

## Design Lock Decisions
- **Notch Filter:** 60 Hz IIR (Zero-phase via `filtfilt`) to eliminate US mains interference.
- **Band-pass Filter:** 0.5-45 Hz Kaiser-window FIR filter. FIR chosen over IIR to guarantee linear phase, preventing temporal distortion of transient EEG epileptiform discharges.

## Technical Specifications
- Taps dynamically derived via `scipy.signal.kaiserord` targeting a 60 dB stop-band attenuation and 1.0 Hz transition bandwidth. At `fs = 256 Hz`, this produces approximately 935 taps.
- **Group Delay Mitigation:** Because an FIR filter of this length introduces roughly 2 seconds of delay, it is applied offline using reflect-padding and `scipy.signal.oaconvolve(mode="same")` to re-align the output, achieving a net zero phase delay.

## Test Confirmations
- DC components successfully removed.
- 10 Hz (Alpha band) signal power maintained within 0.5 dB.
- 60 Hz power attenuated by > 40 dB.
- See generated response plot at `figures/A_fir_response.png`.