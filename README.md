# wearable-pipeline

Puts wearable streams on one clock and preprocesses pupil and heart-rate variability.

- **One clock:** the eye tracker is the reference. UTC and phone-local timestamps are converted, and sync checks give each logger a constant offset.
- **Pupil:** after Kret & Sjak-Shie (2019), each eye separately: 1.5–9 mm gate, blinks plus 500 ms masked, dilation-speed filter, gap padding, short-gap interpolation, 4 Hz low pass, then the eyes combined. Every step reports what it removed.
- **HRV:** RR cleaning, RMSSD, SDNN, pNN50 and LF/HF. No difference is taken across a removed beat or a dropout.

![Simulated signals on one clock](docs/demo.png)

## Install

Tested with Python 3.12.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick example

```sh
python -m wearable_pipeline.demo
```

```python
from wearable_pipeline import hrv
from wearable_pipeline.demo import process, simulate

data = simulate()
result = process(data)
print(result["fit"].offset_s)
print(hrv.rmssd(data["rr"], result["keep"], result["beats"]))
```

## Limits

No clock drift is fitted and one UTC offset is assumed per recording. Pupil size is not corrected for luminance, and the published method's isolated-island filter is not implemented. The 20% RR rule re-seeds after five agreeing beats. Pass `reseed_after=None` for the strict rule, which discards everything after a true step in heart rate. This is research code and not a medical device.

## Licence and credit

Pupil method: Kret, M. E., & Sjak-Shie, E. E. (2019). Preprocessing pupil size data: Guidelines and code. *Behavior Research Methods, 51*, 1336–1342.

GPL-3.0-only, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The pupil module was developed with reference to [the authors' GPL-3.0 MATLAB code](https://github.com/ElioS-S/pupil-size). Developed by the Cambridge Cognitive Architecture Lab.
