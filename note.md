# Spot the Fake Photo — Note

**Approach:** No trained model. Three classic signal-processing features, combined
with fixed weights into one score:

1. **FFT ring energy** — screens leak a faint periodic pixel-grid pattern into the
   mid-frequency band of the spectrum; real objects mostly don't.
2. **Color-channel fringing** — RGB subpixels are physically offset on a screen, so
   high-frequency detail per channel doesn't align as well as on a real object.
3. **Autocorrelation peak** — a repeating grid produces a strong off-center peak
   when the image is autocorrelated with itself.

**Accuracy:** Not measured on real photos yet — I don't have your camera/screen
photo set. The three features are directionally correct (verified on synthetic
grid-vs-noise test images) but the normalization ranges and weights in `predict.py`
are placeholders. **Run `calibrate.py real/ screen/` on your ~50/50 photo set and
paste the fitted weights back in — that's required to hit the 95% bar honestly.**

**Latency:** ~35–110 ms per image on CPU (measured on this sandbox's CPU, no GPU).
Dominated by the two FFTs; resizing to 512×512 first keeps it bounded regardless of
input size.

**Cost per image:** On-device — free, runs in tens of ms on a phone CPU, no
network call. If run server-side instead: at ~50–100ms of CPU per image, a cheap
cloud CPU instance (~$0.05/hr) could do a few hundred thousand images/day, roughly
**$0.01–0.05 per 1,000 images** — but on-device is strictly better here since there's
no ML model to protect and no reason to pay for a network round-trip.

**What I'd improve with more time:**
- Calibrate against real photos (see above) — this is the single biggest accuracy lever.
- Add a 4th signal: specular glare/hotspot detection (screens often show a bright
  reflected patch), which is cheap and complements the frequency-based features.
- As cheaters adapt (better screens, anti-glare filters, higher-res recaptures),
  periodically refresh the calibration set with new screen models and re-fit
  `calibrate.py` rather than hand-tuning weights.
- Choose the cutoff by looking at the ROC curve on the calibration set and picking
  the threshold that hits the desired false-accept rate for fraud, not just accuracy.
