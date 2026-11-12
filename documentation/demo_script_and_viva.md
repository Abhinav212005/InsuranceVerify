# Demo script (5 minutes) and viva notes

## Before the demo
1. Start the backend and the frontend (see the README). Open http://localhost:3000.
2. Keep a degraded sample ready (for example a rotated, noisy page made from `dataset/synthetic/clean` with `backend/scripts/dataset/degrade.py`, or a phone-style photo) and one clean policy.
3. Have `evaluation_results/charts/01_ablation_bar.png` open in a second tab as a backup.

## Script
1. **Problem (30 s).** Staff retype policy numbers, dates and amounts from poor scans. The system assists; it never approves claims (point at the banner).
2. **Upload (45 s).** Drop the rotated, noisy page. Show the stage chips moving: preprocessing, OCR, classifying, extracting, validating.
3. **Image Processing Studio (90 s).** Original to Final. Use the before/after slider. Show the detected skew angle, the histogram, and the global / Otsu / adaptive comparison. Say why each step exists.
4. **OCR (30 s).** Word boxes, low-confidence words in amber, the threshold slider.
5. **Intelligence (45 s).** Category with confidence and evidence keywords. Correct one wrong field; the status flips from Needs Review to Valid (data complete).
6. **Validation Center (20 s).** Queue of Needs Review and Incomplete documents.
7. **Money slide (40 s).** The preprocessing ablation chart: mean CER falls from 15.7% on the raw image to 3.2% with the default recipe.

## Viva questions
- **Why Otsu, and when does it fail?** It picks the threshold that maximises between-class variance, which is ideal for evenly lit pages. Under shadows one global value cannot work, so the default uses adaptive thresholding (experiment 2).
- **Why adaptive thresholding as the default?** It handled noise, shadow and faded print in the experiments. Its weakness is very heavy noise, which the median filter before it mitigates.
- **Why CLAHE is not in the default recipe?** On text-on-white scans it amplified background noise and slightly raised CER (A3 versus A2). It stays in R4 for faded documents.
- **Why median over Gaussian denoising?** Median removes salt-and-pepper noise and keeps edges; Gaussian 5x5 blurred thin strokes and raised CER to 0.238 (experiment 3).
- **How does deskew work and how accurate is it?** Text lines are found on the binary image and the angle is estimated, then the page is rotated. Mean angle error 0.04 degrees (max 0.5); CER at 15 degrees drops from 0.99 to 0.13.
- **What happens when the page boundary is not found?** The warp is skipped and the page is processed as is; nothing crashes.
- **Does preprocessing ever hurt?** Yes. Adaptive thresholding alone raised WER (A4); deskew and morphology recovered it. Thresholding at 100 DPI hurts too (experiment 5), so upscaling comes first in grayscale pipelines. Stating this is part of the result.
- **How is CER computed?** Levenshtein edit distance over characters divided by reference length, on normalised text.
- **Is 99-100% classification accuracy credible?** It measures this system on a synthetic benchmark whose categories have distinctive vocabulary; real-world accuracy would be lower. The report says so.
- **What is not covered?** Handwriting, tables, non-English text, real scans at scale, authentication, cloud deployment.
