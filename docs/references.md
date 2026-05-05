# References

## Adopted components

### DeepFilterNet 3

- Paper: Schröter et al., *DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering*, ICASSP 2022.
- Repository: https://github.com/Rikorose/DeepFilterNet
- License: MIT / Apache 2.0 dual.
- Adopted version: DeepFilterNet 3.
- Configuration:
  - Sampling rate: 48 kHz.
  - Frame length: 20 ms (`fft_size: 960`).
  - Hop length: 10 ms (`hop_size: 480`).
  - Lookahead: 20 ms (`df_lookahead: 2`).
  - Algorithmic latency: ~30 ms.

### silero-vad

- Repository: https://github.com/snakers4/silero-vad
- License: MIT.
- Frame length: 30 ms.
- Output: speech confidence in `[0, 1]`.
- Distributed in ONNX form, lightweight (~2 MB).

### ECAPA-TDNN (SpeechBrain)

- Paper: Desplanques et al., *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*, Interspeech 2020.
- Published model: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- License: Apache 2.0 (code) + dependency on VoxCeleb training data.
- Output: 192-dim embedding.
- Training data: VoxCeleb1 + VoxCeleb2.

### F0 extraction

- **YIN** (candidate 1): De Cheveigné & Kawahara, 2002; the algorithm itself is public domain.
- **CREPE** (candidate 2): Kim et al., *CREPE: A Convolutional Representation for Pitch Estimation*, ICASSP 2018, https://github.com/marl/crepe (MIT).
- **SwiftF0** (candidate 3): lightweight, optimised for Apple Silicon.

## Methods / models considered and rejected

### TSE (offline)

- **ConVoiFilter** (Nguyen et al., ICASSP 2024): https://huggingface.co/nguyenvulebinh/voice-filter
  - License: Apache 2.0 (code + weights).
  - 5 s chunks, not real-time, rejected.
- **ESPnet TD-SpeakerBeam** (LibriMix 16 kHz): https://huggingface.co/espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw
  - License: CC BY 4.0.
  - Bidirectional attention; causal-ising would require retraining; rejected.
- **SpEx+** (Ge et al., 2020): https://github.com/gemengtju/SpEx_Plus
  - License: MIT (code); training data WSJ0 is under LDC's commercial licence.
  - 8 kHz fixed; rejected.
- **MossFormer2 family**: https://github.com/modelscope/ClearerVoice-Studio
  - License: Apache 2.0.
  - The audio-only TSE variant is SpEx+ 8 kHz only; the 48 kHz variant is SE / SR.

### TSE (streaming, paper-based)

- **VoiceFilter-Lite** (Wang et al., Interspeech 2020): log-mel I/O — cannot resynthesise waveforms; ASR-only; unsuitable for call use.
- **E3Net** (Liu et al., Microsoft, 2022): no official code released.
- **pDCCRN** (Eskimez et al., Microsoft, ICASSP 2022): no official code released.
- **SpeakerBeam-SS** (Sato et al., NTT, Interspeech 2024): https://arxiv.org/abs/2407.01857
  - S4D-based, causal, lightweight; no official code released.
- **TEA-PSE 1 / 2 / 3** (Ju et al., Tencent): closed source for commercialisation.
- **pDeepFilterNet2** (Orosound, SHNU): no official code released.

### 48 kHz PSE / TSE (search result: no public model)

This is a commercially valuable area, so even when papers are published the results tend not to be opened structurally:

- Personalized PercepNet (Amazon, 2021): closed.
- TEA-PSE 1 / 2 / 3 (Tencent): closed.
- DNS Challenge baseline (Microsoft): only output samples are released; weights are closed.
- pDeepFilterNet2: closed.

After a systematic search of Hugging Face / GitHub, the conclusion is that no 48 kHz cleanly-licensed TSE / PSE model exists.

### Speech Restoration / Super Resolution

- **MossFormer2_SR_48K (HiFi-SR)** (Zhao et al., ICASSP 2025): https://huggingface.co/alibabasglab/MossFormer2_SR_48K
  - License: Apache 2.0.
  - 4 s chunks; GAN-generated (TTS training data); not real-time; concerns about distortion of speaker characteristics; rejected.

## Theoretical basis for hard-gating

### Personal VAD

- Paper: Ding et al., *Personal VAD: Speaker-Conditioned Voice Activity Detection*, 2019, https://arxiv.org/abs/1908.04284
- Key passage: presents "Score Combination (SC)" as a baseline — combining a pretrained VAD and a SV system, explicitly stating that no new model training is required.
- The theoretical basis for this project.
- Personal VAD 2.0: https://arxiv.org/abs/2204.03793
- Unofficial implementation: https://github.com/pirxus/personalVAD

### Speaker-Dependent VAD

- Sholokhov et al., *End-to-End Speaker-Dependent Voice Activity Detection*, 2020, https://arxiv.org/abs/2009.09906

## Training datasets (for reference)

This project does no additional training; the datasets used by the existing models are recorded only for reference:

| Dataset | License | Comment |
|---|---|---|
| LibriSpeech | CC BY 4.0 | Commercial OK |
| VoxCeleb1 / VoxCeleb2 | Custom | BBC / YouTube-derived; grey zone |
| VCTK | CC BY 4.0 / ODC-By 1.0 | Commercial OK |
| MUSAN | Apache 2.0 | Commercial OK |
| DEMAND | CC BY-SA 3.0 | Commercial OK |
| DNS Challenge | MIT (code) / CC BY 4.0 (data) | Commercial OK |
| WHAM! | CC BY-NC 4.0 | **Non-commercial only**; not used in this project |
| WSJ0 | LDC proprietary | Paid; not used in this project |

## Related toolkits

- **WeSep** (Wang et al., 2024): https://github.com/wenet-e2e/WeSep
  - TSE toolkit; commercial use blocked due to missing LICENSE.
  - No pretrained models published.
- **SpeechBrain**: https://github.com/speechbrain/speechbrain
  - Apache 2.0; distribution source for the ECAPA-TDNN weights.
- **ESPnet**: https://github.com/espnet/espnet
  - Apache 2.0; distribution source for TSE model weights.
- **Asteroid**: https://github.com/asteroid-team/asteroid
  - MIT; covers source separation in general, including TSE.

## Related commercial products

- **Krisp**: https://krisp.ai/ — closed source; reference for comparison.
- **NVIDIA Maxine**: https://developer.nvidia.com/maxine — GPU-only; non-commercial-licence-based.
- **Microsoft Teams Personalized Speech Enhancement**: built-in feature; technical details undisclosed.
