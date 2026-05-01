# References

## 採用コンポーネント

### DeepFilterNet 3

- 論文: Schröter et al., *DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering*, ICASSP 2022
- リポジトリ: https://github.com/Rikorose/DeepFilterNet
- ライセンス: MIT / Apache 2.0 デュアル
- 採用バージョン: DeepFilterNet 3
- 設定:
  - サンプリングレート: 48 kHz
  - フレーム長: 20 ms (`fft_size: 960`)
  - ホップ長: 10 ms (`hop_size: 480`)
  - Lookahead: 20 ms (`df_lookahead: 2`)
  - アルゴリズム遅延: 約 30 ms

### silero-vad

- リポジトリ: https://github.com/snakers4/silero-vad
- ライセンス: MIT
- フレーム長: 30 ms
- 出力: speech 確信度 [0, 1]
- ONNX 形式で配布、軽量（約 2 MB）

### ECAPA-TDNN（SpeechBrain）

- 論文: Desplanques et al., *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*, Interspeech 2020
- 公開モデル: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- ライセンス: Apache 2.0（コード）+ VoxCeleb 訓練データ依存
- 出力: 192 次元埋め込み
- 訓練データ: VoxCeleb1 + VoxCeleb2

### F0 抽出

- **YIN**（採用候補 1）: De Cheveigné & Kawahara, 2002, アルゴリズム自体は public domain
- **CREPE**（採用候補 2）: Kim et al., *CREPE: A Convolutional Representation for Pitch Estimation*, ICASSP 2018, https://github.com/marl/crepe (MIT)
- **SwiftF0**（採用候補 3）: 軽量、Apple Silicon 最適化

## 検討して却下した手法・モデル

### TSE（オフライン）

- **ConVoiFilter** (Nguyen et al., ICASSP 2024): https://huggingface.co/nguyenvulebinh/voice-filter
  - License: Apache 2.0（コード+重み）
  - チャンク 5 秒、リアルタイム不可、却下
- **ESPnet TD-SpeakerBeam** (LibriMix 16 kHz): https://huggingface.co/espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw
  - License: CC BY 4.0
  - 双方向アテンション、causal 化には再訓練必要、却下
- **SpEx+** (Ge et al., 2020): https://github.com/gemengtju/SpEx_Plus
  - License: MIT（コード）、訓練データ WSJ0 が LDC 商用ライセンス
  - 8 kHz fixed、却下
- **MossFormer2 系**: https://github.com/modelscope/ClearerVoice-Studio
  - License: Apache 2.0
  - 音声のみ TSE は SpEx+ 8 kHz のみ、48 kHz 版は SE/SR

### TSE（ストリーミング、論文ベース）

- **VoiceFilter-Lite** (Wang et al., Interspeech 2020)
  - log-mel 入出力で波形再合成不可、ASR 専用、通話用途には不適合
- **E3Net** (Liu et al., Microsoft, 2022): 公式コード非公開
- **pDCCRN** (Eskimez et al., Microsoft, ICASSP 2022): 公式コード非公開
- **SpeakerBeam-SS** (Sato et al., NTT, Interspeech 2024): https://arxiv.org/abs/2407.01857
  - S4D ベース、causal、軽量、公式コード非公開
- **TEA-PSE 1/2/3** (Ju et al., Tencent): 商用化のため非公開
- **pDeepFilterNet2** (Orosound, SHNU): 公式コード非公開

### 48 kHz PSE/TSE（探索結果: 公開モデルなし）

商用化価値が高い領域のため、論文発表されてもオープン化されない構造的傾向：

- Personalized PercepNet (Amazon, 2021): 非公開
- TEA-PSE 1/2/3 (Tencent): 非公開
- DNS Challenge baseline (Microsoft): 出力サンプルのみ、モデル重み非公開
- pDeepFilterNet2: 非公開

Hugging Face / GitHub 全般を系統的に探索した結果、48 kHz クリーンライセンスの TSE/PSE モデルは存在しないと結論。

### Speech Restoration / Super Resolution

- **MossFormer2_SR_48K (HiFi-SR)** (Zhao et al., ICASSP 2025): https://huggingface.co/alibabasglab/MossFormer2_SR_48K
  - License: Apache 2.0
  - 4 秒チャンク、GAN 生成（TTS 訓練データ）、リアルタイム不可、話者個性変質懸念、却下

## ハードゲーティング型の理論的基盤

### Personal VAD

- 論文: Ding et al., *Personal VAD: Speaker-Conditioned Voice Activity Detection*, 2019, https://arxiv.org/abs/1908.04284
- 重要な記述: "Score Combination (SC)" を baseline として提示。事前訓練済み VAD と SV を組み合わせる方式で、新規モデル訓練不要であることを明示している。
- 本プロジェクトの理論的根拠となる手法
- Personal VAD 2.0: https://arxiv.org/abs/2204.03793
- 非公式実装: https://github.com/pirxus/personalVAD

### Speaker-Dependent VAD

- Sholokhov et al., *End-to-End Speaker-Dependent Voice Activity Detection*, 2020, https://arxiv.org/abs/2009.09906

## 訓練データセット（参考）

本プロジェクトでは追加訓練を行わないため、既存モデルが使用したデータセットを記録のみ：

| データセット | License | コメント |
|---|---|---|
| LibriSpeech | CC BY 4.0 | 商用 OK |
| VoxCeleb1 / VoxCeleb2 | Custom | BBC/YouTube 由来、グレーゾーン |
| VCTK | CC BY 4.0 / ODC-By 1.0 | 商用 OK |
| MUSAN | Apache 2.0 | 商用 OK |
| DEMAND | CC BY-SA 3.0 | 商用 OK |
| DNS Challenge | MIT (code) / CC BY 4.0 (data) | 商用 OK |
| WHAM! | CC BY-NC 4.0 | **非商用のみ**、本プロジェクトでは未使用 |
| WSJ0 | LDC proprietary | 有償、本プロジェクトでは未使用 |

## 関連ツールキット

- **WeSep** (Wang et al., 2024): https://github.com/wenet-e2e/WeSep
  - TSE 用ツールキット、LICENSE 不在のため商用利用不可
  - 事前学習済みモデルは未公開
- **SpeechBrain**: https://github.com/speechbrain/speechbrain
  - Apache 2.0、ECAPA-TDNN 重みの配布元
- **ESPnet**: https://github.com/espnet/espnet
  - Apache 2.0、TSE モデル重みの配布元
- **Asteroid**: https://github.com/asteroid-team/asteroid
  - MIT、TSE 含む音源分離全般

## 関連商用製品

- **Krisp**: https://krisp.ai/ — クローズドソース、参考リファレンス
- **NVIDIA Maxine**: https://developer.nvidia.com/maxine — GPU 前提、非商用ライセンスベース
- **Microsoft Teams Personalized Speech Enhancement**: 内蔵機能、技術詳細非公開
