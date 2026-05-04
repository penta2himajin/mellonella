# Design Decisions

本ドキュメントは、設計過程で検討した代替案と却下理由、および重要な設計判断の記録を残す。

## D-001: 真の TSE ではなくハードゲーティング型を採用

### 検討した代替案

A. ConVoiFilter（オフライン TSE）の自前運用
B. ESPnet TD-SpeakerBeam の causal 化（要再訓練）
C. SpeakerBeam-SS / E3Net 等の論文ベース実装（要自前実装）
D. ハードゲーティング型（VAD + SV + NS、Personal VAD の Score Combination 方式）

### 採用: D（ハードゲーティング型）

理由:
- A は 5 秒チャンクのオフライン処理で、リアルタイム通話に不適合
- B/C はいずれも自前訓練が必要、「追加訓練不要」要件と整合しない
- D は全コンポーネントが既存事前学習モデルで構成可能
- 「特定単一話者ターゲット」という本質要件を踏まえると、N 人分離の複雑性は不要
- 対象話者音声への副作用が最小（マスク方式の人工感や GAN 生成系のスペクトル変質なし）

### トレードオフ

- 同時発話シーンで完全な分離はできない → FP 許容方針で対応
- 短時間発話（相槌等）では SV 判定が不安定 → 動的チャンクと時間平滑化で対応

## D-002: 48 kHz 出力 + 内部 16 kHz 判定のハイブリッド構成

### 検討した代替案

A. 16 kHz 統一（ConVoiFilter 単体構成）
B. 48 kHz NS + 16 kHz TSE ハイブリッド（MossFormer2_SE_48K + ConVoiFilter）
C. 48 kHz 出力 + DFN3 + 16 kHz 判定（本構成の前身）
D. 3 段カスケード（DFN3 → TSE → MossFormer2_SR_48K）

### 採用: C（DFN3 + 16 kHz 判定）

理由:
- A は ConVoiFilter のチャンク制約（5 秒）でリアルタイム不可
- B は MossFormer2_SE_48K + ConVoiFilter で計算量大、かつ ConVoiFilter のリアルタイム制約は解消されない
- D は MossFormer2_SR_48K が 4 秒チャンク・GAN 生成・TTS 訓練データのため通話用途に不適合（[詳細](references.md#mossformer2_sr_48k-評価)）
- C は DFN3 の 48 kHz フルバンド処理を活用しつつ、判定は ECAPA-TDNN の native レート（16 kHz）で実行できる

### MossFormer2_SR_48K 却下の決定的理由

- アルゴリズム遅延 4 秒（`decode_window: 4`）→ リアルタイム不可
- 訓練データが TTS 合成音声 → out-of-distribution
- GAN 生成による高域生成 → 話者個性の変質
- 3 段カスケードによるアーティファクト累積

## D-003: 順序設計は案A（NS → 判定 → DFN3 後音声を出力）

### 検討した代替案

A. 入力 → DFN3 → 判定 → ゲート → DFN3 後音声を出力
B. 入力 → 判定 → ゲート → DFN3 → 出力
C. 入力 → DFN3 → 判定 / 元音声をゲート → 出力

### 採用: A

理由:
- 判定をクリーン音声で実行できる（精度向上）
- 出力に NS 効果を反映できる（通話品質向上）
- DFN3 を 1 回だけ計算し、判定パスと出力パスで共有 → 計算コスト最小

### 案C の検討経緯

設計途中で「DFN3 アーティファクトを最終出力に乗せない」目的で案C を一時推奨したが、通話用途では NS の利点が NS アーティファクトより大きいと判断し直し、案A に戻した。録音編集等の音楽的用途では案C が再検討の余地あり。

## D-004: 明示登録 + 自動学習の併用

### 検討した代替案

A. 明示登録のみ
B. 自動学習のみ（zero-enrollment）
C. 明示登録 + 自動学習併用

### 採用: C

理由:
- A は経時変化（風邪・疲労・マイク変更）に追従できない
- B は登録時の信頼性が低い（最初のセッションが純粋に対象話者であるという保証がない）
- C は明示登録の信頼性と自動学習の適応性を両立

### Drift 対策（重要）

FP 許容方針 + 自動学習の組み合わせは drift リスクが高い：

- FP 許容で他話者混合フレームも pass しやすい
- それを自動学習に流すと他話者声紋がプールに浸透

対策として以下を必須とする：

1. **二段階閾値**: `θ_learn (0.80) > θ_pass (0.50)` を厳守
2. **Anchor 保護**: 明示登録時の埋め込みは永久保持、自動学習で削除されない
3. **整合性チェック**: 自動学習プールへの追加前に anchor 距離検証
4. **異常検知**: プール中央値の anchor からの逸脱を定期監視、逸脱時はリセット

## D-005: F0 補助判定の追加

### 検討した代替案

A. ECAPA-TDNN 単独（cos similarity のみで判定）
B. ECAPA-TDNN + F0 マッチ
C. ECAPA-TDNN + F0 + Harmonic 構造解析

### 採用: B

理由:
- A だけでは「対象話者と声紋が似た別人」を区別しきれない場合がある
- F0 は個人差が大きく、補助判定として有効
- C は実装コストが高く、初期実装では過剰

### 設計詳細

- F0 はハードフィルタではなく統合スコアに重み β を以って加味
- ガウシアン当てはまりで連続値化（厳密なレンジチェックを避ける）
- 登録時と推論時で発話状態が異なっても誤検出を最小化する設計
- 実装は YIN（DSP ベース、軽量）を第一候補、CREPE（ONNX）を高精度オプションとする

### 重み (α, β) の calibration 履歴

当初は直感で `α=0.8, β=0.2` を仮置き。`scripts/calibrate_alpha_beta.py` で librosa libri1/2/3 × white/pink ノイズ × SNR -5..20 dB に対し α ∈ [0.0, 1.0]、θ ∈ [0.20, 0.55] の joint sweep を実施した結果、**`α=0.9, β=0.1, θ_pass=0.30`** が FP 許容方針 (mean FPR ≤ 0.05) を満たしつつ TPR_median を最大化する操作点だった (α=0.8 と同じ TPR_median=0.84 で FPR_mean が 0.046 → 0.017 に低減)。

参考:
- α=1.0 (cosine 単独, F0 不使用) は TPR_median=0.81 / FPR_mean=0.008 — F0 を全く使わないと TPR が ~3 ポイント下がる
- α=0.9 はその間で「F0 を控えめに使う」スイートスポット

詳細は [`benchmarks/calibration_alpha_beta_summary.json`](benchmarks/calibration_alpha_beta_summary.json) と [`../poc/notebooks/02_alpha_beta_sweep.py`](../poc/notebooks/02_alpha_beta_sweep.py)。

> **Caveat**: calibration の話者は全て英語 LibriSpeech で F0 分布が近い。男女混合や母語横断では β の最適値が上がる可能性が高い。CI baseline (libri1 specific) では α=0.9 で低 SNR の TPR がやや下がる (例: SNR 5 dB で 0.46 → 0.32) — 集計平均で α=0.9 が勝つが特定話者の頑健性とのトレードオフがある。本格的な再 calibration は CommonVoice / VCTK + 話者多様性込みで Phase 2 に行う想定。

## D-006: VoiceFilter-Lite を採用しない

### 検討経緯

Google の VoiceFilter-Lite（2020）は軽量・ストリーミング対応で一見魅力的に見えた。

### 却下理由

論文を直接読むと、VoiceFilter-Lite は：
- 入力: log-mel filterbank energies
- 出力: enhanced log-mel filterbank energies

つまり **波形を出力しない**。ASR 前処理専用設計で、通話用途には根本的に使えない。SpeakerBeam-SS 論文（Sato et al., Interspeech 2024）も明示的に指摘している：

> "Since VoiceFilter-Lite enhances filterbank features for ASR, it is not suitable for communication applications."

オリジナル VoiceFilter（2019）は波形出力対応だが、ConVoiFilter はその上位互換であり、自前実装する場合でも ConVoiFilter ベースの方が合理的。

## D-007: WHAM! データセットの扱い

### 状況

- ConVoiFilter は WHAM! ノイズで訓練されている
- WHAM! は CC BY-NC 4.0（非商用）
- ただし本プロジェクトはハードゲーティング型のため、ConVoiFilter は使わない

### 結論

本構成では WHAM! 由来モデルを使用しないため、グレーゾーン問題は発生しない：

- DFN3: DNS Challenge データ（CC BY 4.0）+ 独自データ
- silero-vad: 独自データ
- ECAPA-TDNN: VoxCeleb1+2（公開、非商用利用にやや配慮要）

VoxCeleb のライセンス条項は「メディアが BBC/YouTube から取得されている」点で完全クリーンとは言い難い。本プロジェクトの最終的な商用展開時には、ECAPA-TDNN の代替（CommonVoice 等で訓練したもの）を検討する可能性あり。

## D-008: ECAPA-TDNN を話者埋め込みとして採用

### 検討した代替案

A. d-vector（VoiceFilter 系で標準）
B. x-vector（古典的、ConVoiFilter で採用）
C. ECAPA-TDNN（現代の標準、SpeechBrain で公開）
D. ECAPA2（2024 改良版）
E. WavLM ベースの埋め込み

### 採用: C（ECAPA-TDNN）

理由:
- 性能・効率のバランスが現時点で最良
- SpeechBrain の `spkrec-ecapa-voxceleb` が Apache 2.0 で公開、即動作
- ONNX 変換が容易、モバイル展開対応可能
- D（ECAPA2）は新しいが、公開重みの整備が C より劣る
- E（WavLM）はモデルサイズが大きく、リアルタイム性に懸念

### 将来的な代替検討

- ECAPA2 への移行（性能優位が確認できれば）
- TitaNet（NVIDIA、ただし CC-BY-NC で非商用）
- 自前訓練（CommonVoice ベース、完全クリーンライセンス確保のため）

## D-009: 言語非依存の設計を維持

### 判断

DFN3、silero-vad、ECAPA-TDNN いずれも言語非依存。日本語特化のファインチューンは行わない。

### 理由

- ECAPA-TDNN は VoxCeleb（多言語）で訓練済み、日本語話者でも動作
- 日本語特化の話者埋め込みモデルはオープンソースでは限定的
- 言語非依存設計は使用国を選ばないというメリットがある

### 将来的な再検討タイミング

日本語話者で実機検証して EER が悪化した場合のみ、日本語データでのファインチューンを検討。

## D-010: AS-Norm（Adaptive S-Norm）でスコア正規化を導入

### 背景

PR #17 で導入した Scenario 5 (多言語ロバスト性) の初回 baseline 計測 (real
pipeline, MLS+Emilia-YODAS, 6 言語 × 4 SNR) で、global θ_pass=0.30 では
言語間で score 分布が偏ることが定量的に確認された:

| Lang | TPR | FPR |
|---|---|---|
| de | 0.77 | 0.02 |
| en | 0.78 | 0.00 |
| fr | 0.80 | 0.00 |
| ko | 0.80 | 0.00 |
| **ja** | **0.67** | 0.07 |
| **zh-CN** | 0.86 | **0.23** |

ja は SNR≤5dB で TPR=0.59 に落ち (FN 偏り)、zh-CN は同 SNR 帯で FPR=0.33
に上がる (FP 偏り)。**1 つの global θ_pass では同時最適化不能**。

### 検討した代替案

| 案 | 採否 | 理由 |
|---|---|---|
| A. per-language θ_pass オーバーライド | 不採用 | 言語ごと手動チューニング、保守困難 |
| B. **AS-Norm (Adaptive S-Norm)** | **採用** | 業界標準 (>20年)、推論時オーバーヘッド軽量 (cohort K=30 で +30 cosine sim/call)、global threshold 1 つで済む |
| C. Language-Dependent AS-Norm (Thienpondt 2020) | 将来検討 | LID head 必要、B より複雑度↑ |
| D. TAS-Norm (2025 trainable) | 将来検討 | 学習データ必要、PoC scope を超える |
| E. Discriminative condition-aware backend (Ferrer 2019) | 将来検討 | 大量の calibration data + 学習が必要 |

### 採用: B（AS-Norm）

仕組み:

```
S_norm = (S_target - μ_top-K(S_impostor)) / σ_top-K(S_impostor)
```

- 推論時に target embedding と enrollment との cosine sim を求めるのに加え、
  事前構築した **impostor cohort** (多言語の非ターゲット話者 30-50 名分の
  embedding) との cosine sim も計算
- top-K (K=10 程度) impostor score の平均と標準偏差で z-score 正規化
- 言語/ノイズ/録音条件に依存する score 分布の系統バイアスが消え、
  global θ_pass で複数言語をカバー可能

### 実装フェーズ

1. **Phase 1** ✅ (PR #18): cohort build script
   `scripts/build_impostor_cohort.py` で MLS+Emilia の manifest から
   ECAPA embedding を抽出し `.npz` で出力
2. **Phase 2** ✅ (PR #19): `gating.py` に `as_norm_score` / `load_cohort`
   実装、`GatingConfig` に `use_as_norm` / `as_norm_cohort_path` /
   `as_norm_top_k` / `theta_pass_as_norm` / `theta_learn_as_norm` 追加、
   `pipeline.process_offline` の score 経路を分岐、CI で cohort
   自動 build → scenario_5 に反映
3. **Phase 3** (次 PR): `scripts/calibrate.py` に AS-Norm 拡張、
   per-language sweep で `theta_pass_as_norm` を data 駆動で確定、
   scenario_5 hard-fail 閾値引き締め
4. **Phase 4** (任意): C/D/E への拡張、または cohort 拡大 (per-language
   5 → 10、top-K 10 → 20) — Phase 3 の結果次第で要否判断

### Phase 2 の設計ノート

- AS-Norm 経路では F0 を per-frame gate decision から外し、cohort
  正規化された SV 類似度のみで判定する (F0 は引き続き auto-learn 入口の
  `theta_f0` で使用)。理由: AS-Norm の literature は SV 類似度に直接適用
  するのが標準で、F0 と z-score を加算するとスケール不整合になる。
- `theta_pass_as_norm = 1.5` / `theta_learn_as_norm = 2.5` はヒューリスティック
  初期値。Phase 3 で `scripts/calibrate.py` を AS-Norm 経路で再走して
  data 駆動で確定する。
- `use_as_norm = False` を default に維持し、既存 PoC + bench テストが
  bit-identical に通ることを担保 (`enable_auto_learn` と同じパターン)。

### Phase 2 実観測 (PR #19 初回 CI run, real pipeline, MLS+Emilia-YODAS, 6 言語)

PR #17 の legacy `α·cs + β·f0` ベースラインと、PR #19 の AS-Norm (default
`theta_pass_as_norm=1.5`、cohort 30 embeddings × 6 言語) を per-language
で比較した結果:

| Lang | TPR (legacy) | TPR (AS-Norm) | Δ TPR | FPR (legacy) | FPR (AS-Norm) | Δ FPR |
|---|---|---|---|---|---|---|
| de | 0.77 | 0.69 | **−0.08** | 0.02 | 0.04 | +0.02 |
| en | 0.78 | 0.80 | +0.02 | 0.00 | 0.00 | 0 |
| fr | 0.80 | 0.83 | +0.03 | 0.00 | 0.03 | +0.03 |
| ja | 0.67 | **0.85** | **+0.18** ✅ | 0.07 | 0.12 | +0.05 |
| ko | 0.80 | 0.86 | +0.06 | 0.00 | 0.00 | 0 |
| zh-CN | 0.86 | 0.87 | +0.01 | 0.23 | **0.42** | **+0.19** ❌ |
| **mean** | 0.78 | **0.82** | **+0.04** | 0.05 | 0.10 | +0.05 |
| **stddev** | 0.058 | 0.060 | +0.002 | 0.084 | 0.148 | +0.064 |

**評価**:

- **大成功**: ja TPR が 0.67 → 0.85 (+18pp)。低 SNR (0/5dB) で 0.59 → 0.85
  に改善し、本決定の主目的だった日本語取りこぼし問題が解消。en/fr/ko も
  微改善で aggregate TPR は +4pp。
- **新規 regression (2 件)**:
  - **zh-CN FPR**: 0.23 → 0.42 (+19pp)。SNR=0 で 0.54 まで上昇。tonal 言語の
    impostor 区別が cohort 30 embeddings (per-language 5) では薄く、top-K=10
    が cohort 全体の 33% を占めるため normalization が弱い可能性。
  - **de TPR**: 0.77 → 0.69 (−8pp)。SNR=0 で 0.48 まで落ちる新規 regression。
    AS-Norm が de の cohort 分布バイアスを意図せず作っている可能性。
- aggregate FPR は 0.05 → 0.10 (+5pp 悪化、zh-CN 起因)、cross-lang stddev も
  0.084 → 0.148 と拡大。

**Phase 3 への示唆**:

1. `theta_pass_as_norm = 1.5` のヒューリスティック値が言語ごと過剰/不足。
   `calibrate.py` の AS-Norm 拡張で sweep してグローバル最適値を求めるのが
   主軸。
2. 上記で吸収できない場合のみ Phase 4 として cohort 拡大 (per-language
   5 → 10) や top-K 引き上げを検討する。事前に決め打ちしない。
3. 最終的に `theta_pass_as_norm` を更新した時点で scenario_5 hard-fail
   閾値も引き締める (現在 `--tpr-min 0.3 --fpr-max 0.7` は legacy 観測値
   からの 27pp バッファ — Phase 3 完了後に縮められる見込み)。

### 参考文献

- Thienpondt et al. (2020) "Cross-Lingual Speaker Verification with
  Domain-Balanced Hard Prototype Mining and Language-Dependent Score
  Normalization", https://arxiv.org/abs/2007.07689
- Park et al. (2025) "Trainable Adaptive Score Normalization for
  Automatic Speaker Verification", https://arxiv.org/abs/2504.04512
- Ferrer et al. (2019) "A Discriminative Condition-Aware Backend for
  Speaker Verification", https://arxiv.org/abs/1911.11622
- Klusáček et al. (2025) "On the influence of language similarity in
  non-target speaker verification trials",
  https://arxiv.org/abs/2506.02777
