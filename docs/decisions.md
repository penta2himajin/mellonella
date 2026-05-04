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
3. **Phase 3** ✅ (PR #24 + 後段 PR): `scripts/calibrate.py` に AS-Norm 拡張、
   per-language sweep で `theta_pass_as_norm` を data 駆動で確定、
   CI 観測値で baseline を文書化。`scenario_5.yml --fpr-max` の引き締めは
   後述の理由により **見送り** (cohort 規模に起因する run-to-run variance が
   CI hard-fail を不安定にするため)。
4. **Phase 4** ✅ (PR #22 + #23): cohort-disjoint fix + actions/cache 化
5. **Phase 5** ✅ (本 PR): cohort 決定化 — `mls.prepare` / `emilia.prepare`
   の streaming-arrival 依存ラベル割当を撤廃し、speaker は upstream id
   lex 順、clip は audio sha1 順で選択。同じ split に対して常に
   bit-identical な manifest を出力し、Phase 4 cache の "miss 時に別 cohort"
   弱点を塞ぐ。
6. **Phase 6** (任意): cohort 拡大 (per-language 8 → 50-100、top-K 10 → 20-30)、
   `--fpr-max` 引き締め、別 scenario への AS-Norm 横展開 — 規模拡大が
   実現してから再開

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

### Phase 2 後の再観測 (PR #21 cohort 診断 + 構造的バグの発見)

PR #21 で cohort summary を artifact 化した後、PR #19 と PR #21 の cohort
を比較した結果、**variance はランダム noise ではなく構造的なバグ起因**
だったことが判明:

1. **cohort が test 話者を含んでいた** (致命的 algorithm 違反):
   `scenario_5_from_manifest.py` は同じ manifest から target / other を選択し、
   cohort も同じ manifest から構築されていた。各 manifest は 3 話者しか
   含んでいなかったため、cohort = {speaker01, 02, 03}、test = 2 of those。
   → AS-Norm の "impostor cohort" のはずが target/other 自身を含んでおり、
   z-score 正規化の前提が崩れていた。
2. **cohort が想定の半分** (18 vs 30): `mls.prepare` / `emilia.prepare` の
   default `top_speakers=3` で manifests に 3 話者しか入っていなかった。
   `--per-language 5` を渡しても 3 しか取れない。top-K=10 / 18 = 56% で
   literature の上限 (10-30%) を大きく超過。
3. **同名の "speaker01" が run 間で異なる upstream 話者だった**: prepare
   段階で「streaming で見つかった順」にラベルを振っていたため、HF datasets
   の並行 IO 由来の順序揺れがそのまま cohort 内訳に伝播していた。

### Phase 4: cohort-disjoint fix ✅ (PR #22) + cohort cache stability ✅ (PR #23)

Phase 3 (calibrate.py 拡張) の前に、**まず構造的バグを潰す必要がある** ため
Phase 4 として優先実施した:

- `mls.prepare` / `emilia.prepare` の default `top_speakers` を 3 → 10 に
  引き上げ。各 manifest に 10 話者用意。
- `scripts/build_impostor_cohort.py` に `--skip-top-n N` 追加。
  scenario_5 が test に使う rank を cohort から carve out。
- `.github/workflows/scenario_5.yml` で `--skip-top-n 2 --per-language 8` を
  渡す。結果: 各言語 8 cohort 話者 = 48 embeddings、top-K=10 = 21%
  (literature 範囲)、test と完全分離。

PR #22 マージ後、disjoint な cohort で 2 回連続 CI を回したところ
zh-CN FPR が 0.76 → 0.85 と run-to-run で揺れた。HF datasets streaming の
非決定的順序で manifest 自体が再生成されるたびに変わるのが原因。**Phase 4
追加対応** として cohort を `actions/cache@v4` で **永続化 + skip-if-exists
guard** を追加 (PR #23):

- cache key に `scripts/build_impostor_cohort.py` のハッシュを追加 →
  selection ロジックが変わったら自動 invalidate。
- cache key を v1 → v2 に bump して既存の broken-cohort cache を強制廃棄。
- workflow の cohort build step に "if exists, skip" guard。1 度生成された
  `.npz` は cache hit が続く限りそのまま使い回される → 完全決定的。
- 新言語追加 / `top_speakers` 変更などで再生成したい場合は cache key を
  bump (v2 → v3) すれば良い。repo に commit する必要なし (~38 KB だが
  毎回 git に乗せるよりキャッシュの方が運用が軽い)。

これで Phase 3 の calibrate.py 拡張に進める前提が整った。Phase 4 の修正前に
キャリブレーションしても "壊れた cohort" の上で fitting するだけになる。

### Phase 3: calibrate.py AS-Norm 拡張 (PR #24)

PR #23 で cohort が CI cache 経由で完全決定的になった後、`scripts/calibrate.py`
に AS-Norm 経路の sweep を追加した:

- 新 CLI フラグ `--use-as-norm --cohort PATH`。両方必須。
- 新 sweep 範囲 `THETA_GRID_AS_NORM = (0.5, 0.75, ..., 3.0)`。z-score scale
  に合わせて 11 段階。legacy の `THETA_GRID` (0.20-0.55、cosine scale) は維持。
- 出力ファイルが mode で分岐:
  - legacy → `docs/benchmarks/calibration_{results.csv,summary.json}`
  - as_norm → `docs/benchmarks/calibration_as_norm_{results.csv,summary.json}`
- `recommend_theta()` を引数化 (`max_mean_fpr` / `min_tpr_floor`)。AS-Norm 用
  default は `MAX_MEAN_FPR_AS_NORM = 0.10` (PR #23 の per-language FPR spread
  が広いため legacy の 0.05 を緩めた)。
- CSV/summary に `mode` 列を追加。schema_version 1 → 2。
- ライトユニットテスト (10 件) を `bench/tests/test_scripts_calibrate.py` に追加:
  theta grid 範囲、`_simulate_gate` の AS-Norm/legacy 分岐、`recommend_theta`
  の各 fallback、`--use-as-norm` で `--cohort` 必須のチェック。

実行手順 (user 手元):

```bash
# 1. cohort を build (まだなら)
python scripts/build_impostor_cohort.py \
    --manifest en=$MELLONELLA_DATA_DIR/emilia_yodas/en/manifest.csv \
    --manifest ja=$MELLONELLA_DATA_DIR/emilia_yodas/ja/manifest.csv \
    --manifest de=$MELLONELLA_DATA_DIR/mls/de/manifest.csv \
    --skip-top-n 2 --per-language 8 \
    --output bench/data/cohorts/scenario5_cohort_v1.npz

# 2. AS-Norm calibration sweep (per language で繰り返し or 連結 manifest)
python scripts/calibrate.py \
    --use-as-norm \
    --cohort bench/data/cohorts/scenario5_cohort_v1.npz \
    --manifest ja=$MELLONELLA_DATA_DIR/emilia_yodas/ja/subset/manifest.csv \
    --language ja
```

結果 (`calibration_as_norm_summary.json`) の `recommended_theta_pass` を
`GatingConfig.theta_pass_as_norm` のデフォルトに反映する PR を別立てで
出す (Phase 3 後段)。その後 `scenario_5.yml` の `--fpr-max` を引き締め予定
だったが、後述のとおり **CI 観測 variance により今回は見送り**。

### Phase 3 後段: CI baseline 観測と threshold 据え置き判断

PR #24 マージ後、cohort-disjoint + cache-frozen な状態 (PR #22 + #23) で
複数回 scenario_5 を回し、`theta_pass_as_norm = 1.5` のままどの程度安定
するかを観察した:

| Run | TPR mean | FPR mean | zh-CN FPR mean | zh-CN per-row max |
|---|---|---|---|---|
| PR #23 直後 | 0.79 | 0.15 | 0.31 | 0.62 |
| PR #24 マージ直後 | 0.77 | 0.13 | 0.31 | ~0.6 |
| 続き run 1 | 0.74 | 0.11 | 0.34 | 0.71 |
| 続き run 2 | 0.77 | 0.10 | 0.31 | ~0.6 |

aggregate 値 (TPR mean ~0.77、FPR mean ~0.12) は run 間で ±2-3pp 程度に
収束しているが、**zh-CN per-row max は 0.6-0.85 で大きく揺れる**。原因は
Phase 4 で議論した HF datasets streaming の非決定性が cache miss のたびに
再発し、cohort 構成 (どの 8 話者が ranks 2-9 に入るか) が変わるため、
AS-Norm の μ/σ がずれて zh-CN の特定 row のみ突き抜ける。

**判断**: `--fpr-max` を観測 baseline に合わせて引き締める (例 0.95 → 0.4)
と PR #25 で実証されたとおり 1-2 run に 1 回 hard-fail し、CI が
"AS-Norm の真の regression 検出" ではなく "cohort cache の世代差による
ノイズ" を拾うようになる。Phase 3 完了の意味付けを **「閾値引き締め」**
ではなく **「閾値の data 駆動候補値の特定 + CI 観測値の文書化」** に
変更する:

- `theta_pass_as_norm = 1.5` は CI baseline (PR #23-25 で TPR mean 0.77、
  FPR mean 0.13 前後) を達成する spec として確定。
- `scenario_5.yml --fpr-max 0.95` は据え置き。catastrophic regression
  (例: cohort が壊れて FPR > 0.9) を catch する safety net としては
  機能するが、この緩さは **conscious choice** で、Phase 4 で cohort 規模を
  拡大して variance を抑えるまでは tightening しない。
- 真の data 駆動 calibration (Phase 3 当初目標) は cohort 規模が
  literature 推奨の 50-100 spk/lang に達してから再走する。現在の
  per-language 8 spk = 48 cohort embeddings は足元の variance を切るには
  小さすぎる。

### Phase 3 補足: ローカル sweep による mechanism 検証

ユーザー手元 (mvenv: torch 2.4.1+cpu, speechbrain 1.1.0, DeepFilterNet
0.5.6) で `scripts/calibrate.py --use-as-norm --cohort cohort_v1.npz` を
108 cells × 11 θ で 8.5 分かけて流し、`_simulate_gate` の AS-Norm 経路、
`recommend_theta` の AS-Norm 専用 budget (0.10)、CSV/summary の
`mode=as_norm` 出力を end-to-end で確認した。**ただしこの sweep の
recommended θ (= 3.0) は production 値として採用しない**:

- ローカル cohort = MLS de + fr (2 言語、16 話者)、test = librosa libri 英語。
- cohort 言語と test 言語が disjoint な構図で、AS-Norm の μ がそもそも
  低めに出るため全 θ で FPR ≫ 0.10 になり、`recommend_theta` の
  fallback 経路 (= 最厳の θ) で 3.0 が選ばれている。
- production の cohort は 6 言語 48 話者、test との overlap も含めた
  実分布なので意味が違う。

ローカル sweep の役割は **「コードが落ちずに end-to-end 動く」** の確認に
留め、production 値の決定は Phase 3 後段 (上記表) で行った。

### Phase 5: cohort 決定化 (cohort-determinism fix)

Phase 3 後段で観測した zh-CN per-row FPR の 0.6-0.85 揺れは、Phase 4 で
`actions/cache` を導入してもなお **cache miss のたびに manifest が異なる
upstream 話者で再生成される** ことが根本原因だった。これは
`mls.prepare` / `emilia.prepare` の構造的バグ:

1. HF datasets streaming は同じ split に対して同じ sample 集合を返すが
   **順序は保証されない** (parallel IO、retry、shard interleave)。
2. 旧実装は「streaming で出会った順に `speaker01..N` ラベル」「同 speaker
   から最初に来た K clips」「top-N 揃ったら early-break」だったため、
   順序揺れがそのまま manifest 内訳の揺れになっていた。

**修正** (`bench/mellonella_bench/datasets/mls.py` /
`bench/mellonella_bench/datasets/emilia.py`):

- early-break 撤廃。streaming window (`max_stream=5000`) を最後まで scan。
- per-speaker bucket cap を `clips_per_speaker × 4` に引き上げ、後段で選別。
- 後処理で deterministic に選択:
  - speaker 選択: `(clips_count desc, speaker_id lex asc)` の sort で top-N。
    数 tied でも lex tiebreak で順序確定。
  - ラベル割当: 選択集合を **upstream speaker_id 昇順** に並べ替え、
    `speaker01..N` を順に振る。同じ upstream 話者は常に同じスロット。
  - clip 選択: `(-len(audio), sha1(audio.tobytes()))` で sort し先頭 K を
    採用。長い clip = ECAPA に渡す concat が情報量豊かで TPR 安定化に
    寄与。tiebreak は content-hash で arrival 順非依存。**初版 (sha1
    のみ) では Emilia-YODAS の 1-2 秒 snippet を引いて ko/fr で per-row
    TPR が 0.3 を切る事例が出たため length-first に修正。**
- `scripts/build_impostor_cohort.py` の `select_speakers_for_language` も
  `(-audio_size, speaker_id)` の lex tiebreak を追加。同 size 話者間の
  選択が dict-iteration 順に依存していた残りの leak を塞ぐ。

**契約テスト** (`bench/tests/test_datasets_{mls,emilia}.py` に
`test_prepare_is_deterministic_under_streaming_reorder` を追加):

- 同じ fixture を `random.Random(seed)` で 2 種の異なる順序に shuffle し、
  prepare を 2 回実行
- manifest.csv を bytes 比較、各 wav を `filecmp.cmp(shallow=False)` で
  binary 比較。すべて bit-identical を要求。

加えて build_impostor_cohort 側にも
`test_select_speakers_uses_lex_tiebreak_for_equal_audio_lengths` を追加し、
audio 長が tied な 4 話者から lex 上位 2 が選ばれることを assert。

**cache 影響**: 旧 manifest は CSV としては valid だが speaker01..N が
別の upstream 話者を指しているため、cache hit でそのまま使うと AS-Norm の
μ/σ が無音で壊れる。`scenario_5.yml` の cache key を v2 → v3 に bump し、
旧 cache を破棄して新 prep で再生成する。

**期待する効果**:

- HF streaming の順序揺れに対して manifest が完全に invariant になり、
  cache miss → 再 prep → 再 cohort build の chain が full deterministic に
  なる。Phase 4 で `actions/cache` 導入時に残っていた "cache miss = 別
  cohort" の弱点が消える。
- これで cohort 規模拡大 (Phase 4 当初目標) や `--fpr-max` 引き締め
  (Phase 3 後段で見送り) を、再現可能な baseline の上で再開できる。

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
