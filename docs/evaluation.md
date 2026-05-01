# Evaluation

本ドキュメントは、[benchmarks.md](benchmarks.md) で定めたデータセット・シナリオ・指標を実際に**どう運用し、どう判定し、どう記録するか**を定める。

## 位置づけ

| ドキュメント | 役割 |
|---|---|
| benchmarks.md | 何を使って何を測るか（データセット・シナリオ・指標の選定） |
| **evaluation.md（本文書）** | どう実施・判定・記録するか（プロトコル・合否基準・結果管理） |

## 評価プロトコル

### 全体ワークフロー

```
[1] 評価環境セットアップ
    \u2502
    \u25bc
[2] データセット準備（ダウンロード・前処理・ミキシング）
    \u2502
    \u25bc
[3] モデル登録（明示登録音声からの埋め込み生成）
    \u2502
    \u25bc
[4] 各シナリオ実行（バッチで全件処理）
    \u2502
    \u25bc
[5] 指標計算（PESQ, STOI, EER 等）
    \u2502
    \u25bc
[6] CSV / JSON 出力
    \u2502
    \u25bc
[7] レポート生成（マークダウン + プロット）
    \u2502
    \u25bc
[8] 合否判定（Phase ゲート条件と照合）
```

### シナリオ別実施手順

#### Scenario 1: Solo Target + Noise

```
INPUT:
  target_speaker_audio (clean) + noise (MUSAN / DEMAND, SNR ∈ {-5, 0, 5, 10, 15, 20} dB)
  enrollment_audio (target speaker, 30s)

EXECUTE:
  1. enrollment から埋め込みプール生成
  2. 各 SNR 条件で混合音声生成
  3. パイプライン通す（DFN3 → 判定 → 出力）
  4. 出力音声を保存

MEASURE:
  - PESQ, STOI, SI-SDR（出力 vs ground truth target）
  - DNSMOS P.835（出力単体）
  - True Positive Rate（フレーム単位、speech フレームでの pass 率）
  - 出力音声の RMS（mute されていないことの確認）
```

#### Scenario 2: Solo Other Speaker + Noise

```
INPUT:
  other_speaker_audio (target ではない話者) + noise
  enrollment_audio (target speaker, 別人の音声)

EXECUTE:
  1. パイプライン通す
  2. 出力音声を保存

MEASURE:
  - True Negative Rate（speech フレームでの mute 率）
  - False Positive Rate（誤って pass する率）
  - 出力 RMS（十分減衰しているか）
  - SI-SDR（出力 vs zero）→ 大きいほど良い（よく mute されている）
```

#### Scenario 3: Alternating Speech

```
INPUT:
  concatenated audio: target → silence → other → silence → target → silence → other ...
  各セグメント 3-5 秒、計 30-60 秒
  silence セグメント 0.5-2 秒
  enrollment_audio (target)

EXECUTE:
  1. パイプライン通す
  2. フレームレベルのゲート判定を記録
  3. ground truth ラベル（各フレームが target/other/silence）と照合

MEASURE:
  - Frame-level accuracy
  - Confusion matrix（target/other/silence × pass/mute）
  - Onset latency（target 開始から pass 確立までの ms）
  - Offset latency（other 開始から mute 確立までの ms）
  - attack/release 時定数の実測（理論値 attack=15ms, release=100ms）
```

#### Scenario 4: Simultaneous Speech

```
INPUT:
  target_speaker_audio + other_speaker_audio（同時発話）
  enrollment_audio (target)
  ミキシング比: target:other ∈ {0:1, 1:3, 1:1, 3:1, 1:0}

EXECUTE:
  1. 各ミキシング比で混合
  2. パイプライン通す
  3. 出力保存

MEASURE:
  - SI-SDR（出力 vs target only ground truth）
  - 主観評価（対象話者の聞き取りやすさ、5段階）
  - target:other 比による pass 判定の閾値（目的話者成分が何 dB 以上で pass か）
  - FP 許容方針が機能しているか（目的話者成分があれば pass か）
```

#### Scenario 5: Multilingual Robustness

```
INPUT:
  各言語（en, ja, de, fr, zh, es）の対象話者音声 + noise
  各言語につき: 50 utterances × 異なる話者
  enrollment: 各言語の話者ごとに 30s 登録

EXECUTE:
  1. 言語ごとに enrollment 実行
  2. 同じパイプラインで Scenario 1 と同等の評価

MEASURE:
  - 言語別 EER
  - 言語別 gating accuracy
  - 言語間ばらつき（標準偏差）
  - DNSMOS の言語間ばらつき
```

#### Scenario 6: Drift Verification

```
INPUT:
  対象話者の長時間音声（30-60 分）
  音声には経時変化を模擬（風邪・疲労・感情の異なる発話セグメント）
  enrollment: 最初の 30 秒のみ
  自動学習: ON

EXECUTE:
  1. 最初の 30 秒で明示登録
  2. 以降の音声をリアルタイム模擬で処理
  3. 自動学習プールへの追加履歴を記録
  4. anchor_distance を時系列で記録

MEASURE:
  - 時間経過での gating accuracy 推移
  - auto_learn_pool size の変動
  - anchor_distance の中央値推移
  - drift 検出のトリガー回数
  - リセット発動の有無
```

### 入出力フォーマット

#### 入力データの命名規則

```
data/
├── target_speakers/
│   └── {speaker_id}/
│       ├── enrollment.wav           # 30s 以上のクリーン録音
│       └── test/
│           ├── utt_001.wav
│           ├── utt_002.wav
│           └── ...
├── other_speakers/                  # 同様の構造
├── noise/
│   ├── musan/
│   │   ├── speech/
│   │   ├── music/
│   │   └── noise/
│   └── demand/
│       ├── kitchen/
│       ├── office/
│       └── ...
└── mixtures/                        # シナリオ別生成
    ├── scenario_1/
    ├── scenario_2/
    └── ...
```

#### 出力 CSV フォーマット（共通）

```csv
sample_id,scenario,language,snr_db,target_speaker,other_speaker,
gate_tpr,gate_tnr,gate_fpr,gate_fnr,
pesq,stoi,si_sdr,dnsmos_sig,dnsmos_bak,dnsmos_ovrl,
attack_ms,release_ms,
processing_time_ms,
notes
```

#### 出力 JSON サマリ

```json
{
  "evaluation_id": "eval_20260501_153045",
  "git_commit": "abc1234",
  "model_versions": {
    "dfn3": "0.5.6",
    "silero_vad": "5.1",
    "ecapa_tdnn": "speechbrain-1.0.0"
  },
  "system_info": {
    "platform": "Linux x86_64",
    "cpu": "AMD Ryzen 9 5950X",
    "ram_gb": 64,
    "python_version": "3.11.5"
  },
  "thresholds": {
    "theta_pass": 0.50,
    "theta_learn": 0.80,
    "alpha": 0.8,
    "beta": 0.2
  },
  "scenarios": {
    "scenario_1": { "n_samples": 100, "tpr_mean": 0.94, ... },
    "scenario_2": { "n_samples": 100, "tnr_mean": 0.92, ... },
    ...
  },
  "phase_gate_status": {
    "phase_1_eligible": true,
    "phase_2_eligible": false,
    "blocking_criteria": ["scenario_3_frame_accuracy"]
  }
}
```

## ハードゲーティング型の評価観点

ハードゲーティング型固有の論点として、汎用 SE/TSE 評価指標だけでは捉えきれない側面がある。

### TPR と FPR のバランス

FP 許容方針を採用しているため：

- **TPR 最大化が最優先**: 対象話者を切らない
- **FPR は二次評価**: ある程度の他話者漏れは許容

評価時は混同行列を必ず保持し、TPR/FPR の比をモニタリング。「TPR が高くても FPR がほぼ 1.0」のような状況（事実上ゲートが機能していない）を検出する。

### attack/release 挙動の検証

理論値（attack=15ms, release=100ms）からの乖離を測定：

- 急峻な遷移はクリック音を生む → attack < 5ms は警告
- 過度に遅い release は他話者を漏らす → release > 300ms は警告
- 周波数応答の評価: 矩形パルスを入力し、応答の立ち上がり/立ち下がり波形を確認

### 短時間発話での SV 安定性

「うん」「はい」等の 200-500ms 発話で SV 判定が不安定になる現象を測定：

- 各発話長（200ms, 500ms, 1s, 2s, 5s）での EER を分離記録
- 発話長が短いほど EER が悪化する想定だが、許容範囲を確認

### 同時発話の挙動

Scenario 4 で評価する目的話者の聞き取りやすさは、客観指標では測りにくい。SI-SDR は **target only ground truth に対する歪み**を見るので、他話者漏れがある状態でも SI-SDR 自体は計算可能だが、以下の追加情報を併記する：

- 対象話者音声の RMS 比率（出力中の対象話者由来音声 vs 他話者由来音声）
- スペクトル重複度（対象話者と他話者の周波数帯域の重なり）
- 主観評価（聞き取りやすさ）

### Drift の長期挙動

Scenario 6 では、以下の長期メトリクスを追う：

- **anchor_distance の時間推移**: 単調増加していたら drift 警告
- **auto_learn_pool の入れ替わり頻度**: FIFO で古い埋め込みが捨てられる頻度
- **リセット発動回数**: 多発する場合は drift 対策の閾値見直しが必要

## 想定される失敗モード

評価実施前に「何が壊れうるか」を列挙し、各失敗モードを意図的にテストする。

| Failure Mode | 想定シナリオ | 検証方法 |
|---|---|---|
| 高ノイズ環境での SV 判定低下 | SNR < 0 dB | Scenario 1 の SNR スイープで確認 |
| 残響条件での SV 判定低下 | 大ホール、浴室 | RIR 畳み込みでテスト |
| 似た声紋の他話者を pass | 同性家族・兄弟 | 既知の似話者ペアで Scenario 2 |
| 短い相槌で誤判定 | 「うん」「はい」 | 短発話 EER 評価 |
| 言語横断での性能差 | 英語以外 | Scenario 5 で確認 |
| 自動学習プールの drift | 長時間運用 | Scenario 6 で確認 |
| 登録音声の品質依存 | 低 SNR / 録音品質劣悪 | enrollment SNR スイープ |
| マイク特性の違い | 異なるマイク | 推論時のマイク変更テスト |
| 話者の体調変化 | 風邪、声枯れ | 経時変化模擬データ |

各失敗モードについて：

1. 想定される影響範囲（どの指標がどの程度劣化するか）を事前予測
2. 実測で予測との乖離を確認
3. 大きな乖離は失敗モード再評価のトリガー

## ベースライン比較と解釈方針

### 比較対象

| 比較対象 | 役割 | 期待される結果 |
|---|---|---|
| 何も処理しない原音声 | 下限 | 最悪値、これより悪化していたら根本問題 |
| DFN3 単体 | NS 効果の純粋測定 | NS 部分は同等。SV 機能の有無で TPR/FPR が異なる |
| オラクル VAD（ground truth） | 上限 | TPR=1, TNR=1。実装の上限値 |
| ConVoiFilter（オフライン） | 真の TSE のリファレンス | 同時発話シーンで優位、ただし 5 秒遅延 |
| ESPnet TD-SpeakerBeam | causal 寄り TSE | 性能と遅延のトレードオフ参照 |

### 解釈の指針

| 観測 | 解釈 |
|---|---|
| ハードゲーティング ≈ DFN3 単体 + 完璧な VAD | 想定通り、SV 部分が良好に機能 |
| ハードゲーティング < DFN3 単体 | SV 誤判定で対象話者を切っている、threshold 緩和を検討 |
| ハードゲーティング ≈ ConVoiFilter（順番発話シーン） | ハードゲーティング型として最良 |
| ハードゲーティング << ConVoiFilter（同時発話シーン） | 想定通り、ハードゲーティング型の限界 |

### 「真の TSE 比較」の罠

ConVoiFilter 等は同時発話シーンで優位だが、以下を区別する：

- **同時発話のみのデータセット**: ConVoiFilter が大幅優位
- **実通話シナリオ（順番発話が大半）**: ハードゲーティング型と同等以上

実通話の挙動を反映するシナリオで比較しないと、ハードゲーティング型が不当に低評価される。

## 主観評価プロトコル

### 試聴環境の標準化

- ヘッドホン: 密閉型、品質指定なしだが評価者間で同一
- 試聴音量: -23 LUFS で正規化
- 環境ノイズ: 静音環境（< 30 dB SPL）

### 評価者

PoC 段階：
- 開発者本人
- 家族 1-2 名
- 同僚（可能なら）1-2 名

本実装段階：
- 第三者 5-10 名による A/B test

### 評価項目

A/B test 形式：

```
評価対象: 元音声 vs 処理後音声、ハードゲーティング vs DFN3 単体

質問項目:
  Q1. 対象話者の声は明瞭ですか？  (1: 全く / 5: 非常に)
  Q2. 他の話者の声は気になりますか? (1: 非常に / 5: 全く気にならない)
  Q3. 対象話者の声に違和感はありますか? (1: 強い / 5: 全くない)
  Q4. ノイズはどの程度残っていますか? (1: 多い / 5: 全くない)
  Q5. 通話相手として聞きやすいですか? (1: 聞きづらい / 5: 聞きやすい)
```

各サンプル 5-10 ペアを評価、結果を MOS として平均。

### MOS 収集の自動化

シンプルな Web UI または Jupyter Notebook で実装：

- ランダム順で A/B サンプルを再生
- 評価者は数値入力
- 結果は CSV に追記
- ブラインド評価（A/B どちらがどれか伏せる）

## 合否基準（Phase ゲート条件）

各 Phase の次フェーズへ進むためのゲート条件。**初期値は仮であり、Phase 1 の実測で調整する**。

### Phase 1 → Phase 2 のゲート

最小限のパイプラインが機能していることの確認：

| 指標 | 初期目標値 | 備考 |
|---|---|---|
| Scenario 1 TPR | > 0.85 | 対象話者を 85% 以上 pass |
| Scenario 2 TNR | > 0.80 | 他話者を 80% 以上 mute |
| Scenario 1 PESQ improvement | > 0.3 | 元音声に対して PESQ +0.3 以上 |
| Scenario 5 言語間 EER 標準偏差 | < 0.10 | 言語間ばらつき 10% 以内 |
| End-to-end latency（PoC 計測） | < 300ms | Python 実装、最終目標 100ms |

### Phase 2 → Phase 3 のゲート

機能完全性の確認：

| 指標 | 初期目標値 |
|---|---|
| Scenario 3 Frame accuracy | > 0.90 |
| Scenario 6 drift 検出機能 | リセット発動回数 < 5 回 / 60 分 |
| Anchor 保護機能 | 自動学習で anchor が削除されていないこと（保証） |
| F0 補助の効果 | F0 ありで EER が +0% 以上（劣化なし） |

### Phase 3 → Phase 4 のゲート

性能要件達成：

| 指標 | 初期目標値 |
|---|---|
| End-to-end latency（Rust 実測） | < 100ms |
| CPU 使用率（単一スレッド） | < 30%（M1/Ryzen 5 相当） |
| Scenario 1 TPR | > 0.92 |
| Scenario 2 TNR | > 0.90 |
| メモリフットプリント | < 100 MB |

### Phase 4（モバイル）のゲート

モバイル展開可能性の確認：

| 指標 | 初期目標値 |
|---|---|
| iOS / Android での連続動作 | 30 分以上クラッシュなし |
| バッテリー消費 | 通話相当の電力プロファイル内に収まる |
| バイナリサイズ | < 30 MB |
| 起動時遅延 | < 2 秒 |

各 Phase の終了時にゲート条件と照合し、未達項目があれば該当機能の改善または閾値見直しを行う。閾値見直しの判断は、対応する Phase の総合的な達成状況とトレードオフを踏まえて検討。

## 結果記録の継続管理

### バージョン管理

各評価実行時に以下を記録：

- Git commit hash（実装側）
- モデルバージョン（DFN3, silero-vad, ECAPA-TDNN）
- 評価データセットのバージョン（Common Voice なら v18 等）
- 評価実施日時
- システム情報（CPU, RAM, OS）
- 閾値・パラメータ設定

### 履歴ディレクトリ構造

```
benchmark_results/
├── 20260501_153045_phase1_initial/
│   ├── summary.json
│   ├── scenario_1.csv
│   ├── ...
│   └── plots/
├── 20260508_104530_phase1_threshold_tuning/
│   ├── summary.json
│   └── ...
└── latest -> 20260508_104530_phase1_threshold_tuning  (symlink)
```

### 回帰検出

PoC 実装後、機能追加・修正のたびに評価を再実行し、過去結果と比較：

- **regression**: 主要指標が前回比 -3% 以上劣化 → 修正必須
- **neutral**: ±3% 以内 → 許容
- **improvement**: +3% 以上 → 記録

差分は自動でレポート化（next/prev 比較表）。

### 再実行のトリガー

以下の変更があった場合、評価を再実行する：

- 任意のモデルのバージョン更新
- 閾値・パラメータの変更
- パイプライン構造の変更
- 依存ライブラリ（ONNX Runtime 等）のメジャー更新
- 評価データセットのバージョン更新

## レポーティング

### Phase 完了時のレポート構造

各 Phase 完了時に `benchmark_results/<eval_id>/REPORT.md` を生成：

```markdown
# Phase N Evaluation Report

## Summary
- Eval ID: <eval_id>
- Date: YYYY-MM-DD HH:MM
- Git commit: <hash>
- Phase gate status: PASS / FAIL

## Highlights
- <主要結果のサマリ 3-5 行>

## Scenario Results
### Scenario 1: Solo Target + Noise
| SNR | TPR | TNR | PESQ improvement | DNSMOS OVRL |
|...|...|...|...|...|

### Scenario 2: ...
...

## Phase Gate Check
| Criterion | Target | Actual | Status |
|...|...|...|...|

## Comparison vs Baselines
...

## Failure Mode Analysis
...

## Next Steps
- <次 Phase に向けた改善項目>
```

### 関係者への共有

PoC 段階：開発者本人のみ。レポートは Git に追加、リポジトリ内で管理。

将来的な共有（チーム化した場合）：
- マークダウンレポートを GitHub Issue / Discussion で共有
- 主要指標の推移を時系列でグラフ化
- 重大な regression は即時アラート

## ベンチマーク自動化スクリプト構成

[benchmarks.md](benchmarks.md) で示した `bench/` 構造に対応する評価スクリプト：

```python
# bench/runners/run_all.py
def run_evaluation(
    config_path: str,
    output_dir: str,
    scenarios: list[str] = None,  # None なら全シナリオ
    quick: bool = False,           # ミニマル評価セットのみ
) -> EvaluationResult:
    """
    全評価を実施し、結果を output_dir に保存
    Returns: EvaluationResult（summary.json と同等の構造）
    """
    ...
```

CLI 想定：

```bash
# クイック評価（< 1 時間）
python bench/runners/run_all.py --quick --output benchmark_results/eval_$(date +%Y%m%d_%H%M%S)

# フル評価
python bench/runners/run_all.py --output benchmark_results/...

# 特定シナリオのみ
python bench/runners/run_all.py --scenarios scenario_1,scenario_5
```

## CI 統合（将来）

PoC が安定したら、評価の一部を CI で自動実行：

- Pull Request 時: クイック評価（10 分以内、Scenario 1 のミニマルセット）
- main へのマージ時: 標準評価（1 時間以内、ミニマル評価セット全件）
- 週次: フル評価（実機 Phase 4 含む）

CI で regression を検出したら自動で Issue 起票。実装段階で詳細を詰める。
