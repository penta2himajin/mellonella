# Architecture

## 処理パイプライン全体

```
入力 (任意 SR, mono)
  │
  ▼
[Stage 0] Resampler 48 kHz                        ── 入力 SR を統一
  │
  ▼
[Stage 1] DeepFilterNet 3 (48 kHz NS)             ── ノイズ抑制
  │
  ├──→ resample 16 kHz ──┐
  │                       │
  │                       ▼
  │                  [Stage 2] silero-vad         ── 発話/非発話判定
  │                       │
  │                       ▼
  │                  [Stage 3] 動的チャンク蓄積   ── 発話フレームのみ蓄積
  │                       │
  │                       ▼
  │                  [Stage 4]
  │                  ├─ ECAPA-TDNN: 話者埋め込み
  │                  └─ F0 抽出 (補助)
  │                       │
  │                       ▼
  │                  [Stage 5] 統合判定           ── ゲート on/off
  │                       │
  │                       ▼
  │                  [Stage 6] 自動学習プール更新 (条件付き)
  │                       │
  │                       │ ゲート信号
  │                       ▼
  └──→ [Stage 7] エンベロープ適用 ◀───────────────┘
        (attack 10-20 ms, release 50-200 ms)
        │
        ▼
        出力 (48 kHz, mono)
```

## ステージ詳細

### Stage 0: Resampler

- 入力サンプリングレートを 48 kHz に統一
- 高品質リサンプラ（`soxr` 推奨、`scipy.signal.resample_poly` でも可）

### Stage 1: DeepFilterNet 3 (NS)

- 48 kHz フルバンド処理
- アルゴリズム遅延: 約 30 ms（フレーム 20 ms + lookahead 20 ms、内部の overlap-add で実効 30 ms）
- 出力はクリーン化された対象話者 + 残った他話者音声
- 後段の VAD/SV はこの **クリーン化された音声** を使うことで判定精度が向上する

### Stage 2: silero-vad

- フレーム単位（30 ms）で speech/non-speech を 2 値判定
- ONNX 実装で軽量、Rust から ONNX Runtime 経由で呼び出し可能
- 出力は確信度スコア（[0, 1]）

### Stage 3: 動的チャンク蓄積（VAD-conditioned chunking）

- silero-vad で speech 判定されたフレームのみを内部バッファに append
- 沈黙区間のフレームはスキップ → SV 計算コストを削減
- バッファが一定長（例: 1 秒）に達したら Stage 4 をトリガー
- 連続発話中はスライディング更新（例: 250 ms ごとに最新 1 秒の埋め込みを再計算）

### Stage 4: 話者特徴抽出

#### ECAPA-TDNN（必須）
- 入力: 16 kHz, 蓄積バッファ（最低 1 秒）
- 出力: 192 次元の話者埋め込みベクトル
- 推論時間: 約 70 ms（1 秒チャンクあたり、CPU）

#### F0 抽出（補助、推奨）
- 入力: 16 kHz, 蓄積バッファ
- 出力: 平均 F0、F0 軌跡
- 用途: 登録時の F0 レンジと比較し、SV 判定の補強に使う
- 候補手法: YIN（DSP ネイティブ実装、軽量）、CREPE（ONNX、より高精度）

### Stage 5: 統合判定

```
target_score = α × cos_sim_max(emb, enrollment_pool)
             + β × f0_match(f0_mean, enrollment_f0_range)

if target_score > θ_pass:
    gate = ON
else:
    gate = OFF
```

- `cos_sim_max`: 登録埋め込みプール内の各埋め込みとの cos 類似度の最大値
- `f0_match`: 平均 F0 が登録 F0 レンジ内なら 1.0、外れるほど減衰
- `α + β = 1`（推奨初期値: `α=0.8, β=0.2`）
- 詳細は [gating.md](gating.md) 参照

### Stage 6: 自動学習プール更新（条件付き）

Stage 5 の判定で「対象話者」と確信度高く判定された場合のみ、埋め込みを自動学習プールに追加：

```
if cos_sim_max > θ_learn          (高い確信度閾値)
   and f0_match > θ_f0
   and continuous_speech > 1.0 sec
   and anchor_distance(emb) < δ:  (drift 防止)
        add(emb, auto_learn_pool)
```

詳細は [gating.md](gating.md) 参照。

### Stage 7: エンベロープ適用

- バイナリゲート信号を直接適用するとプチノイズが発生するため、attack/release エンベロープで平滑化
- `attack`: ゲート ON への遷移（推奨 10-20 ms）
- `release`: ゲート OFF への遷移（推奨 50-200 ms）
- 適用対象は **DFN3 後の 48 kHz 音声**

## 順序設計の根拠（案A 採用）

3 つの順序を検討した：

### 案A（採用）: 入力 → DFN3 → 判定 → ゲート → DFN3 後音声を出力

- 判定: クリーン音声で実施 → 高精度
- 出力: NS 処理後音声 → 通話品質向上
- DFN3 を 1 回だけ計算し、判定パスと出力パスで共有

### 案B（不採用）: 入力 → 判定 → ゲート → DFN3 → 出力

- 判定: ノイズ込み音声 → 低 SNR 時に SV 精度低下
- ゲートで mute されたフレームで DFN3 をスキップできる利点はあるが、判定精度低下が支配的

### 案C（不採用）: 入力 → DFN3 → 判定 / 元音声をゲート → 出力

- 出力に DFN3 アーティファクトを乗せたくない用途には適合
- 通話用途では NS の利点が NS アーティファクトより大きいため不採用
- 録音編集等の音楽的用途では再検討の余地あり

## レイテンシ予算

| ステージ | 寄与遅延 |
|---|---|
| Resampler | < 5 ms |
| DFN3 (NS) | ~30 ms |
| silero-vad | < 10 ms |
| エンベロープ attack | 10-20 ms |
| **合計（出力遅延）** | **~50-65 ms** |

ECAPA-TDNN の埋め込み計算は判定の更新間隔（chunk shift = 250 ms）を決めるが、出力遅延には影響しない（直近の判定結果を継続適用するため）。

つまり「ゲート判定の応答性」と「出力の絶対遅延」は別管理：

- 絶対遅延: 50-65 ms（VoIP として優秀な範囲）
- 判定更新間隔: 250-500 ms（話者切替への追従性）

## サンプリングレート方針

- 入力・出力: 48 kHz（DFN3 の native 動作レート、フルバンド品質）
- 内部判定: 16 kHz（ECAPA-TDNN の native 動作レート）

DFN3 後の 48 kHz 音声を 16 kHz にダウンサンプルして判定に使う。出力は 48 kHz のまま。これにより最終出力品質を保ちつつ、判定モデルの訓練分布と整合させる。
