# Benchmarks

## 評価方針

PoC 段階での性能検証のため、**ミニマルかつ商用クリーンライセンス**のベンチマークデータセットを組み合わせる。本ドキュメントは：

- 評価したい指標
- 採用するデータセットとライセンス
- 評価シナリオ
- ミニマル評価セットの構成

を定める。

## 評価したい指標

### A. NS（DFN3）品質

| 指標 | 説明 | 範囲 |
|---|---|---|
| PESQ | ITU-T P.862、知覚的音声品質 | -0.5〜4.5 |
| STOI | Short-Time Objective Intelligibility | 0〜1 |
| SI-SDR | Scale-Invariant Signal-to-Distortion Ratio | dB |
| DNSMOS P.835 | Microsoft の non-intrusive perceptual quality metric (SIG/BAK/OVRL) | 1〜5 |
| UTMOS | non-intrusive UTokyo MOS predictor | 1〜5 |

### B. VAD 精度

| 指標 | 説明 |
|---|---|
| Frame-level F1 | speech/non-speech のフレーム単位 F1 |
| Frame accuracy | 正解フレーム / 全フレーム |
| Onset/Offset error | 発話開始/終了時刻の誤差（ms） |

### C. SV 判定精度

| 指標 | 説明 |
|---|---|
| EER | Equal Error Rate（target vs non-target） |
| Gating accuracy | フレームレベルでの正解判定率 |
| False Positive rate | 他話者を pass してしまう率 |
| False Negative rate | 対象話者を mute してしまう率 |

### D. 統合パイプライン

| 指標 | 説明 |
|---|---|
| 全指標の組合せ | A〜C すべて |
| 総レイテンシ実測 | 入力 → 出力の wall-clock |
| CPU 使用率 | 単一スレッド時の使用率 |
| メモリフットプリント | 推論時のピーク使用量 |

### E. 主観評価（補助）

PoC 段階で自分・家族・同僚による試聴：

- 5段階 MOS 評価（1: bad → 5: excellent）
- A/B test: 元音声 vs 処理後、ハードゲーティング vs DFN3単体

## 採用データセット

### NS 品質評価: VoiceBank+DEMAND

業界標準の SE 評価ベンチマーク。

- **VoiceBank (VCTK)**: CC BY 4.0 / ODC-By 1.0
- **DEMAND**: CC BY-SA 3.0
- **構成**:
  - test set: 824 paired utterances, 2 話者（unseen）, 5-10 noise types
  - SNR: 2.5 / 7.5 / 12.5 / 17.5 dB
  - 16 kHz / 48 kHz どちらでも利用可能
- **役割**: DFN3 単体の NS 性能、PESQ/STOI/CSIG/CBAK/COVL の標準ベースライン取得

### 多言語ロバスト性: Mozilla Common Voice

CommonVoice は CC0 ライセンスで 250+ 言語をカバー。本プロジェクトの主軸となる多言語評価データセット。

- **License**: CC0-1.0（パブリックドメイン）
- **制約**:
  - 再ホスト・再配布不可（自身のプロジェクト内利用は OK）
  - 話者の身元特定試行不可
- **規模**: v18 時点で 31,841 時間（validated 20,789 時間）、v23/v24 でさらに拡張
- **PoC 用サブセット**: 主要 5-10 言語、各 50 発話程度を抽出
- **対象言語候補**:

| 言語 | コード | 用途 |
|---|---|---|
| 英語 | en | 標準・ベースライン |
| 日本語 | ja | 主要利用言語 |
| ドイツ語 | de | 印欧語、子音多め |
| フランス語 | fr | 印欧語、母音強め |
| 中国語 | zh-CN | 声調言語 |
| スペイン語 | es | 印欧語、ロマンス |
| 韓国語 | ko | 膠着語 |
| アラビア語 | ar | 非印欧語 |

ECAPA-TDNN は VoxCeleb（多言語）で訓練されているため、原則として言語非依存だが、各言語での EER 検証で実機性能を確認する。

### 補助多言語: Multilingual LibriSpeech (MLS)

audiobook 由来の高品質スタジオ録音。Common Voice より発話品質が均一で、評価結果のばらつきを抑えるのに向く。

- **License**: CC0（パブリックドメイン、LibriVox + Project Gutenberg 由来）
- **言語**: 8 言語（英語、ドイツ語、オランダ語、スペイン語、フランス語、イタリア語、ポルトガル語、ポーランド語）
- **規模**: 英語 44.5K 時間、その他合計 6K 時間
- **配布**: HuggingFace `facebook/multilingual_librispeech`
- **PoC 用途**: test split から各言語 30-50 発話抽出

### ノイズデータセット

#### MUSAN (Apache 2.0)

- **License**: Creative Commons (flexible)
- **規模**: 約 60 GB
- **構成**: speech / music / noise の 3 カテゴリ
- **役割**: 多様なノイズ条件でのカスタム評価セット生成
- **配布**: OpenSLR (http://www.openslr.org/17/)

#### DEMAND (CC BY-SA 3.0)

- **License**: CC BY-SA 3.0
- **構成**: 18 種類のリアル環境録音（kitchen, office, park, traffic 等）
- **役割**: VoiceBank+DEMAND として標準利用、リアルな環境ノイズ
- **派生物 ShareAlike 義務あり**（最終出力データの公開時）

#### DNS Challenge dataset (CC BY 4.0 / MIT)

- **License**: コード MIT、データ CC BY 4.0
- **規模**: fullband (48 kHz) 大規模
- **役割**:
  - DFN3 が訓練に使用したデータセットの一部
  - DNS5 (ICASSP 2023) test set はパーソナライズタスク含む
- **配布**: https://github.com/microsoft/DNS-Challenge

### 多話者シナリオ: LibriMix

- **License**: CC BY 4.0（LibriSpeech 由来）
- **生成方法**: スクリプト公開、自前で生成可能
- **構成**: 2話者・3話者混合
- **WHAM! 不使用版**: `mix_clean` モードで WHAM! を使わずに生成可能 → 商用クリーン
- **役割**: 同時発話・順番発話シナリオの定量評価

### 日本語の選択肢

CommonVoice ja を主軸とするが、以下も参考データとして利用可能：

| データセット | License | 商用利用 |
|---|---|---|
| **CommonVoice (ja)** | CC0 | ✅ 推奨 |
| JSUT corpus | CC BY-SA 4.0 (text/labels), audio は要個別交渉 | △ TLO 経由 |
| JVS corpus | 同上 | △ TLO 経由 |
| ReazonSpeech | CDLA-Sharing-1.0 | ⚠️ 研究用途中心 |

JSUT/JVS は東京大学 TLO 経由の個別契約が必要。本プロジェクトでは商用展開を見据え、**CommonVoice ja のみを採用**する。

### 採用しないデータセット

以下は前回の調査で除外：

- **WHAM!**: CC BY-NC 4.0、非商用のみ
- **WSJ0**: LDC proprietary、有償
- **VoxCeleb1/2**: ECAPA-TDNN の訓練データだが、BBC/YouTube 由来でグレーゾーン。評価専用利用に留める

## 評価シナリオ

### シナリオ 1: ソロ対象話者 + ノイズ

```
入力: 対象話者音声 + 環境ノイズ (MUSAN / DEMAND)
期待: ゲート pass、ノイズ抑制された対象話者音声
```

評価指標:
- PESQ, STOI, SI-SDR（NS 品質）
- True Positive rate（gating）
- Onset/Offset error（VAD 精度）

### シナリオ 2: ソロ他話者 + ノイズ

```
入力: 他話者音声 + 環境ノイズ
期待: ゲート mute、無音出力
```

評価指標:
- True Negative rate（gating）
- False Positive rate（誤って pass する率）

### シナリオ 3: 順番発話（対象 → 他者 → 対象）

```
入力: 対象話者発話 → 沈黙 → 他話者発話 → 沈黙 → 対象話者発話
期待: 対象部分のみ pass、他者部分は mute
```

評価指標:
- Frame-level accuracy
- Attack time（pass への遷移時間）
- Release time（mute への遷移時間）

### シナリオ 4: 同時発話（対象 + 他者）

```
入力: 対象話者と他話者の同時発話
期待: pass（FP 許容方針）、対象話者の明瞭度を保持
```

評価指標:
- 主観評価（対象話者の聞き取りやすさ）
- 客観 SI-SDR（対象話者音声の歪み）

### シナリオ 5: 多言語ロバスト性

```
入力: 各言語の対象話者音声 + ノイズ（言語横断）
期待: 言語に関係なく安定した SV 判定
```

評価指標:
- 言語別 EER
- 言語別 gating accuracy
- 言語間ばらつきの統計（標準偏差）

### シナリオ 6: 経時変化への対応（自動学習効果）

```
入力: 同一話者の異なる声質変化（風邪・疲労を模擬、感情の異なる発話）
期待: 自動学習プールが更新され、判定精度を維持
```

評価指標:
- 時間経過での gating accuracy 推移
- Anchor 距離（drift 検知）

## ミニマル評価セット（PoC 段階）

PoC では実行時間 < 1 時間 を目標とした最小構成：

### サンプル数

| データセット | サンプル数 | 用途 |
|---|---|---|
| VoiceBank+DEMAND test | 100 utterances（ランダム選択） | NS 品質ベース評価（シナリオ 1 の一部） |
| CommonVoice 5言語 | 各 50 utterances = 250 | 多言語ロバスト性（シナリオ 5） |
| MLS 5言語 | 各 30 utterances = 150 | 高品質多言語補助 |
| MUSAN | 各カテゴリ 10 noises = 30 | カスタム混合用ノイズ |
| LibriMix mix_clean test | 100 mixtures | 多話者シナリオ（シナリオ 2/3/4） |

合計 約 630 評価サンプル

### 評価実行構成

PoC 段階では：

1. **シナリオ 1（ソロ + ノイズ）**: VoiceBank+DEMAND を主軸、MUSAN を補助
2. **シナリオ 2/3/4（多話者）**: LibriMix を主軸
3. **シナリオ 5（多言語）**: CommonVoice + MLS をクロス利用
4. **シナリオ 6（自動学習）**: 自前収録音声で長時間試験

評価実行はバッチで一括実行可能とし、CSV 形式で結果を出力する：

```
benchmark_results/
├── scenario_1_solo_noise.csv
├── scenario_2_other_speaker.csv
├── scenario_3_alternating.csv
├── scenario_4_simultaneous.csv
├── scenario_5_multilingual.csv
├── scenario_6_drift.csv
└── summary.json
```

## 比較対象

ハードゲーティング型の性能を相対化するため、以下と比較：

### ベースライン

1. **何も処理しない原音声**: 下限ベースライン
2. **DFN3 単体**: NS のみ、SV なし。NS 効果の純粋測定
3. **オラクル VAD（ground truth）**: 完全な VAD 情報を与えた場合の上限

### 既存手法との比較

| 手法 | 用途 | 想定結果 |
|---|---|---|
| ConVoiFilter（オフライン） | 真の TSE のリファレンス | より高い分離精度を達成するが 5 秒遅延 |
| ESPnet TD-SpeakerBeam | causal 寄りの TSE | 性能と遅延のトレードオフ評価 |

これらは「リアルタイム可能な TSE と同等の精度をハードゲーティング型で達成できているか」の参照点。

## ベンチマーク用ツール

実装で使用する評価ライブラリ：

| 用途 | ライブラリ | License |
|---|---|---|
| PESQ | `pesq` (PyPI) | MIT |
| STOI | `pystoi` (PyPI) | MIT |
| SI-SDR | `torchmetrics` または自前計算 | Apache 2.0 |
| DNSMOS | Microsoft の P.835 ONNX モデル | MIT |
| UTMOS | UTokyo の MOS predictor | BSD-3 |
| SV (EER 計算) | `speechbrain.utils.metric_stats` | Apache 2.0 |

## ベンチマーク実行の自動化

`bench/` ディレクトリに以下を配置（実装段階で構築）：

```
bench/
├── datasets/                # データダウンロードスクリプト
│   ├── download_vbd.sh
│   ├── download_commonvoice.py
│   ├── download_mls.py
│   ├── download_musan.sh
│   └── generate_librimix_clean.py
├── scenarios/               # シナリオ別評価スクリプト
│   ├── scenario_1_solo_noise.py
│   ├── scenario_2_other_speaker.py
│   ├── scenario_3_alternating.py
│   ├── scenario_4_simultaneous.py
│   ├── scenario_5_multilingual.py
│   └── scenario_6_drift.py
├── metrics/                 # 評価指標計算
│   ├── ns_quality.py        # PESQ, STOI, SI-SDR, DNSMOS
│   ├── vad_accuracy.py
│   ├── sv_eer.py
│   └── gating_accuracy.py
├── runners/
│   └── run_all.py           # 全シナリオ一括実行
└── results/                 # 出力先
    └── (CSV, JSON, plots)
```

## 推奨実行順序

1. **Phase 1 PoC 完了後すぐ**: シナリオ 1 + シナリオ 5（NS 品質と多言語ロバスト性）
2. **Phase 2 自動学習実装後**: シナリオ 6（drift 検証）
3. **Phase 3 Rust 移植後**: 全シナリオ + レイテンシ・CPU 実測
4. **Phase 4 モバイル展開後**: モバイル実機でのレイテンシ・バッテリー測定

各 Phase で次に進む前のゲート条件として、対応するシナリオの最低基準クリアを設定する。具体的な閾値は Phase 1 の初期測定後に確定する。
