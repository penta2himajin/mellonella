# Implementation Plan

## 技術スタック

### コア言語

**Rust** をコア実装言語とする：

- DeepFilterNet 3 が公式 Rust 実装（`deep_filter` crate）を提供している
- ONNX Runtime の Rust binding（`ort`）が成熟しており、PyTorch モデルを ONNX 経由で利用可能
- 単一バイナリでデスクトップ・モバイル両対応が現実的
- ユーザーの Rust 適性とも整合

Python は PoC・検証用途に限定し、本実装は Rust に寄せる。

### 推論ランタイム

**ONNX Runtime** を統一推論ランタイムとする。理由：

- silero-vad、ECAPA-TDNN、CREPE すべて ONNX 化されたモデルが公開されている
- INT8 量子化が容易（モバイル展開時のサイズ削減）
- CPU/GPU 両対応、CoreML / NNAPI 経由で各プラットフォームのアクセラレータ利用可能

DFN3 のみ独立した Rust 実装を使う（公式の `deep_filter` crate）。

### コンポーネント別実装

| コンポーネント | 実装 | 形式 |
|---|---|---|
| Resampler | `rubato` crate | Rust ネイティブ |
| DeepFilterNet 3 | `deep_filter` crate | Rust ネイティブ |
| silero-vad | ONNX | `ort` 経由 |
| ECAPA-TDNN | ONNX（SpeechBrain → ONNX 変換） | `ort` 経由 |
| F0 (YIN) | 自前実装または `pitch-detection` crate | Rust ネイティブ |
| F0 (CREPE, optional) | ONNX | `ort` 経由 |

## プラットフォーム対応

### デスクトップ

- **Linux**: x86_64-unknown-linux-gnu
- **macOS**: aarch64-apple-darwin（Apple Silicon）, x86_64-apple-darwin
- **Windows**: x86_64-pc-windows-msvc

統合方法:
- ライブラリ（`.so`/`.dylib`/`.dll`）として配布
- CLI ツールとしての単体実行も可能
- 仮想オーディオデバイス連携（PipeWire / CoreAudio / WASAPI）は将来検討

### モバイル

- **iOS**: aarch64-apple-ios
- **Android**: aarch64-linux-android

最適化:
- INT8 量子化で ECAPA-TDNN を 23 MB → 約 6 MB に圧縮
- CoreML / NNAPI バックエンド利用でアクセラレータ活用
- 起動時遅延短縮のため、モデルファイルは事前バンドル

推定バイナリサイズ:
- DFN3: 6 MB
- silero-vad: 2 MB
- ECAPA-TDNN (INT8): 6 MB
- F0 (YIN): 0 MB（DSP コードのみ）
- ランタイム + 周辺: 約 10 MB
- **合計**: 約 25 MB

## 実装フェーズ

### Phase 1: PoC（Python + PyTorch）

目的: アルゴリズム妥当性検証、閾値・パラメータ初期チューニング

タスク:
- silero-vad + ECAPA-TDNN + DFN3 の Python パイプライン構築
- 明示登録機構の実装
- ゲートロジック（統合判定 + ハングオーバー + エンベロープ）の実装
- 自分の声 + 環境音の混合データで動作確認
- 閾値（θ_pass, θ_learn）の初期値検証

想定期間: 1-2 週間
成果物: 動作する Jupyter Notebook + 簡易 CLI

### Phase 2: 拡張機能追加（Python）

目的: F0 補助判定と自動学習の検証

タスク:
- F0 抽出（YIN または CREPE）追加
- F0 マッチによる統合判定の改善検証
- 自動学習プール実装
- Anchor 保護、drift 検出機構の実装
- 長時間通話シミュレーションでの安定性検証

想定期間: 1-2 週間
成果物: 機能完成版 Python 実装

### Phase 3: Rust 移植（デスクトップ）

目的: デスクトップ向け本実装、性能最適化

タスク:
- ONNX 変換: ECAPA-TDNN（SpeechBrain → ONNX）、silero-vad は既製
- Rust crate 構造設計（`mellonella-core`, `mellonella-cli`, `mellonella-ffi` 等）
- ストリーミング処理（リングバッファ、フレーム同期）
- DFN3 の Rust 実装統合
- ONNX Runtime 統合（`ort` crate）
- ベンチマーク（Python 版との性能比較、CPU 使用率測定）

想定期間: 2-3 週間
成果物: Linux/macOS/Windows 動作する CLI + ライブラリ

### Phase 4: モバイル対応

目的: iOS/Android で動作するバイナリ

タスク:
- iOS: Swift 経由で Rust ライブラリ呼び出し（`cbindgen` + Swift Package）
- Android: Kotlin 経由で JNI 呼び出し
- INT8 量子化適用
- CoreML / NNAPI バックエンドの動作確認
- バッテリー消費測定

想定期間: 2-3 週間
成果物: iOS/Android 用 SDK + サンプルアプリ

### Phase 5: 仮想オーディオデバイス連携（オプション）

通話アプリ（Zoom、Google Meet 等）と統合するための、システム全体への適用：

- macOS: BlackHole / Loopback 経由、または CoreAudio HAL プラグイン
- Linux: PipeWire filter-chain（DFN3 が既にこの形式で提供されている、参考になる）
- Windows: VB-Cable + WASAPI、または独自 APO 開発

複雑度が高いため、Phase 5 は別プロジェクトとして切り出す可能性あり。

## ベンチマーク方針

### 客観評価メトリクス

- **PESQ / STOI**: NS 部分の品質
- **SI-SNR**: 抽出された対象話者音声と原音の比較
- **EER（Equal Error Rate）**: 対象話者 vs 他話者の判別性能
- **CPU 使用率**: 各コンポーネント単体 + 統合パイプライン
- **メモリフットプリント**: モバイル展開を意識した実測

### 比較対象

- ConVoiFilter（オフライン TSE のリファレンス）
- ESPnet TD-SpeakerBeam（オフライン TSE）
- DFN3 単体（NS のみのリファレンス）
- 何も処理しない原音（ベースライン）

比較は LibriMix の causal 評価セット、または自家製録音セットで実施。

### 主観評価

PoC 段階で：
- 自分の声 + 家族・同僚の混合録音
- 異なる SNR 条件での試聴
- 同時発話シーンでの体感評価

本実装段階で：
- 通話アプリと連携した実環境テスト
- 第三者（数名）による A/B テスト

## 開発環境とディレクトリ構成（暫定）

```
mellonella/
├── docs/                          # 本仕様書群
├── poc/                           # Phase 1-2: Python PoC
│   ├── notebooks/
│   ├── mellonella_poc/
│   └── pyproject.toml
├── crates/                        # Phase 3: Rust 本実装
│   ├── mellonella-core/           # コアロジック
│   ├── mellonella-cli/            # CLI
│   ├── mellonella-ffi/            # FFI（モバイル用）
│   └── mellonella-bench/          # ベンチマーク
├── models/                        # ONNX 変換済みモデル（git-lfs）
├── mobile/                        # Phase 4
│   ├── ios/
│   └── android/
└── tests/                         # 統合テスト・ベンチマーク用音声
```

## 依存関係（暫定リスト）

### Rust

- `deep_filter`（DFN3 公式）
- `ort`（ONNX Runtime binding）
- `rubato`（リサンプリング）
- `ndarray`（テンソル操作）
- `pitch-detection`（YIN）または自前実装
- `crossbeam` または `flume`（ストリーミング channel）
- `serde` + `serde_json`（設定ファイル）

### Python（PoC）

- `torch`, `torchaudio`
- `speechbrain`（ECAPA-TDNN）
- `silero-vad`
- `deepfilternet`
- `librosa`（オーディオ前処理）
- `numpy`, `scipy`
