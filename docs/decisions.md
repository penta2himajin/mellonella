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

- F0 はハードフィルタではなく統合スコアに重み β=0.2 で加味
- ガウシアン当てはまりで連続値化（厳密なレンジチェックを避ける）
- 登録時と推論時で発話状態が異なっても誤検出を最小化する設計
- 実装は YIN（DSP ベース、軽量）を第一候補、CREPE（ONNX）を高精度オプションとする

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
