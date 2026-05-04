# Gating Logic & Enrollment

## 設計方針

### FP（False Positive）許容

通話用途では「相手の声を漏らす（FP）」と「自分の声を切る（FN）」の二者択一が発生する瞬間がある。本システムは **FP 許容** を選択する：

- 対象話者の声紋成分が含まれていれば pass
- 同時発話時に他話者の音漏れがあっても、対象話者の発話は確実に通す
- 自分の発話が短時間でも切れる方が体験を損なうという判断

### 単一話者ターゲット前提

本システムは「特定の 1 人の話者を通す」設計に特化する。複数話者をターゲットにする拡張は将来的な検討事項とし、初期実装では考慮しない。

## 統合判定式

```
target_score(t) = α × cos_sim_max(t)  +  β × f0_match(t)

where:
    cos_sim_max(t) = max over emb_i ∈ enrollment_pool ∪ auto_learn_pool {
                        cos_similarity(current_embedding(t), emb_i)
                     }
    f0_match(t)    = exp(-((f0_mean(t) - μ_enroll) / σ_enroll)^2 / 2)
                     ※ 登録 F0 ガウシアンへの当てはまり度
    α + β = 1.0
    推奨初期値: α = 0.8, β = 0.2
```

## 二段階閾値

FP 許容方針 + 自動学習併用のため、用途を分離した 2 つの閾値を設ける：

| 閾値 | 用途 | 推奨初期値 |
|---|---|---|
| `θ_pass` | 出力ゲート判定 | 0.30 |
| `θ_learn` | 自動学習プールへの追加可否 | 0.80 |

`θ_pass < θ_learn` という関係を厳守する。理由：

- 出力ゲートは多少緩く（FP 許容 = 取りこぼし防止）
- 自動学習は厳格に（drift 防止 = 確信度の高いソロ発話のみ採用）

> **`θ_pass` の calibration 履歴**: 当初は clean-vs-clean の cos 類似度直感から `0.50` を仮置きしていたが、`scripts/calibrate.py` で librosa libri1/2/3 × white/pink ノイズ × SNR -5..20 dB の 108 セルを sweep した結果 `0.50` ではノイズ下で gate が完全閉になることが判明。FP 許容方針 (mean FPR ≤ 0.05) を満たす最小 θ_pass として `0.30` を選択（median TPR ≈ 0.84, mean FPR ≈ 4.6 %）。詳細は [`benchmarks/calibration_summary.json`](benchmarks/calibration_summary.json) 参照。

この分離により、自動学習による drift リスクを抑制しつつ FP 許容を実現する。

## ハングオーバー

短時間の無声音（破裂音前の閉鎖期、息継ぎ等）で誤って OFF に切り替わるのを防ぐ：

```
if gate(t-1) == ON and target_score(t) < θ_pass:
    if elapsed_off_duration < hangover_ms (推奨 200-500 ms):
        gate(t) = ON  # 維持
    else:
        gate(t) = OFF
```

## エンベロープ

バイナリ ON/OFF を直接適用するとクリック音や音切れが目立つため、ゲート信号に attack/release を適用：

```
attack_coef  = 1 - exp(-1 / (attack_ms  × sr / 1000))
release_coef = 1 - exp(-1 / (release_ms × sr / 1000))

if target_gate(t) == ON:
    envelope(t) = envelope(t-1) + attack_coef × (1.0 - envelope(t-1))
else:
    envelope(t) = envelope(t-1) + release_coef × (0.0 - envelope(t-1))

output(t) = dfn3_output(t) × envelope(t)
```

推奨値:
- `attack_ms = 15`（瞬時反応）
- `release_ms = 100`（緩やかなフェードアウト）

## 登録（Enrollment）

### 明示登録プロトコル

1. ユーザーに 30 秒〜1 分の音声録音を要求
   - 短文・長文・相槌などバリエーションを含む発話文脈
   - SNR > 20 dB のクリーン録音
2. 録音から 5-10 個の埋め込みを抽出
   - スライディングウィンドウ（例: 3 秒チャンク、1.5 秒シフト）
   - 各チャンクごとに ECAPA-TDNN を実行
3. F0 統計量も記録
   - `μ_enroll`, `σ_enroll` を発話部分のみから計算
4. 上記を `enrollment_pool` として永続保存

### 自動学習（Auto-learning）

通話中に高確信度で「対象話者」と判定されたフレームから埋め込みを継続的に追加：

```
採用条件:
    cos_sim_max(t) > θ_learn        (= 0.80)
    AND f0_match(t) > θ_f0          (= 0.7 推奨)
    AND continuous_speech > 1.0 sec
    AND anchor_distance(emb) < δ    (drift 防止、後述)
    AND auto_learn_pool.is_consistent()
```

### Anchor 保護

明示登録時の埋め込みを **anchor** として永久保持し、自動学習で削除されない：

```
struct EmbeddingPool:
    anchors: Vec<Embedding>          # 明示登録、不変
    auto_learn: VecDeque<Embedding>  # 自動学習、FIFO 上限あり
    max_auto_learn_size: usize = 20
```

### 整合性チェック

自動学習プールへの追加前に、anchor との距離を検証：

```
fn anchor_distance(emb: &Embedding, anchors: &[Embedding]) -> f32 {
    1.0 - anchors.iter()
                  .map(|a| cos_similarity(emb, a))
                  .max()
}

if anchor_distance(emb) > δ (= 0.4 推奨):
    reject  // anchor から離れすぎている、drift 兆候
```

### 定期的な異常検知

プール全体の中央値を監視し、anchor から大きく逸脱したら自動学習部分をリセット：

```
周期: 5 分ごと、または auto_learn_pool 更新 N 回ごと

if median(auto_learn_pool) の anchor_distance > δ_reset (= 0.5):
    auto_learn_pool.clear()
    log_warning("auto-learn pool drifted, resetting")
```

## VAD-conditioned 動的チャンキング

ECAPA-TDNN は本質的に 1 秒以上のサンプルで安定する。しかし固定 1 秒バッファを常時保持すると、沈黙区間が混入し精度が低下する。

対策: **silero-vad で speech 判定されたフレームのみを内部バッファに append**：

```
let mut speech_buffer: VecDeque<f32> = VecDeque::new();
let mut last_emb_update: Instant = Instant::now();

for frame in input_stream {
    let dfn3_out = dfn3.process(frame);
    let downsampled = resample(dfn3_out, 48000, 16000);
    let vad_score = vad.process(downsampled);

    if vad_score > 0.5 {
        speech_buffer.extend(downsampled);
        if speech_buffer.len() > MAX_BUFFER {
            speech_buffer.drain(..speech_buffer.len() - MAX_BUFFER);
        }
    }

    // 250 ms ごと、かつバッファが 1 秒以上で SV 更新
    if last_emb_update.elapsed() > Duration::from_millis(250)
       && speech_buffer.len() >= 16000 {
        let emb = ecapa.embed(&speech_buffer);
        let f0_mean = f0.estimate(&speech_buffer);
        update_target_score(emb, f0_mean);
        last_emb_update = Instant::now();
    }

    let envelope = update_envelope(target_gate);
    output_stream.push(dfn3_out * envelope);
}
```

## F0 補助判定の意義

ECAPA-TDNN だけでは「対象話者と声紋が似た別人」を区別しきれない場合がある。F0 レンジは個人差が大きく、補助判定として有効：

- 男性平均 F0: 約 120 Hz（個人差 80-180 Hz）
- 女性平均 F0: 約 220 Hz（個人差 150-300 Hz）
- 同性であっても F0 の標準偏差レベルでの違いは判定に寄与する

本システムでは F0 を「ハードフィルタ」ではなく「ソフトな補強」として使う：

- F0 マッチ度をガウシアンで計算（0.0-1.0）
- 統合スコアに重み β = 0.2 で加味
- F0 レンジ外でも cos sim が十分高ければ pass する設計

これにより、登録時と推論時で発話状態（普段の声 vs 興奮した声）が異なっても誤検出を最小化できる。

## 古典手法との将来的な統合

F0 マッチ以外にも、信号処理ベースの補強候補がある（優先度低）：

- **Harmonic + Residual Model (HNM)**: 音声を周期成分・非周期成分に分解、周期成分のみ pass
- **Computational Auditory Scene Analysis (CASA)**: ハーモニック構造に沿った時間-周波数マスク
- **Spectral envelope matching**: 登録音声の MFCC/LPC エンベロープと推論時を比較

初期実装ではスキップ。F0 マッチで判定精度が不足する場合に追加検討。

## AS-Norm（Adaptive S-Norm）によるスコア正規化

詳細は `docs/decisions.md` D-010 参照。

### 動機

`target_score(emb, pool) = α·max_cos_sim + β·f0_match` の生 score は、
言語/ノイズ/録音条件によって絶対値分布がシフトする。Scenario 5 (PR #17)
の baseline 計測では、単一 global θ_pass=0.30 で:

- ja TPR=0.59 (低 SNR、target を取りこぼす方向)
- zh-CN FPR=0.33 (低 SNR、other を通す方向)

という対称的な失敗が出た。**1 つの threshold で複数言語を同時最適化できない**。

### 仕組み

事前構築した「impostor cohort」(多言語の非ターゲット話者 30-50 名分の
ECAPA embedding) と test 時 embedding の cosine sim を計算し、その top-K
スコアの平均/標準偏差で target score を z-score 正規化する:

```
S_norm = (S_target - μ_top-K(S_impostor)) / σ_top-K(S_impostor)
```

これにより score の絶対値分布シフト (言語別 bias) が消え、global
θ_pass で複数言語をカバーできる。

### コホート構築

`scripts/build_impostor_cohort.py` が manifest (CommonVoice / MLS /
Emilia-YODAS いずれの形式も可) から ECAPA embedding を抽出し
`.npz` で保存する。

```bash
python scripts/build_impostor_cohort.py \
    --manifest en=$MELLONELLA_DATA_DIR/emilia_yodas/en/manifest.csv \
    --manifest de=$MELLONELLA_DATA_DIR/mls/de/manifest.csv \
    --manifest ja=$MELLONELLA_DATA_DIR/emilia_yodas/ja/manifest.csv \
    --per-language 5 \
    --output bench/data/cohorts/impostor_cohort_v1.npz
```

各言語 5 話者程度で十分 (cohort size は EER に対し対数的に効く)。
出力ファイルは ~38 KB (50 話者 × 192 dim × 4B)、package 同梱可。

### 段階的導入

Phase 1 ✅ (PR #18): cohort build script + docs。
Phase 2 ✅ (PR #19): `gating.py` に AS-Norm 実装、`GatingConfig` 拡張、
`pipeline.process_offline` 分岐、CI で cohort 自動 build → scenario_5 に反映。
**Phase 2 実観測**: ja TPR が 0.67 → 0.85 (+18pp) で取りこぼし問題解消、
ただし zh-CN FPR が 0.23 → 0.42 (+19pp) と悪化、de TPR も -8pp の新規
regression。`docs/decisions.md` D-010「Phase 2 実観測」/`docs/benchmarks.md`
シナリオ 5「Phase 2: AS-Norm 適用後の deltas」参照。default
`theta_pass_as_norm=1.5` のヒューリスティック値が言語ごと過剰/不足。

Phase 3 (進行中): `scripts/calibrate.py` に AS-Norm 拡張 + per-language sweep
で `theta_pass_as_norm` を data 駆動で確定。完了後に scenario_5 hard-fail
閾値も引き締める。
Phase 4 (任意): C/D/E への拡張、または cohort 拡大 (per-language 5 → 10、
top-K 10 → 20) — Phase 3 結果次第で要否判断。

### 実装メモ (Phase 2)

- 公開 API:
  - `mellonella_poc.gating.load_cohort(path) -> np.ndarray`
  - `mellonella_poc.gating.as_norm_score(emb, raw, cohort, top_k) -> float`
  - `mellonella_poc.gating.target_score_as_norm(emb, pool, cohort, config) -> float`
- `GatingConfig` 新フィールド:
  - `use_as_norm: bool = False`
  - `as_norm_cohort_path: str | None = None`
  - `as_norm_top_k: int = 10`
  - `theta_pass_as_norm: float = 1.5` (z-score scale)
  - `theta_learn_as_norm: float = 2.5`
- `PipelineComponents` が cohort を build 時に 1 度ロードし、`process_offline`
  が per-frame で参照する (毎呼び出し再ロードはしない)。
- AS-Norm 経路では per-frame の score formula が
  `α·cs + β·f0_match` から `as_norm(cs vs cohort)` に切り替わる。F0 は
  引き続き auto-learn 入口の `theta_f0` で使われる。
- `use_as_norm = False` のとき既存パスは bit-identical (deault False)。

### CLI / CI 統合

- `scripts/scenario_5_from_manifest.py --as-norm-cohort PATH` で AS-Norm
  経路に切り替え可能 (`--real-pipeline` 必須)。
- `.github/workflows/scenario_5.yml` は MLS + Emilia 準備後に
  `scripts/build_impostor_cohort.py` で cohort を auto-build し、
  scenario_5 に渡す。第三者 cohort artifact 管理は不要。
