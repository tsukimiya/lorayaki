# 🍡 lorayaki

> **新キャラ LoRA を、発表のその日に焼き上げる竈(かまど)。**

新キャラクターが発表されたら、画像をフォルダに放り込んで 1 コマンドで LoRA(LoCon)を焼くためのパイプライン。

```
画像投入 → タグ付け(OppaiOracle + WD14)→ キャプション確認 → データセット整備 → 学習(sdxl_train_network.py)→ サンプル画像
```

- **対象モデル**: Illustrious 系(SDXL ベース)。Anima 系は設計のみ準備済み(後述)
- **学習実行**: [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) を subprocess で叩きます(venv 構築済みの Linux GPU マシンを想定)
- **タグ付け**: [OppaiOracle](https://huggingface.co/Grio43/OppaiOracle)(in-process ONNX)+ sd-scripts 同梱の WD14 タガ

## クイックスタート(GPU マシン)

```bash
# 0. 前提: sd-scripts が venv 込みで動いていること(torch+CUDA, bitsandbytes, accelerate, xformers, onnxruntime)
git clone <this repo> && cd lorayaki
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[onnx]"            # GPU 推論が良ければ ".[onnx-gpu]"

lorayaki init --download-models  # configs/lorayaki.yaml 作成 + OppaiOracle 取得
$EDITOR configs/lorayaki.yaml    # sd_scripts_dir と models を設定
lorayaki doctor                  # 環境チェック(全 OK を確認)

lorayaki new hitori              # characters/hitori/{images/, character.yaml}
# images/ に画像(10〜40枚程度)を投入
$EDITOR characters/hitori/character.yaml   # trigger と base_model を設定

lorayaki run hitori              # タグ付け → データセット → 学習 → サンプル
```

成果物: `characters/hitori/work/output/hitori.safetensors` と `work/output/sample/` のエポック毎サンプル画像。

## コマンド

| コマンド | 内容 |
|---|---|
| `lorayaki init [--download-models]` | グローバル設定の作成(+ OppaiOracle モデル取得) |
| `lorayaki doctor` | 環境診断(sd-scripts/venv/モデル/GPU) |
| `lorayaki new <char>` | キャラ雛形の作成 |
| `lorayaki tag <char> [--force] [--oppai-only\|--wd14-only]` | タグ付けして画像の隣に `.txt` キャプションを生成 |
| `lorayaki prep <char>` | `work/dataset/` 組立 + `dataset.toml` + `sample_prompts.txt` |
| `lorayaki train <char> [--dry-run] [--preset 12gb\|16gb\|24gb] [--resume]` | 学習実行(`--dry-run` でコマンド表示のみ) |
| `lorayaki run <char>` | tag → prep → train を一括 |

共通: `--config <path>`(既定 `./configs/lorayaki.yaml`)、`-v`(詳細ログ)。
**プロジェクトルートで実行**してください(相対パスは CWD 基準)。

## キャプションのワークフロー

1. `tag` は **OppaiOracle**(全身/体型系の詳細タグが強い)と **WD14**(general/キャラ語彙)の両方を実行し、タグをマージして `<画像>.txt` を画像と同じディレクトリに書きます
2. 新キャラには既存キャラ名などの誤タグが付くため、カテゴリ `artist/copyright/character` は自動で除去し、代わりに `trigger` を先頭に固定します(`keep_tokens` で shuffle 時も位置維持)
3. **`.txt` は人手で確認・修正できます** — 誤タグの削除、特徴タグの追加など。再タグ付けは `--force`
4. `prep` がそのキャプション群を `work/dataset/` へコピーして学習に使います

タグ順序は `trigger, (always_first_tags), WD14(確信度順), OppaiOracle 固有タグ(確率順)` で決定的です。

## 設定

### configs/lorayaki.yaml(マシン設定)

```yaml
sd_scripts_dir: /home/user/sd-scripts       # 必須・絶対パス
sd_scripts_python: null                     # null なら venv/bin/python を自動検索
models:                                     # 論理キー -> パス
  illustrious-xl-0.1: /data/models/Illustrious-XL-0.1.safetensors
defaults: { backend: illustrious, preset: 16gb }
oppai_oracle:
  mode: local                               # local | http (EagleOppaiTagger のサーバを使う場合)
  model_dir: models/OppaiOracle/V1.1_onnx
wd14:
  repo_id: SmilingWolf/wd-v1-4-convnext-tagger-v2
  general_threshold: 0.35
  character_threshold: 0.85
```

### characters/<char>/character.yaml(キャラ設定)

`lorayaki new` の雛形は主要項目のみ。未設定の項目は `CHARACTER_DEFAULTS` と
VRAM プリセットから補完されます。推奨値と根拠は以下の通り:

```yaml
name: hitori
trigger: htri girl               # 必須。生成時にこの語でキャラを召喚する
base_model: illustrious-xl-0.1   # 必須。グローバルの models キー
network:                         # null の項目はプリセット値が使われる
  dim: null                      # キャラなら preset の 32 で十分。複雑/複数キャラは 64
  alpha: null                    # dim/2(=16)が既定 → 実効LRを半減し安定
  conv_dim: null                 # conv 系を設定すると LoCon になる(既定で有効)
  conv_alpha: null
training:
  epochs: 10
  num_repeats: null              # null なら ~2000 steps になるよう自動計算
  preset: null                   # null ならグローバル既定(12gb/16gb/24gb)
  unet_lr: 1e-4                  # 安定重視。似が弱い(学習不足)なら 2e-4 へ
  # extra_args:                  # 既定で noise_offset: 0.03 が入る(Illustrious ネイティブ値)
  #   noise_offset: null         #   無効化したい場合は null を指定
samples:
  prompts:
    - { prompt: "htri girl, 1girl, solo, looking at viewer, smile", width: 832, height: 1216, seed: 42 }
```

**学習率の目安**: LoRA の実効学習率は `unet_lr × (alpha/dim)` です。既定の
`alpha=16/dim=32` では `1e-4 × 0.5 = 実効 5e-5`(過学習しにくい安定側)。
より強く学習させたい場合は `alpha: 32`(=dim, 無スケール)+ `unet_lr: 1e-4`、
または `alpha: 16` + `unet_lr: 2e-4` が Illustrious 系の目安です。

### VRAM プリセット(Illustrious LoCon 既定)

| preset | dim/alpha | conv_dim/alpha | batch | max_bucket | 目安 GPU |
|---|---|---|---|---|---|
| 12gb | 16/16 | 8/8 | 1 | 1536 | RTX 3060/4070 |
| 16gb | 32/16 | 16/8 | 2 | 1536 | RTX 4080 |
| 24gb | 32/16 | 16/8 | 4 | 2048 | RTX 3090/4090 |

固定で付くフラグ: `bf16 / gradient_checkpointing / cache_latents / cache_text_encoder_outputs /
network_train_unet_only / xformers / AdamW8bit / cosine_with_restarts(num_cycles=3) / unet_lr 1e-4 /
noise_offset 0.03`(既定の `training.extra_args`)。
`training.extra_args` で任意の sd-scripts フラグを追加・上書きできます(例: `sdpa: true`)。

### チューニングの目安

- 学習が進みすぎ(硬直・焼きすぎ): `epochs` を下げる / dim を下げる
- 似ない: 画像を増やす、`num_repeats` を増やす、dim を上げる
- step 数の感覚: `画像数 × repeats × epochs / batch`(`prep` 時にログ表示)

## sd-scripts venv の前提

WD14(`--onnx`)とタグ付け全般のため、sd-scripts の venv には以下が入っていること:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # 環境に合わせる
pip install bitsandbytes accelerate xformers onnxruntime huggingface_hub
accelerate config  # 初回のみ(単一 GPU の既定で OK)
```

lorayaki 本体は **別の venv** で動かし、sd-scripts の venv と分離しています
(onnxruntime と torch の CUDA 競合を避けるため)。

## Anima 系(将来)

`backend: anima` は設定とバリデーションのみ対応済み、学習コマンド生成は未実装
(`NotImplementedError`)。sd-scripts の `anima_train_network.py` は別アーキテクチャ
(DiT + Qwen3 + Qwen-Image VAE)で、`--qwen3` `--vae` `--network_module networks.lora_anima`
など追加引数が必要。タグ付け/データセット/サンプル生成は共通化済みなので、
`backends/anima.py` の `build_train_command` を実装すれば完成します。

## 開発

```bash
pip install -e ".[dev]"
pytest tests/            # GPU 不要・ネットワーク不要で全テスト実行
```

- OppaiOracle の前処理は `tests/fixtures/expected_*.json`(EagleOppaiTagger のリファレンステンソル)
  との parity テストで保護されています
- `train --dry-run` のコマンド列はゴールデンテストで固定されています

## ライセンス

- lorayaki: MIT
- `lorayaki/taggers/oppai_oracle.py` は [EagleOppaiTagger](../EagleOppaiTagger)(Apache-2.0)由来の移植です。詳細は `NOTICE` / `LICENSE-APACHE`
- OppaiOracle のモデル重みは HuggingFace (`Grio43/OppaiOracle`) から別途取得されます(同リポジトリのライセンスを参照)
