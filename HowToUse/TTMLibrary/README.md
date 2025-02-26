# About TTM Library

You should attend to these files.  
```
./tsfm/tsfm_public/models/tinytimemixer/configuration_tinytimemixer.py
./tsfm/tsfm_public/models/tinytimemixer/modeling_tinytimemixer.py
```


## configuration_tinytimemixer.py

This file defines defult parameters.  
You can change this parameter at './src/architectures/ttm.py'  



## modeling_tinytimemixer.py

### class TinyTimeMixerForPrediction()

`ttm.py`から最初に呼び出されるクラスです。

1. `__init__()`  
   - バックボーン  
   - デコーダ  
   - 出力ヘッド

2. `forward()`  
   - `past_values`をバックボーンに渡す  
   - `decoder(option, default=True)`で`hidden_state`を処理  
   - ヘッドを通して予測値`y_hat`を計算  
   - 損失を計算  
   - 出力の後処理


### class TinyTimeMixerForPredictionHead()

([B, c, patch_length, d_model] → [B, c, patch_length, FL])  ← 現在
([B, c', patch_length, d_model] → [B, c', patch_length, FL])  ← 本来
TTMのヘッドクラスです。

1. `__init__()`  
   - 入力（`hidden_features: [B, n_vars, n_patch, d_model]`）を受け取る

2. `forward()`  
   - 入力形状を調整する（`[B, n_vars, n_patch, d_model] → [B, n_vars, n_patch × d_model]`）  
   - ドロップアウト処理  
   - リニア層でアフィン変換（`[B, n_vars, n_patch, d_model] → [B, n_vars, prediction_length]`）  
     - デフォルトでは、`prediction_length`は96です。


### class TinyTimeMixerDecoder()

([B, c, patch_length, d_model] → [B, c, patch_length, d_model])  ← 現在
([B, c, patch_length, d_model] → [B, c', patch_length, d_model])  ← 本来
`hidden_state`を受け取り、最終的な他所奥の形式に変換するクラスです。

1. `__init__()`

2. `forward()`  
   - `config.d_model`と`config.decoder_d_model`が異なる場合、`nn.Linear()`で次元を揃える  
   - 生のパッチデータをデコーダ入力に加算（残差接続）  
   - `TinyTimeMixerBlock()`でミキシング  


### class TinyTimeMixerBlock()

([B, c, patch_length, d_model] → [B, c, patch_length, d_model])  ← 現在
([B, c, patch_length, d_model] → [B, c', patch_length, d_model])  ← 本来
ミキサーレイヤーまたは適応パッチのいずれかを複数層（`n = n_layers(=3)`）適用するクラスです。

1. `__init__()`

2. `forward()`  
   - `self.mixer`に`TinyTimeMixerLayer()`または`TinyTimeMixerAdaptivePatchingBlock()`のいずれかを定義（`adaptive_patching_levels(=0) > 0`の場合、適応パッチ）  
   - 各層で`embedding (=hidden_state)`を更新


### class TinyTimeMixerLayer()

([B, c, patch_length, d_model] → [B, c, patch_length, d_model])  ← 現在
([B, c, patch_length, d_model] → [B, c', patch_length, d_model])  ← 本来
([B, c, patch_length, d_model] → [B, c, patch_length, d_model])  
チャネル、パッチ、特徴量をミキシングするクラスです。

1. `__init__()`

2. `forward()`  
   - `channel_feature_mixer`でチャネル情報をミキシング  
   - `patch_mixer`でパッチ情報をミキシング  
   - `feature_mixer`で特徴量情報をミキシング  
   - 各ミキシングでデータの形状変化はありません  
   - 各ミキシングで以下の操作を行います：
     - データの正則化（インスタンス正則化）
     - Permute
     - MLP（2層） (Linear → Dropout → Linear → Dropout)


### class TinyTimeMixerModel()

([B, SL, c] → [B, c, patch_length, d_model])  ← 現在
([B, SL, c] → [B, c', patch_length, d_model])  ← 本来
時系列データを前処理し、パッチ化を行い、エンコーダに入力するクラスです。

1. `__init__()`  
   - スケーリングの初期化（デフォルト = 'std'）  
     - `config.scaling = 'mean'` → `TinyTimeMixerScaler()`  
     - `config.scaling = 'stg'` → `TinyTimeMixerStdScaler()`  
   - 重みの初期化（`config.post_init(=False)`が`True`の場合、モデルの重みを初期化しますが、これを`True`にしないこと）

2. `forward()`  
   - マスクの設定（デフォルトでは`past_values`の形状だけ「1」で埋めます）  
   - スケーリングの適用  
   - パッチ化  
     - `n_patches = SL/config.patch_length`  
     - `[B, SL, num_input_channels] → [B, num_input_channels, patches × patch_length]`  
   - パッチ化したデータをエンコーダに渡す  
     - `[B, num_input_channels, patches × patch_length] → [B, num_output_channels, patches × patch_length]`


### class TinyTimeMixerEncoder()

([B, c, patch_length, patch_size] → [B, c, patch_length, d_model])  ← 現在
([B, c, patch_length, patch_size] → [B, c', patch_length, d_model])  ← 本来
パッチ化された`past_values([B, SL, num_input_channels])`を受け取り、`d_model`次元に変換し、`TinyTimeMixerBlock`でミキシングします。

1. `__init__()`

2. `forward()`  
   - パッチごとに`nn.Linear()`で`config.patch_length` → `d_model`に変換  
   - （オプション、デフォルト=False）`resolution_prefix_tuning`  
     - 周波数情報（トークン）をエンコーダ入力に加算  
   - （オプション、デフォルト=False）位置エンコーディング  
   - `TinyTimeMixerBlock()`でミキシング


### class TinyTimeMixerPositionalEncoding()

([B, c, patch_length, d_model] → [B, c, patch_length, d_model])  
位置エンコーディングを行うクラスです。  
モデルが時系列データの位置情報を認識するための操作を行います。

1. `__init__()`  
   - 適用可否の判断（デフォルト = False）  
     - `config.use_positional_encoding = True` → 'random' または 'sincos'  
     - `config.use_positional_encoding = False` → 'zero'（実質無意味）

2. `forward()`  
   - `nn.Parameter()`を利用して`position_enc`を定義  
   - `position_enc`を`hidden_state`に加算し、位置エンコーディングを行います