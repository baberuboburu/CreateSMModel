class TTMConfiguration():
  def __init__(
    self,
    num_input_channels: int = 7,
    num_output_channels: int = 1,
    sl: int = 16,
    fl: int = 1,
    patch_length: int = 1,
    patch_stride: int = 1,
    d_model: int = 192,
    dropout: float = 0.4,
    head_dropout: float = 0.2,
    encoder_num_layers: int = 3,
    decoder_num_layers: int = 8,
    encoder_mode: str = "mix_channel",
    decoder_mode: str = "common_channel",
    use_positional_encoding: bool = False,
    positional_encoding_type: str = 'sincos',
    adaptive_patching_levels: int = 0,
    norm_type: str = 'LayerNorm',
    norm_eps: float = 1e-5,
    mlp_expansion: int = 2,
    scaler_type: str = 'std',
    scaling_dim: int = 1,
    scaling_keepdim: bool = True,
    minimum_scale: float = 1e-5
  ):

    # Depend on data properties
    self.num_input_channels = num_input_channels
    self.num_output_channels = num_output_channels
    self.sl = sl
    self.fl = fl

    # Fundamental model configuration
    self.patch_length = patch_length
    self.patch_stride = patch_stride
    self.d_model = d_model
    self.dropout = dropout
    self.head_dropout = head_dropout

    # Encoder
    self.encoder_num_layers = encoder_num_layers
    self.encoder_mode = encoder_mode

    # Positional Encoding
    self.use_positional_encoding = use_positional_encoding
    self.positional_encoding_type = positional_encoding_type

    # Decoder
    self.decoder_num_layers = decoder_num_layers
    self.decoder_mode = decoder_mode

    # Adaptive Patch
    self.adaptive_patching_levels = adaptive_patching_levels

    # Normalization
    self.norm_type = norm_type
    self.norm_eps = norm_eps

    # MLP
    self.mlp_expansion = mlp_expansion

    # Scaler
    self.scaler_type = scaler_type
    self.scaling_dim = scaling_dim
    self.scaling_keepdim = scaling_keepdim
    self.minimum_scale = minimum_scale

    # num_patches
    if not hasattr(self, "init_processing"):
      self.check_and_init_preprocessing()


  def check_and_init_preprocessing(self):
    self.init_processing = True
    
    # Define num_patches at the first time only.
    if not hasattr(self, "num_patches"):
      self.num_patches = (max(self.sl, self.patch_length) - self.patch_length) // self.patch_stride + 1