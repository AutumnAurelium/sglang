from transformers import AfmoeConfig as TransformersAfmoeConfig


class AfmoeConfig(TransformersAfmoeConfig):
    """Native AFMoE config with SGLang's defaults and legacy routing fields."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 256
    num_attention_heads: int = 32
    head_dim: int | None = None
    max_position_embeddings: int = 131072
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    num_shared_experts: int | None = 0
    num_dense_layers: int | None = 0
    sliding_window: int | None = None
    score_func: str = "sigmoid"
    route_norm: bool = True
    n_group: int | None = 1
    topk_group: int | None = 1

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)
