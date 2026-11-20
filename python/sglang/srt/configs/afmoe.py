from transformers import AfmoeConfig as TransformersAfmoeConfig


class AfmoeConfig(TransformersAfmoeConfig):
    """Native AFMoE config with the routing fields used by legacy checkpoints."""

    score_func: str = "sigmoid"
    route_norm: bool = True
    n_group: int | None = 1
    topk_group: int | None = 1
    head_dim: int | None = None

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)
