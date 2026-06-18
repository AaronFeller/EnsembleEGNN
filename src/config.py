"""Configuration and token constants."""

from dataclasses import dataclass

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
ID_TO_AA = {idx: aa for aa, idx in AA_TO_ID.items()}
PAD_TOKEN_ID = len(AA_TO_ID)
UNK_TOKEN_ID = len(AA_TO_ID) + 1
MASK_TOKEN_ID = len(AA_TO_ID) + 2

# Atom-node tokens used by all-atom representations.
ATOM_TOKEN_OFFSET = 32
ATOM_SYMBOLS = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se"]
ATOM_TO_ID = {sym: ATOM_TOKEN_OFFSET + i for i, sym in enumerate(ATOM_SYMBOLS)}
ATOM_UNK_TOKEN_ID = ATOM_TOKEN_OFFSET + len(ATOM_SYMBOLS)

VOCAB_SIZE = ATOM_UNK_TOKEN_ID + 1


@dataclass
class ModelConfig:
    hidden_dim: int = 192
    egnn_layers: int = 6
    k_neighbors: int = 12
    set_inducing_points: int = 4
    attn_heads: int = 8
    dropout: float = 0.1
    fusion_mode: str = "setattn"
    use_conformer_prior: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    grad_clip: float = 1.0
