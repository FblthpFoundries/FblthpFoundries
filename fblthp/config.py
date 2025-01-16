HYPERS = {
    "model": "TransformerVAE",
    "embed_dim": 512,
    "num_heads": 16,
    "hidden_dim": 1024,
    "num_encoder_layers": 6,
    "num_decoder_layers": 12,
    "max_len": 125,
    "batch_size": 32,
    "accumulation_steps": 2,
    "dropout": 0.1,
    "num_steps": 500000,
    "learning_rate": 0.00001,
    "decay_rate": 0.9995,
    "beta_start": 0.0,
    "beta_end": 0.01,
    "beta_warmup_steps": 40000,
    "free_bits": 0.2,
    "lr_warmup_steps": 10000,
    "vocab_size": 20000,
    "test_set_portion": 0.05,
    "seed": 42,
    "name": "battlefield-promotion"
}
# Bond of Discipline
# Bulwark Giant
# Charmed Stray
# Defiant Strike
# Divine Arrow
# Enforcer Griffin
# Finale of Glory
# HYPERS = {
#     "model": "ADVAE",
#     "embed_dim": 256,
#     "latent_dim": 512,
#     "num_heads": 16,
#     "hidden_dim": 1024,
#     "num_encoder_layers": 6,
#     "num_decoder_layers": 12,
#     "max_len": 125,
#     "batch_size": 32,
#     "accumulation_steps": 2,
#     "dropout": 0.0,
#     "num_steps": 500000,
#     "learning_rate": 0.00001,
#     "decay_rate": 0.9995,
#     "beta_start": 0.0,
#     "beta_end": 0.01,
#     "beta_warmup_steps": 40000,
#     "free_bits": 0.2,
#     "lr_warmup_steps": 10000,
#     "vocab_size": 20000,
#     "test_set_portion": 0.05,
#     "seed": 42,
#     "name": "bond-of-discipline"
# }