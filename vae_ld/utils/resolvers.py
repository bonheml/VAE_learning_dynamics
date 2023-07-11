from omegaconf import OmegaConf

OmegaConf.register_new_resolver("range", lambda x: [a for a in range(x)])