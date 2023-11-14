import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def process_training(cfg : DictConfig):
    print (cfg)

if __name__ == '__main__':
    process_training()