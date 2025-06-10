import hydra
from omegaconf import DictConfig
from core.runner.runner import *



@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_for_data(config: DictConfig):
    episode_generation(config)
    return


if __name__ == "__main__":
    training_for_data()