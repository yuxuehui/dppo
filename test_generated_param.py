import hydra
from omegaconf import DictConfig
from core.runner.runner import *



@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_for_data(config: DictConfig):
    train_for_generated_parameter(config)
    return


if __name__ == "__main__":
    training_for_data()