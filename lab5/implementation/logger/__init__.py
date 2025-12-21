import logging
import logging.config
import yaml

def setup_logger_from_yaml(path="logging_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
    return logging.getLogger()