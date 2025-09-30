# src/cnnClassifier/pipeline/stage_03_multi_task_model_training.py
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.multi_task_model_trainer import MultiTaskModelTrainer
from cnnClassifier import logger

STAGE_NAME = "Multi-Task Model Training stage"

class MultiTaskModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        multi_task_model_trainer_config = config.get_multi_task_model_trainer_config()
        trainer = MultiTaskModelTrainer(config=multi_task_model_trainer_config)
        trainer.train()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = MultiTaskModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e