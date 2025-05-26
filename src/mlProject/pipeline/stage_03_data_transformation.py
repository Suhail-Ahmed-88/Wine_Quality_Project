from pathlib import Path
from mlProject import logger
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_transformation import DataTransformation

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
  def __init__(self):
    pass  

  def main(self):
    try:
      with open(Path("artifacts/data_validation/status.txt"), "r") as f:
        status = f.read().split(" ")[-1].strip()

      
      if status == "True":
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config = data_transformation_config)
        data_transformation.train_test_spliting()
      else:
        raise Exception("Your data schema is not valid ")
        
    except Exception as e:
      print(e)
      
      
if __name__ == '__main__':
  try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
  except Exception as e:
    logger.exception(e)
    raise e
  
 
   
# from pathlib import Path
# from mlProject import logger
# from mlProject.config.configuration import ConfigurationManager
# from mlProject.components.data_transformation import DataTransformation

# STAGE_NAME = "Data Transformation Stage"

# class DataTransformationTrainingPipeline:
#   def __init__(self):
#     pass  

#   def main(self):
#     try:
#       with open(Path("artifacts/data_validation/status.txt"), "r") as f:
#         status = f.read().split(" ")[-1].strip()

      
#       if status == "True":
        
#         config_manager = ConfigurationManager()

#         # Step 2: Get Data Transformation Config
#         data_transform_config = config_manager.get_data_transformation_config()

#         # Step 3: Run Data Transformation
#         data_transformer = DataTransformation(config=data_transform_config)
#         data_transformer.train_test_spliting()

#         logger.info("Data Transformation pipeline completed successfully.")
        
#       else:
#         raise Exception("Your data schema is not valid ")
        
#     except Exception as e:
#       print(e)
      
      
# if __name__ == '__main__':
#   try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = DataTransformationTrainingPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#   except Exception as e:
#     logger.exception(e)
#     raise e