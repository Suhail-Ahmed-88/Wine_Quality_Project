# import os
# import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from urllib.parse import urlparse
# import mlflow
# import mlflow.sklearn
# import numpy as np
# import joblib
# from mlProject.entity.config_entity import ModelEvaluationConfig
# from mlProject.utils.common import save_json
# from pathlib import Path


# class ModelEvaluation:
#     def __init__(self, config: ModelEvaluationConfig):
#         self.config = config

    
#     def eval_metrics(self,actual, pred):
#         rmse = np.sqrt(mean_squared_error(actual, pred))
#         mae = mean_absolute_error(actual, pred)
#         r2 = r2_score(actual, pred)
#         return rmse, mae, r2
    


#     def log_into_mlflow(self):

#         test_data = pd.read_csv(self.config.test_data_path)
#         model = joblib.load(self.config.model_path)

#         test_x = test_data.drop([self.config.target_column], axis=1)
#         test_y = test_data[[self.config.target_column]]


#         mlflow.set_tracking_uri(self.config.mlflow_uri)
#         tracking_url_type_store = urlparse(self.config.mlflow_uri).scheme



#         with mlflow.start_run():

#             predicted_qualities = model.predict(test_x)

#             (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
#             # Saving metrics as local
#             scores = {"rmse": rmse, "mae": mae, "r2": r2}
#             save_json(path=Path(self.config.metric_file_name), data=scores)

#             mlflow.log_params(self.config.all_params)

#             mlflow.log_metric("rmse", rmse)
#             mlflow.log_metric("r2", r2)
#             mlflow.log_metric("mae", mae)


#             # Model registry does not work with file store
#             if tracking_url_type_store != "file":

#                 # Register the model
#                 # There are other ways to use the Model Registry, which depends on the use case,
#                 # please refer to the doc for more information:
#                 # https://mlflow.org/docs/latest/model-registry.html#api-workflow
#                 mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
#             else:
#                 mlflow.sklearn.log_model(model, "model")

    



import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path
from dotenv import load_dotenv


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        # ✅ Load environment variables from .env file
        load_dotenv()

        # ✅ Set tracking URI from .env or config
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", self.config.mlflow_uri))

        # ✅ Set auth (if using DagsHub or remote MLflow)
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # ✅ Detect URI scheme (file, http, https)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Save metrics to local JSON
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log to MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # ✅ Log model with/without registry
            if tracking_url_type_store in ["http", "https"]:
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")

