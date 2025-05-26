# import os 
# from mlProject import logger
# from sklearn.model_selection import train_test_split
# import pandas as pd
# from mlProject.entity.config_entity import DataTransformationConfig
# class DataTransformation:
#     def __init__(self, config: DataTransformationConfig):
#         self.config = config
        
#     ## Note: you can add different data transformation techniques such as Scaler, PCA and all
#     ## you can perform all kinds of EDA in ML cycle here before passing this data to the model
    
#     # I am only adding train_test_spliting  because this data is already cleaned
    
#     def train_test_spliting(self):
#         data = pd.read_csv(self.config.data_path)
        
#         # Split the data into training and test sets. (0.75, 0.25) split.
#         train, test = train_test_split(data)
        
#         train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index = False)
#         test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index = False)
        
#         logger.info("Splited data into training and test sets")
#         logger.info(train.shape)
#         logger.info(test.shape)
        
#         print(train.shape)
#         print(test.shape)

import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        # Step 1: Read Data
        data = pd.read_csv(self.config.data_path)
        logger.info("Data loaded successfully")

        # Step 2: Basic EDA
        logger.info(f"Data Head:\n{data.head()}")
        logger.info(f"Data Summary:\n{data.describe()}")
        logger.info(f"Missing Values:\n{data.isnull().sum()}")

        # Step 3: Drop missing values (or you can fill them)
        data = data.dropna()
        logger.info("Missing values dropped")

        # Step 4: Separate features and target (assume last column is target)
        X = data.iloc[:, :-1]   # All columns except last
        y = data.iloc[:, -1]    # Last column (target)

        # Step 5: Scale the features using MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Features scaled using MinMaxScaler")

        # Step 6: Apply PCA (reduce to 2 components)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        logger.info("PCA applied, reduced to 2 features")

        # Step 7: Combine X_pca and target again
        df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
        df_pca["target"] = y.reset_index(drop=True)

        # Step 8: Split the data into training and test sets
        train, test = train_test_split(df_pca, test_size=0.25, random_state=42)

        # Step 9: Save the train and test datasets
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splitted and saved train/test data successfully")
        logger.info(f"Train Shape: {train.shape}")
        logger.info(f"Test Shape: {test.shape}")

        print(train.shape)
        print(test.shape)
