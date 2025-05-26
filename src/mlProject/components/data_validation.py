import os
import pandas as pd
from mlProject import logger
from mlProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            logger.info("Starting data validation process...")
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()
            validation_status = True

            # Check for extra columns not in schema
            for col in all_cols:
                if col not in all_schema:
                    logger.error(f"Unexpected column in data: {col}")
                    validation_status = False

            # Check for missing columns
            for col in all_schema:
                if col not in all_cols:
                    logger.error(f"Missing expected column: {col}")
                    validation_status = False

            # Optional: Check for null values
            if data.isnull().sum().any():
                logger.warning("Data contains null/missing values.")
                validation_status = False

            # Write validation status to file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation Status: {validation_status}")

            logger.info(f"Validation completed with status: {validation_status}")
            return validation_status
        except Exception as e:
            logger.exception("Exception occurred during data validation.")
            raise e
