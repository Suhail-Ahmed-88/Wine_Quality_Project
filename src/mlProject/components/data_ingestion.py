import os
import zipfile
from urllib import request
from pathlib import Path
from mlProject import logger  # adjust this import as per your project structure
from mlProject.utils.common import get_size  # make sure this is where your `get_size` is
from mlProject.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
    
        if not zipfile.is_zipfile(self.config.local_data_file):
            raise zipfile.BadZipFile("The downloaded file is not a valid zip file.")

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        logger.info(f"Extracted zip file to: {unzip_path}")