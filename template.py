import os
from pathlib import Path

project_title = "depression_prediction"


project_files = [
    f"{project_title}/__init.py__",
    f"{project_title}/components/__init.py__",
    f"{project_title}/components/data_ingestion.py",
    f"{project_title}/components/data_validation.py",
    f"{project_title}/components/data_transformation.py",
    f"{project_title}/components/model_trainer.py",
    f"{project_title}/components/model_pusher.py",
    f"{project_title}/components/model_evaluation.py",
    f"{project_title}/configuration/__init.py__",
    f"{project_title}/constant/__init.py__",
    f"{project_title}/entity/__init.py__",
    f"{project_title}/entity/config_entity.py",
    f"{project_title}/entity/artifact_entity.py",
    f"{project_title}/exception/__init__.py",
    f"{project_title}/logger/__init__.py",
    f"{project_title}/pipeline/__init__.py",
    f"{project_title}/pipeline/training_pipeline.py",
    f"{project_title}/pipeline/prediction_pipeline.py",
    f"{project_title}/utils/__init__.py",
    f"{project_title}/utils/main_utils.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
   "config/model.yaml",
   "config/schema.yaml",
   "notebook/database_connect.ipynb",
   ".env"

  ]


# automatically create the project folders and files

for path_to_file in project_files:

    path_to_file = Path(path_to_file)
    folder_directory, filename = os.path.split(path_to_file)

    if folder_directory != "":
        os.makedirs(folder_directory, exist_ok=True)
    if (not os.path.exists(path_to_file)) or (os.path.getsize(path_to_file)==0):
        with open(path_to_file, "w") as file_obj:
            pass
    else:
        print(f"This file already exists in {path_to_file}")