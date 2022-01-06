#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track carbon emissions of code runs.

Author:
    Erik Johannes Husom

Created:
    2022-01-05 Wednesday 10:25:52 

Notes:
    

"""
import subprocess

from codecarbon import track_emissions

COUNTRY = "NOR"

@track_emissions(offline=True, country_iso_code=COUNTRY, project_name="Erdre")
def run_erdre():
    subprocess.run(["dvc", "repro", "--force"], check=True)


@track_emissions(offline=True, country_iso_code=COUNTRY, project_name="profile_stage")
def run_profile_stage():
    subprocess.run(["python3", "src/profiling.py", "assets/data/raw/"], check=True)


@track_emissions(offline=True, country_iso_code=COUNTRY, project_name="clean_stage")
def run_clean_stage():
    subprocess.run(["python3", "src/clean.py", "assets/data/raw/"], check=True)


@track_emissions(offline=True, country_iso_code=COUNTRY, project_name="featurize_stage")
def run_featurize_stage():
    subprocess.run(["python3", "src/featurize.py", "assets/data/cleaned/"], check=True)


@track_emissions(offline=True, country_iso_code=COUNTRY, project_name="split_stage")
def run_split_stage():
    subprocess.run(["python3", "src/split.py", "assets/data/featurized/"], check=True)


@track_emissions(offline=True, country_iso_code=COUNTRY, project_name="scale_stage")
def run_scale_stage():
    subprocess.run(["python3", "src/scale.py", "assets/data/split/"], check=True)


@track_emissions(
    offline=True, country_iso_code=COUNTRY, project_name="sequentialize_stage"
)
def run_sequentialize_stage():
    subprocess.run(
        ["python3", "src/sequentialize.py", "assets/data/scaled/"], check=True
    )


@track_emissions(offline=True, country_iso_code=COUNTRY, project_name="combine_stage")
def run_combine_stage():
    subprocess.run(
        ["python3", "src/combine.py", "assets/data/sequentialized/"], check=True
    )


@track_emissions(offline=True, country_iso_code=COUNTRY, project_name="train_stage")
def run_train_stage():
    subprocess.run(
        ["python3", "src/train.py", "assets/data/combined/train.npz"], check=True
    )


@track_emissions(offline=True, country_iso_code=COUNTRY, project_name="evaluate_stage")
def run_evaluate_stage():
    subprocess.run(
        [
            "python3",
            "src/evaluate.py",
            "assets/models/model.h5",
            "assets/data/combined/train.npz",
            "assets/data/combined/test.npz",
            "assets/data/combined/calibrate.npz",
        ],
        check=True,
    )

def run_all_stages():

    run_profile_stage()
    run_clean_stage()
    run_featurize_stage()
    run_split_stage()
    run_scale_stage()
    run_sequentialize_stage()
    run_combine_stage()
    run_train_stage()
    run_evaluate_stage()


def run_pipeline(stage: str = "all", use_dvc: bool = True, force: bool = False):

    # Force DVC usage
    use_dvc = True

    # Set project name according to the stage that is run
    PROJECT_NAME = f"stage_{stage}"

    command = []

    if use_dvc:
        command.append("dvc")
        command.append("repro")

    if stage != "all":
        command.append(stage)

    if force:
        command.append("--force")


    # Function for running command is wrapped to be able to specify project
    # name based on which stage is being run
    @track_emissions(offline=True, country_iso_code=COUNTRY, project_name=PROJECT_NAME)
    def _run_command(command):
        """Run command and track emissions using CodeCarbon.

        Args:
            command (list): A command to be run at the command line, where each
                word is a separate item in the list.

        """

        subprocess.run(command, check=True)

    _run_command(command)


if __name__ == "__main__":

    # run_erdre()
    # run_all_stages()
    run_pipeline("all", force=True)
