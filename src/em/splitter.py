from numpy.typing import NDArray
from scipy.stats import pearsonr
from src.estimator import Estimator, MeanAgeLinearRegressionEstimator, SumEstimator
import itertools
import polars as pl
import numpy as np


LEAF_STAGES = ["Early", "Intermediate", "Late-dry", "Late-wet"]
ALL_STAGES = [
    "Early",
    "Intermediate",
    "Late-dry",
    "Late-wet",
    "Early to intermediate",
    "Late",
]

LEAF_KEYS = list(
    itertools.product(
        ["Male", "Female"], ["Early", "Intermediate", "Late-dry", "Late-wet"]
    )
)


class EMDataSplitter:
    estimators: dict[tuple[str, str], Estimator]

    def __init__(self, data: pl.DataFrame):
        self.orig_data: dict[tuple[str, str], pl.DataFrame] = data.partition_by(
            ["Sex", "Stage"], as_dict=True
        )
        self.data: dict[tuple[str, str], list[dict]] = {
            key: df.rows(named=True) for key, df in self.orig_data.items()
        }
        self.dataframes = {key: pl.DataFrame(rows) for key, rows in self.data.items()}
        self.estimators = {}

    def split_sex(self, stage: str) -> tuple[list[dict], list[dict]]:
        male_rows, female_rows = [], []
        for row in self.data[("NA", stage)]:
            male_estimated_p = self.estimators[("Male", stage)].predict(
                age_start=row["age_start"], age_end=row["age_end"]
            )
            female_estimated_p = self.estimators[("Female", stage)].predict(
                age_start=row["age_start"], age_end=row["age_end"]
            )

            # TODO: use GBD population statistics to estimate ratio
            male_population_ratio = 0.5
            male_patient_ratio = male_estimated_p / (
                male_estimated_p + female_estimated_p
            )

            male_row = row.copy()
            female_row = row.copy()

            male_row["Sex"] = "Male"
            male_row["Sample size"] *= male_population_ratio
            male_row["Patients with AMD"] *= male_patient_ratio

            female_row["Sex"] = "Female"
            female_row["Sample size"] *= 1 - male_population_ratio
            female_row["Patients with AMD"] *= 1 - male_patient_ratio

            male_rows.append(male_row)
            female_rows.append(female_row)

        return male_rows, female_rows

    def split_stage(
        self, sex: str, stage_from: str, stage_to: tuple[str, str]
    ) -> tuple[list[dict], list[dict]]:
        stage_to_1, stage_to_2 = stage_to
        left_rows, right_rows = [], []
        for row in self.data[(sex, stage_from)]:
            p_1 = self.estimators[(sex, stage_to_1)].predict(
                age_start=row["age_start"], age_end=row["age_end"]
            )
            p_2 = self.estimators[(sex, stage_to_2)].predict(
                age_start=row["age_start"], age_end=row["age_end"]
            )
            left_patient_ratio = p_1 / (p_1 + p_2)
            left_row = row.copy()
            right_row = row.copy()
            left_row["Patients with AMD"] *= left_patient_ratio
            right_row["Patients with AMD"] *= 1 - left_patient_ratio
            left_rows.append(left_row)
            right_rows.append(right_row)
        return left_rows, right_rows

    def EStep(self):
        self.data: dict[tuple[str, str], list[dict]] = {
            key: df.rows(named=True) for key, df in self.orig_data.items()
        }
        for stage in ALL_STAGES:
            male_rows, female_rows = self.split_sex(stage)
            self.data[("Male", stage)].extend(male_rows)
            self.data[("Female", stage)].extend(female_rows)
        for sex in ["Male", "Female"]:
            early, intermediate = self.split_stage(
                sex, "Early to intermediate", ("Early", "Intermediate")
            )
            late_dry, late_wet = self.split_stage(sex, "Late", ("Late-dry", "Late-wet"))
            self.data[(sex, "Early")].extend(early)
            self.data[(sex, "Intermediate")].extend(intermediate)
            self.data[(sex, "Late-dry")].extend(late_dry)
            self.data[(sex, "Late-wet")].extend(late_wet)
        # Convert data to Dataframe
        self.dataframes = {key: pl.DataFrame(rows) for key, rows in self.data.items()}

    def MStep(self):
        # M1: Estimate regression models for each leaf node
        for key in LEAF_KEYS:
            self.estimators[key] = MeanAgeLinearRegressionEstimator()
            self.estimators[key].fit(self.dataframes[key])
        # M2: Estimate regression models for each non-leaf node
        for sex in ["Male", "Female"]:
            self.estimators[(sex, "Early to intermediate")] = SumEstimator(
                sub_estimators=[
                    self.estimators[(sex, "Early")],
                    self.estimators[(sex, "Intermediate")],
                ]
            )
            self.estimators[(sex, "Late")] = SumEstimator(
                sub_estimators=[
                    self.estimators[(sex, "Late-dry")],
                    self.estimators[(sex, "Late-wet")],
                ]
            )

    def predict(self, sex, stage, age_start, age_end):
        return self.estimators[(sex, stage)].predict(age_start, age_end)
