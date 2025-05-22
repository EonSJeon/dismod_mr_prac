from abc import ABC, abstractmethod

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from src.utils import logit_to_prob, prob_to_logit


class Estimator(ABC):
    @abstractmethod
    def fit(self, data: pl.DataFrame):
        pass

    @abstractmethod
    def predict(self, age_start: int, age_end: int) -> float:
        pass


class SumEstimator(Estimator):
    def __init__(self, sub_estimators: list[Estimator]):
        self.sub_estimators = sub_estimators

    def fit(self, data: pl.DataFrame):
        raise ValueError("SumEstimator cannot be fitted.")

    def predict(self, age_start: int, age_end: int) -> float:
        return sum(
            sub_estimator.predict(age_start, age_end)
            for sub_estimator in self.sub_estimators
        )


class MeanAgeLinearRegressionEstimator(Estimator):
    def fit(self, data) -> pl.DataFrame:
        mean_age: NDArray = (
            data.select(["age_start", "age_end"]).to_numpy().mean(axis=1)
        )
        sample_size: NDArray = data.select("Sample size").to_numpy()
        patients_with_amd: NDArray = data.select("Patients with AMD").to_numpy()
        prevalence = patients_with_amd / sample_size
        logit_prevalence: NDArray = prob_to_logit(prevalence)
        self.linear_model = LinearRegression(n_jobs=1)
        self.linear_model.fit(mean_age.reshape(-1, 1), logit_prevalence)

    def predict(self, age_start: int, age_end: int) -> float:
        return logit_to_prob(
            self.linear_model.predict([[age_start + (age_end - age_start) / 2]]).item()
        )


if __name__ == "__main__":
    # 예시 데이터 생성
    data = pl.DataFrame({
        "age_start": [40, 50, 60, 70],
        "age_end": [49, 59, 69, 79],
        "Sample size": [1000, 1000, 1000, 1000],
        "Patients with AMD": [50, 100, 150, 200]
    })
    
    # Estimator 인스턴스 생성 및 학습
    estimator = MeanAgeLinearRegressionEstimator()
    estimator.fit(data)
    
    # 예측
    age_start = 45
    age_end = 55
    prediction = estimator.predict(age_start, age_end)
    print(f"Predicted prevalence for age {age_start}-{age_end}: {prediction:.4f}")
