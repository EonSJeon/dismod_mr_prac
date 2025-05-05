from dismod_mr import data, model, fit, plot
from dismod_mr.data import load

'''
* **i**: Incidence (발생률)
  인구 1인년(person-year)당 새로 생기는 사례의 수 ([IHME][1])

* **p**: Prevalence (유병률)
  특정 시점에 인구 중 상태를 보유하고 있는 사례의 비율(비례) ([IHME][1])

* **r**: Remission (관해율)
  인구 1인년당 완치되거나 상태에서 벗어나는 사례의 수 ([IHME][1])

* **f**: Excess mortality rate (과잉 사망률)
  유병 사례 중 일반 인구 대비 초과 사망 수(인구 1인년당) ([IHME][1])

* **rr**: Relative risk (상대위험률)
  질병군의 사망률을 비질병군 사망률로 나눈 비율(비율비) ([IHME][1])

* **pf**: Proportion (비율)
  전체 중 특정 특성을 보이는 사례의 비례(예: HIV 중 성접촉 경로 비율) ([IHME][1])

* **m**: With-condition mortality rate (상태-유발 사망률)
  유병 사례 전체에서 관찰된 사망 수(인구 1인년당); 과잉 사망률과 원인별 사망률의 합과 동일 ([IHME][1])

* **X**: Continuous (연속형 지표)
  혈압·BMI 등 연속 스케일로 측정되는 인구 특성 값 ([IHME][1])

* **csmr**: Cause-specific mortality rate (원인별 사망률)
  인구 전체에서 해당 질환으로 인한 사망 수(인구 1인년당); 유병률×과잉 사망률과 동일 ([IHME][1])
  '''