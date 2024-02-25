import pandas as pd
from sklearn import datasets
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import config

reference = pd.read_csv(config.CURRENT)
current = pd.read_csv(config.SOURCE)

data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=reference, current_data=current)
report = data_drift_report.as_dict()
drift_detected = report["metrics"][0]["result"]["dataset_drift"]
if drift_detected:
    print("Detect dataset drift")
else:
    print("Detect no dataset drift")