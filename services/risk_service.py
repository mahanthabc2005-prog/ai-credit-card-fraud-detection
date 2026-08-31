from dataclasses import dataclass


@dataclass(frozen=True)
class RiskResult:
    level: str
    recommendation: str


def classify_risk(probability, low=0.30, high=0.70):
    if probability < low:
        return RiskResult("LOW", "Approve only after normal monitoring. This is an ML risk estimate.")
    if probability <= high:
        return RiskResult("MEDIUM", "Review transaction context before approval.")
    return RiskResult("HIGH", "Hold and investigate before approval where policy permits.")
