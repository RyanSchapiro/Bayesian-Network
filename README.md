# Type 2 Diabetes – Risk Assessment & Management Decision Network

A Python project that builds a **Bayesian Network** (BN) for 5-year type-2 diabetes risk and an **Influence Diagram** (ID) for management decisions (lifestyle, medication, monitoring). It provides an **interactive CLI**, prints an **Expected Utilities** policy table, and exports **Graphviz** diagrams and BIFXML model files.

---

## Features

- **Bayesian Network** of key factors:
  `Age, FamilyHistory, PhysicalActivity, DietQuality, Smoking, BMI, BloodPressure, Cholesterol, InsulinResistance, FastingGlucose, HbA1c, MetabolicSyndrome, DiabetesRisk`.
- **Influence Diagram** decisions:
  - `LifestyleIntervention`: *None, Diet, Exercise, Combined*
  - `MedicalIntervention`: *None, Metformin, Intensive*
  - `MonitoringFrequency`: *Annual, Biannual, Quarterly, Monthly*
  - Outcome: `FutureDiabetesStatus` → utilities: **Health**, **Cost**, **Burden**
- **Literature-inspired effects & utilities** (DPP-style risk reductions; scaled health utilities; simple cost/burden penalties).
- **Artifacts saved to `./models/`**:
  - `diabetes_bn.bifxml`, `diabetes_id.bifxml`
  - `diabetes_bn.dot`, `diabetes_id.dot`
  - `diabetes_bn.png`, `diabetes_id.png` *(if Graphviz `dot` is installed)*.
- **Interactive CLI**:
  - Assess a patient
  - Run predefined test cases
  - Run guideline-style validation scenarios

---

## Project Structure

```
.
├─ models/
│  ├─ diabetes_bn.bifxml
│  ├─ diabetes_id.bifxml
│  ├─ diabetes_bn.dot
│  ├─ diabetes_bn.png         (if Graphviz installed)
│  ├─ diabetes_id.dot
│  ├─ diabetes_id.png         (if Graphviz installed)
└─ diabetes_decision_network.py   # main script (your filename may differ)
```
> The script creates `models/` automatically.

---

## Requirements

- **Python** 3.9+
- **Packages**: `pyagrum`, `numpy`
- **Optional** (for PNG diagrams): **Graphviz** command-line tools (`dot`) on your PATH

### Installation

```bash
# (recommended) create & activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# install dependencies
pip install pyagrum numpy
```

**Graphviz (optional for PNG rendering):**
- Windows:  `winget install --id Graphviz.Graphviz -e`
- macOS:    `brew install graphviz`
- Ubuntu:   `sudo apt-get update && sudo apt-get install -y graphviz`

If `dot` is unavailable, the script still runs and saves DOT files- PNG generation is skipped.

---

## How to Run

```bash
python BN.py
```

On start the script will:

1. Build the **Bayesian Network** and fill CPTs.  
2. Build the **Influence Diagram** and fill CPTs/utilities.  
3. Save BN/ID to **BIFXML** + **DOT** (and **PNG** if Graphviz present).  
4. Print an **Expected Utilities** table for predefined risk scenarios.  
5. Show the **interactive menu**:

```
[1] Assess patient risk
[2] Run predefined test cases
[3] Run guideline validation
[4] Exit
```

### Menu Details

- **[1] Assess patient risk**  
  Prompts for: `Age, FamilyHistory, PhysicalActivity, DietQuality, Smoking, BMI`, optional `HbA1c`.  
  Outputs:
  - Posterior probabilities for **DiabetesRisk**, **InsulinResistance**, **MetabolicSyndrome**
  - Recommended **Lifestyle/Medical/Monitoring** strategy using risk posteriors and “what-if” lifestyle improvements.

- **[2] Predefined test cases**  
  Runs three illustrative profiles and prints results.

- **[3] Guideline validation**  
  Runs eight ADA-style scenarios for quick face-validity and prints results.

---

## Modeling Overview

- **Qualitative → quantitative mappings** shape CPTs for BMI, BP, Cholesterol, InsulinResistance, FastingGlucose, HbA1c using intuitive risk/benefit scores (normalized to valid probabilities).
- **DiabetesRisk** (Low/Moderate/High/VeryHigh) depends on **InsulinResistance**, **MetabolicSyndrome**, **Age**, **FamilyHistory**, **HbA1c**.
- **FutureDiabetesStatus** (NoDiabetes, Controlled, Uncontrolled) applies baseline risk and **intervention effects**:
  - Lifestyle reduction (e.g., *Combined* ≈ 58%)
  - Medical reduction (e.g., *Metformin* ≈ 31%)
  - **Combined effect**: `1 - (1 - lifestyle)(1 - medical)` (multiplicative complement)
- **Utilities**:
  - **HealthUtility**: scaled values for health states (higher is better)
  - **CostUtility**: penalties for lifestyle program costs, medication, and monitoring frequency
  - **BurdenUtility**: disutility for adherence/time/side-effects  
  - **Total Utility** reported as:  
    `Expected(HealthUtility) + CostUtility + BurdenUtility`

Numbers are **illustrative** and easy to edit for local calibration.

---

## Customization

- **Priors**: edit `Age`, `FamilyHistory`, `PhysicalActivity`, `DietQuality`, `Smoking` distributions.
- **CPT tuning**:
  - `BMI | (PhysicalActivity, DietQuality, Age)` via `health_score` → `p_normal`, `p_obese`
  - `BloodPressure`, `Cholesterol`, `InsulinResistance`, `FastingGlucose`, `HbA1c` risk mappings
- **Intervention effects**: update `lifestyle_reduction` and `medical_reduction`; adjust composition rule if needed.
- **Utilities**:
  - `HealthUtility` values per `FutureDiabetesStatus`
  - `CostUtility` dictionaries (lifestyle/medical/monitoring)
  - `BurdenUtility` dictionaries (lifestyle/medical/monitoring)
- **Recommendation thresholds**: modify heuristics in `run_assessment()`.

---

## Outputs

- **Console**
  - Expected Utilities table across `{Low, Moderate, High, VeryHigh}` risk vs `{None, Diet, Combined, Metformin, Combined+Metformin}`
  - Optimal strategy per risk stratum
  - For interactive cases: posteriors and recommended plan
- **Files** (`./models/`)
  - `diabetes_bn.bifxml`, `diabetes_id.bifxml`
  - `diabetes_bn.dot`, `diabetes_id.dot`
  - `diabetes_bn.png`, `diabetes_id.png` (if Graphviz available)

---
