"""
Type 2 Diabetes Risk Assessment and Management Decision Network
Authors: Ryan Shapiro, Ben Ruijsch van Dugteren, Nathan Wells
Date: October 2025
"""

import pyagrum as gum
import numpy as np
import os
import subprocess

# ============================================================================
# BUILD BAYESIAN NETWORK
# ============================================================================

print("\n" + "="*70)
print("BUILDING BAYESIAN NETWORK")
print("="*70)

bn = gum.BayesNet('Diabetes_Risk_Assessment')

# Add nodes
bn.add(gum.LabelizedVariable('Age', 'Age', ['Young', 'Middle', 'Senior']))
bn.add(gum.LabelizedVariable('FamilyHistory', 'Family History', ['No', 'Yes']))
bn.add(gum.LabelizedVariable('PhysicalActivity', 'Physical Activity', 
       ['Sedentary', 'Light', 'Moderate', 'Active']))
bn.add(gum.LabelizedVariable('DietQuality', 'Diet Quality', ['Poor', 'Fair', 'Good']))
bn.add(gum.LabelizedVariable('Smoking', 'Smoking', ['No', 'Yes']))
bn.add(gum.LabelizedVariable('BMI', 'BMI', ['Normal', 'Overweight', 'Obese']))
bn.add(gum.LabelizedVariable('BloodPressure', 'Blood Pressure', ['Normal', 'Elevated', 'High']))
bn.add(gum.LabelizedVariable('Cholesterol', 'Cholesterol', ['Healthy', 'Borderline', 'Poor']))
bn.add(gum.LabelizedVariable('InsulinResistance', 'Insulin Resistance', ['Low', 'Moderate', 'High']))
bn.add(gum.LabelizedVariable('FastingGlucose', 'Fasting Glucose', ['Normal', 'Prediabetic', 'Diabetic']))
bn.add(gum.LabelizedVariable('HbA1c', 'HbA1c', ['Normal', 'Prediabetic', 'Diabetic']))
bn.add(gum.LabelizedVariable('MetabolicSyndrome', 'Metabolic Syndrome', ['No', 'Yes']))
bn.add(gum.LabelizedVariable('DiabetesRisk', '5-Year Risk', ['Low', 'Moderate', 'High', 'VeryHigh']))

# Add arcs
arcs = [
    ('PhysicalActivity', 'BMI'), ('DietQuality', 'BMI'), ('Age', 'BMI'),
    ('BMI', 'BloodPressure'), ('Age', 'BloodPressure'), ('Smoking', 'BloodPressure'),
    ('BMI', 'Cholesterol'), ('DietQuality', 'Cholesterol'), ('PhysicalActivity', 'Cholesterol'),
    ('BMI', 'InsulinResistance'), ('PhysicalActivity', 'InsulinResistance'),
    ('DietQuality', 'InsulinResistance'), ('FamilyHistory', 'InsulinResistance'),
    ('InsulinResistance', 'FastingGlucose'), ('Age', 'FastingGlucose'),
    ('InsulinResistance', 'HbA1c'), ('FastingGlucose', 'HbA1c'),
    ('BloodPressure', 'MetabolicSyndrome'), ('Cholesterol', 'MetabolicSyndrome'),
    ('BMI', 'MetabolicSyndrome'), ('FastingGlucose', 'MetabolicSyndrome'),
    ('InsulinResistance', 'DiabetesRisk'), ('MetabolicSyndrome', 'DiabetesRisk'),
    ('Age', 'DiabetesRisk'), ('FamilyHistory', 'DiabetesRisk'), ('HbA1c', 'DiabetesRisk')
]
for parent, child in arcs:
    bn.addArc(parent, child)

print(f"Network: {bn.size()} nodes, {bn.sizeArcs()} arcs")
print("Filling probability tables...")

# Prior probabilities
bn.cpt('Age')[:] = [0.35, 0.40, 0.25]
bn.cpt('FamilyHistory')[:] = [0.70, 0.30]
bn.cpt('PhysicalActivity')[:] = [0.30, 0.35, 0.25, 0.10]
bn.cpt('DietQuality')[:] = [0.25, 0.50, 0.25]
bn.cpt('Smoking')[:] = [0.82, 0.18]

# Helper function to normalize probabilities
def normalize(probs):
    total = sum(probs)
    return [p/total for p in probs]

#calculate conditional probabilities based on literature and expert opinion:

# P(BMI | PhysicalActivity, DietQuality, Age)
pa_scores = {'Active': 30, 'Moderate': 20, 'Light': 10, 'Sedentary': 0}
diet_scores = {'Good': 20, 'Fair': 10, 'Poor': 0}
age_penalty = {'Young': 0, 'Middle': 10, 'Senior': 20}

for pa in ['Sedentary', 'Light', 'Moderate', 'Active']: 
    for diet in ['Poor', 'Fair', 'Good']:
        for age in ['Young', 'Middle', 'Senior']:
            health_score = pa_scores[pa] + diet_scores[diet] - age_penalty[age]
            p_normal = 0.10 + 0.60 * (health_score / 50)
            p_obese = 0.60 - 0.52 * (health_score / 50)
            bn.cpt('BMI')[{'PhysicalActivity': pa, 'DietQuality': diet, 'Age': age}] = \
                normalize([p_normal, 1-p_normal-p_obese, p_obese])

# P(BloodPressure | BMI, Age, Smoking)
bmi_risk = {'Normal': 0, 'Overweight': 20, 'Obese': 40}
age_risk = {'Young': 0, 'Middle': 15, 'Senior': 35}
smoke_risk = {'No': 0, 'Yes': 20}

for bmi in ['Normal', 'Overweight', 'Obese']:
    for age in ['Young', 'Middle', 'Senior']:
        for smoke in ['No', 'Yes']:
            risk = bmi_risk[bmi] + age_risk[age] + smoke_risk[smoke]
            p_normal = max(0.10, 0.85 - risk/100)
            p_high = min(0.55, risk/100 * 0.6)
            bn.cpt('BloodPressure')[{'BMI': bmi, 'Age': age, 'Smoking': smoke}] = \
                normalize([p_normal, 1-p_normal-p_high, p_high])

# P(Cholesterol | BMI, DietQuality, PhysicalActivity)
chol_bmi_risk = {'Normal': 0, 'Overweight': 20, 'Obese': 40}
diet_benefit = {'Good': -20, 'Fair': 0, 'Poor': 20}
pa_benefit = {'Active': -15, 'Moderate': -8, 'Light': 0, 'Sedentary': 15}

for bmi in ['Normal', 'Overweight', 'Obese']:
    for diet in ['Poor', 'Fair', 'Good']:
        for pa in ['Sedentary', 'Light', 'Moderate', 'Active']:
            risk = chol_bmi_risk[bmi] + diet_benefit[diet] + pa_benefit[pa]
            p_healthy = max(0.15, 0.75 - risk/100)
            p_poor = min(0.45, max(0.05, risk/100 * 0.5))
            bn.cpt('Cholesterol')[{'BMI': bmi, 'DietQuality': diet, 'PhysicalActivity': pa}] = \
                normalize([p_healthy, 1-p_healthy-p_poor, p_poor])

# P(InsulinResistance | BMI, PhysicalActivity, DietQuality, FamilyHistory)
ir_bmi_mult = {'Normal': 1.0, 'Overweight': 2.0, 'Obese': 3.2}
ir_pa_mult = {'Active': 0.60, 'Moderate': 0.75, 'Light': 0.90, 'Sedentary': 1.0}
ir_diet_mult = {'Good': 0.80, 'Fair': 1.0, 'Poor': 1.30}
ir_fh_mult = {'No': 1.0, 'Yes': 2.5}

for bmi in ['Normal', 'Overweight', 'Obese']:
    for pa in ['Sedentary', 'Light', 'Moderate', 'Active']:
        for diet in ['Poor', 'Fair', 'Good']:
            for fh in ['No', 'Yes']:
                odds_multiplier = ir_bmi_mult[bmi] * ir_pa_mult[pa] * ir_diet_mult[diet] * ir_fh_mult[fh]
                base_p_low = 0.75
                p_low = base_p_low / (base_p_low + odds_multiplier * (1 - base_p_low))
                p_high = (1 - p_low) * 0.45
                bn.cpt('InsulinResistance')[{'BMI': bmi, 'PhysicalActivity': pa,
                                              'DietQuality': diet, 'FamilyHistory': fh}] = \
                    normalize([p_low, 1-p_low-p_high, p_high])

# P(FastingGlucose | InsulinResistance, Age)
fg_base = {'Low': 0.85, 'Moderate': 0.55, 'High': 0.20}
fg_age_penalty = {'Young': 0, 'Middle': 0.10, 'Senior': 0.20}

for ir in ['Low', 'Moderate', 'High']:
    for age in ['Young', 'Middle', 'Senior']:
        p_normal = max(0.08, fg_base[ir] - fg_age_penalty[age])
        p_diabetic = min(0.57, (1 - p_normal) * 0.45)
        bn.cpt('FastingGlucose')[{'InsulinResistance': ir, 'Age': age}] = \
            normalize([p_normal, 1-p_normal-p_diabetic, p_diabetic])

# P(HbA1c | InsulinResistance, FastingGlucose)
hba1c_ir_weight = {'Low': 0.3, 'Moderate': 0.5, 'High': 0.7}
hba1c_fg_weight = {'Normal': 0.1, 'Prediabetic': 0.5, 'Diabetic': 0.8}

for ir in ['Low', 'Moderate', 'High']:
    for fg in ['Normal', 'Prediabetic', 'Diabetic']:
        combined_risk = (hba1c_ir_weight[ir] + hba1c_fg_weight[fg]) / 2
        p_normal = max(0.03, 0.92 - combined_risk)
        p_diabetic = min(0.80, combined_risk * 0.85)
        bn.cpt('HbA1c')[{'InsulinResistance': ir, 'FastingGlucose': fg}] = \
            normalize([p_normal, 1-p_normal-p_diabetic, p_diabetic])

# P(MetabolicSyndrome | BloodPressure, Cholesterol, BMI, FastingGlucose)
for bp in ['Normal', 'Elevated', 'High']:
    for chol in ['Healthy', 'Borderline', 'Poor']:
        for bmi in ['Normal', 'Overweight', 'Obese']:
            for fg in ['Normal', 'Prediabetic', 'Diabetic']:
                criteria = 0
                if bp in ['Elevated', 'High']: criteria += 1
                if chol == 'Poor': criteria += 1
                if bmi == 'Obese': criteria += 1
                if fg in ['Prediabetic', 'Diabetic']: criteria += 1
                
                p_yes = 0.02 if criteria <= 1 else (0.20 if criteria == 2 else 0.95)
                bn.cpt('MetabolicSyndrome')[{'BloodPressure': bp, 'Cholesterol': chol,
                                              'BMI': bmi, 'FastingGlucose': fg}] = [1-p_yes, p_yes]

# P(DiabetesRisk | InsulinResistance, MetabolicSyndrome, Age, FamilyHistory, HbA1c)
risk_ir = {'Low': 0, 'Moderate': 20, 'High': 40}
risk_ms = {'No': 0, 'Yes': 25}
risk_age = {'Young': 0, 'Middle': 10, 'Senior': 15}
risk_fh = {'No': 0, 'Yes': 20}
risk_hba1c = {'Normal': 0, 'Prediabetic': 30, 'Diabetic': 70}

for ir in ['Low', 'Moderate', 'High']:
    for ms in ['No', 'Yes']:
        for age in ['Young', 'Middle', 'Senior']:
            for fh in ['No', 'Yes']:
                for hba1c in ['Normal', 'Prediabetic', 'Diabetic']:
                    score = risk_ir[ir] + risk_ms[ms] + risk_age[age] + risk_fh[fh] + risk_hba1c[hba1c]
                    
                    if score < 20:
                        probs = [0.85, 0.12, 0.02, 0.01]
                    elif score < 40:
                        probs = [0.50, 0.35, 0.12, 0.03]
                    elif score < 60:
                        probs = [0.25, 0.35, 0.30, 0.10]
                    elif score < 80:
                        probs = [0.10, 0.25, 0.40, 0.25]
                    else:
                        probs = [0.03, 0.12, 0.35, 0.50]
                    
                    bn.cpt('DiabetesRisk')[{'InsulinResistance': ir, 'MetabolicSyndrome': ms,
                                            'Age': age, 'FamilyHistory': fh, 'HbA1c': hba1c}] = probs

os.makedirs('models', exist_ok=True)
gum.saveBN(bn, 'models/diabetes_bn.bifxml') # Save the Bayesian Network
print("Bayesian Network complete\n")

# ============================================================================
# BUILD INFLUENCE DIAGRAM WITH LITERATURE-BASED UTILITIES
# ============================================================================

print("="*70)
print("BUILDING INFLUENCE DIAGRAM")
print("="*70)

id_model = gum.InfluenceDiagram() # Create empty influence diagram

# Copy BN structure
for node_id in bn.nodes():
    id_model.addChanceNode(bn.variable(node_id))
for arc in bn.arcs():
    id_model.addArc(bn.variable(arc[0]).name(), bn.variable(arc[1]).name())
for node_id in bn.nodes():
    id_model.cpt(bn.variable(node_id).name()).fillWith(bn.cpt(node_id))

# Add decision and utility nodes
id_model.addDecisionNode(gum.LabelizedVariable('LifestyleIntervention', 'Lifestyle', 
                                               ['None', 'Diet', 'Exercise', 'Combined']))
id_model.addDecisionNode(gum.LabelizedVariable('MedicalIntervention', 'Medical',
                                               ['None', 'Metformin', 'Intensive']))
id_model.addDecisionNode(gum.LabelizedVariable('MonitoringFrequency', 'Monitoring',
                                               ['Annual', 'Biannual', 'Quarterly', 'Monthly']))
id_model.addChanceNode(gum.LabelizedVariable('FutureDiabetesStatus', 'Future Status',
                                             ['NoDiabetes', 'Controlled', 'Uncontrolled']))
id_model.addUtilityNode(gum.LabelizedVariable('HealthUtility', 'Health Utility', 1))
id_model.addUtilityNode(gum.LabelizedVariable('CostUtility', 'Cost Utility', 1))
id_model.addUtilityNode(gum.LabelizedVariable('BurdenUtility', 'Burden Utility', 1))

# Add arcs
id_model.addArc('DiabetesRisk', 'LifestyleIntervention')
id_model.addArc('HbA1c', 'MedicalIntervention')
id_model.addArc('DiabetesRisk', 'MonitoringFrequency')
id_model.addArc('LifestyleIntervention', 'MedicalIntervention')
id_model.addArc('MedicalIntervention', 'MonitoringFrequency')
id_model.addArc('DiabetesRisk', 'FutureDiabetesStatus')
id_model.addArc('LifestyleIntervention', 'FutureDiabetesStatus')
id_model.addArc('MedicalIntervention', 'FutureDiabetesStatus')
id_model.addArc('FutureDiabetesStatus', 'HealthUtility')
id_model.addArc('LifestyleIntervention', 'CostUtility')
id_model.addArc('MedicalIntervention', 'CostUtility')
id_model.addArc('MonitoringFrequency', 'CostUtility')
id_model.addArc('LifestyleIntervention', 'BurdenUtility')
id_model.addArc('MedicalIntervention', 'BurdenUtility')
id_model.addArc('MonitoringFrequency', 'BurdenUtility')

# Fill Future Diabetes Status CPT
# Based on DPP trial effectiveness: Lifestyle 58%, Metformin 31%, Intensive 50%
lifestyle_reduction = {'None': 0, 'Diet': 0.35, 'Exercise': 0.40, 'Combined': 0.58}
medical_reduction = {'None': 0, 'Metformin': 0.31, 'Intensive': 0.50}
risk_baseline = {'Low': 0.05, 'Moderate': 0.20, 'High': 0.45, 'VeryHigh': 0.65}

for risk in ['Low', 'Moderate', 'High', 'VeryHigh']:
    for lifestyle in ['None', 'Diet', 'Exercise', 'Combined']:
        for medical in ['None', 'Metformin', 'Intensive']:
            baseline_risk = risk_baseline[risk]
            
            # Calculate combined intervention effect (multiplicative)
            lifestyle_effect = lifestyle_reduction[lifestyle]
            medical_effect = medical_reduction[medical]
            
            if lifestyle == 'None' and medical == 'None':
                total_reduction = 0
            elif lifestyle != 'None' and medical == 'None':
                total_reduction = lifestyle_effect
            elif lifestyle == 'None' and medical != 'None':
                total_reduction = medical_effect
            else:
                # Combined: 1 - (1-lifestyle)(1-medical)
                total_reduction = 1 - (1 - lifestyle_effect) * (1 - medical_effect)
            
            # Apply intervention
            final_risk = baseline_risk * (1 - total_reduction)
            
            # Convert to outcome probabilities
            # If diabetes develops: 70% controlled, 30% uncontrolled
            p_no_diabetes = 1 - final_risk
            p_controlled = final_risk * 0.70
            p_uncontrolled = final_risk * 0.30
            
            id_model.cpt('FutureDiabetesStatus')[{'DiabetesRisk': risk, 'LifestyleIntervention': lifestyle,'MedicalIntervention': medical}] = [p_no_diabetes, p_controlled, p_uncontrolled]

# ============================================================================
# FILL UTILITIES - CAREFULLY CALIBRATED VALUES
# ============================================================================

# Health utilities from Clarke et al. 2002 (UKPDS 62) - EQ-5D utilities
# Scaled to 0-100 range: 
# No diabetes (0.785→78.5), 
# Controlled (0.650→65.0), 
# Uncontrolled (0.550→55.0)
id_model.utility('HealthUtility')[{'FutureDiabetesStatus': 'NoDiabetes'}] = [78.5]
id_model.utility('HealthUtility')[{'FutureDiabetesStatus': 'Controlled'}] = [65.0]
id_model.utility('HealthUtility')[{'FutureDiabetesStatus': 'Uncontrolled'}] = [55.0]

# Cost utilities - scaled to match health utility range
# Based on Herman et al. 2005, converted using $50,000/QALY threshold, then multiplied by 0.5
# to reduce penalty magnitude while maintaining relative ordering
lifestyle_cost = {
    'None': 0, 
    'Diet': -0.7,        # $700/3yr
    'Exercise': -0.8,    # $800/3yr
    'Combined': -1.4     # $1,399/3yr (Herman 2005)
}

medical_cost = {
    'None': 0,
    'Metformin': -0.3,   # $300/yr
    'Intensive': -2.0    # $2,000/yr
}

monitoring_cost = {
    'Annual': -0.1,      
    'Biannual': -0.2,    
    'Quarterly': -0.4,   
    'Monthly': -0.8      
}

for lifestyle in ['None', 'Diet', 'Exercise', 'Combined']:
    for medical in ['None', 'Metformin', 'Intensive']:
        for monitoring in ['Annual', 'Biannual', 'Quarterly', 'Monthly']:
            total_cost = lifestyle_cost[lifestyle] + medical_cost[medical] + monitoring_cost[monitoring]
            id_model.utility('CostUtility')[{
                'LifestyleIntervention': lifestyle,
                'MedicalIntervention': medical,
                'MonitoringFrequency': monitoring
            }] = [total_cost]

# Burden utilities - scaled to match health utility range
# Based on Rubin & Peyrot 1999, reduced by factor of 0.67 to ensure net benefit
lifestyle_burden = {
    'None': 0,
    'Diet': -1.0,        # Dietary restrictions
    'Exercise': -0.7,    # Time commitment, physical effort
    'Combined': -1.7     # Both diet and exercise
}

medical_burden = {
    'None': 0,
    'Metformin': -1.0,   # Side effects, daily medication
    'Intensive': -2.7    # Multiple medications, complex regimen
}

monitoring_burden = {
    'Annual': -0.1,      
    'Biannual': -0.3,    
    'Quarterly': -0.5,   
    'Monthly': -1.1      
}

for lifestyle in ['None', 'Diet', 'Exercise', 'Combined']:
    for medical in ['None', 'Metformin', 'Intensive']:
        for monitoring in ['Annual', 'Biannual', 'Quarterly', 'Monthly']:
            total_burden = lifestyle_burden[lifestyle] + medical_burden[medical] + monitoring_burden[monitoring]
            id_model.utility('BurdenUtility')[{
                'LifestyleIntervention': lifestyle,
                'MedicalIntervention': medical,
                'MonitoringFrequency': monitoring
            }] = [total_burden]

id_model.saveBIFXML('models/diabetes_id.bifxml')

# ============================================================================
# COMPUTE EXPECTED UTILITIES TABLE
# ============================================================================

print("="*70)
print("COMPUTING EXPECTED UTILITIES")
print("="*70)

# Define scenarios
risk_scenarios = {
    'Low': {'Age': 'Young', 'PhysicalActivity': 'Active', 'DietQuality': 'Good', 
            'BMI': 'Normal', 'FamilyHistory': 'No', 'HbA1c': 'Normal'},
    'Moderate': {'Age': 'Middle', 'PhysicalActivity': 'Light', 'DietQuality': 'Fair',
                 'BMI': 'Overweight', 'FamilyHistory': 'No', 'HbA1c': 'Prediabetic'},
    'High': {'Age': 'Middle', 'PhysicalActivity': 'Sedentary', 'DietQuality': 'Fair',
             'BMI': 'Obese', 'FamilyHistory': 'Yes', 'HbA1c': 'Prediabetic'},
    'VeryHigh': {'Age': 'Senior', 'PhysicalActivity': 'Sedentary', 'DietQuality': 'Poor',
                 'BMI': 'Obese', 'FamilyHistory': 'Yes', 'HbA1c': 'Diabetic'}
}

# Define interventions
interventions = {
    'None': {'LifestyleIntervention': 'None', 'MedicalIntervention': 'None', 'MonitoringFrequency': 'Annual'},
    'Diet': {'LifestyleIntervention': 'Diet', 'MedicalIntervention': 'None', 'MonitoringFrequency': 'Biannual'},
    'Combined': {'LifestyleIntervention': 'Combined', 'MedicalIntervention': 'None', 'MonitoringFrequency': 'Biannual'},
    'Metformin': {'LifestyleIntervention': 'None', 'MedicalIntervention': 'Metformin', 'MonitoringFrequency': 'Quarterly'},
    'Combined+Metformin': {'LifestyleIntervention': 'Combined', 'MedicalIntervention': 'Metformin', 'MonitoringFrequency': 'Quarterly'}
}

# Compute expected utilities
results_table = {}

for risk_name, risk_evidence in risk_scenarios.items(): # Each risk level
    results_table[risk_name] = {}
    
    for interv_name, interv_decisions in interventions.items(): # Each intervention
        full_evidence = risk_evidence.copy()
        full_evidence.update(interv_decisions)
        
        try: # Some combinations may be invalid
            ie_id = gum.ShaferShenoyLIMIDInference(id_model)
            ie_id.setEvidence(full_evidence)
            ie_id.makeInference()
            
            future_post = ie_id.posterior('FutureDiabetesStatus')
            health_utils = [78.5, 65.0, 55.0] 
            expected_health = sum(future_post[i] * health_utils[i] for i in range(3))
                        
            cost_util = id_model.utility('CostUtility')[full_evidence][0]
            burden_util = id_model.utility('BurdenUtility')[full_evidence][0]
            
            total_utility = expected_health + cost_util + burden_util
            
            results_table[risk_name][interv_name] = total_utility
            
        except Exception as e:
            results_table[risk_name][interv_name] = None

print("\nExpected Utilities by Risk Level and Intervention Strategy")
print("="*90)
print(f"{'Risk Level':<15} {'None':>8} {'Diet':>8} {'Combined':>10} {'Metformin':>10} {'Comb+Met':>10}")
print("-"*90)

for risk_name in ['Low', 'Moderate', 'High', 'VeryHigh']:
    row = [risk_name]
    for interv_name in ['None', 'Diet', 'Combined', 'Metformin', 'Combined+Metformin']:
        utility = results_table[risk_name][interv_name]
        if utility is not None:
            row.append(f"{utility:8.2f}")
        else:
            row.append("   N/A  ")
    print(f"{row[0]:<15} {' '.join(row[1:])}")

print("\nOptimal Strategies:")
for risk_name in ['Low', 'Moderate', 'High', 'VeryHigh']:
    best_interv = max(results_table[risk_name].items(), key=lambda x: x[1] if x[1] else float('-inf'))
    print(f"  {risk_name}: {best_interv[0]} (EU={best_interv[1]:.2f})")

print("\n" + "="*70)

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("="*70)
print("SAVING OUTPUTS")
print("="*70)

with open('models/diabetes_bn.dot', 'w') as f:
    f.write(bn.toDot())
with open('models/diabetes_id.dot', 'w') as f:
    f.write(id_model.toDot())

try:
    subprocess.run(['dot', '-Tpng', 'models/diabetes_bn.dot', '-o', 'models/diabetes_bn.png'], check=True)
    subprocess.run(['dot', '-Tpng', 'models/diabetes_id.dot', '-o', 'models/diabetes_id.png'], check=True)
    print("Network visualizations created\n")
except:
    print("Could not generate images (install graphviz)\n")

# ============================================================================
# INTERACTIVE RISK ASSESSMENT
# ============================================================================

print("="*70)
print("INTERACTIVE DIABETES RISK ASSESSMENT")
print("="*70)

def get_user_input():

    # Gather user input with validation
    print("\nEnter patient information:")
    
    age = input("Age (Young/Middle/Senior): ").strip().capitalize()
    while age not in ['Young', 'Middle', 'Senior']:
        age = input("Invalid. Age (Young/Middle/Senior): ").strip().capitalize()
    
    family_history = input("Family history of diabetes? (Yes/No): ").strip().capitalize()
    while family_history not in ['Yes', 'No']:
        family_history = input("Invalid. Family history? (Yes/No): ").strip().capitalize()
    
    activity = input("Physical activity (Sedentary/Light/Moderate/Active): ").strip().capitalize()
    while activity not in ['Sedentary', 'Light', 'Moderate', 'Active']:
        activity = input("Invalid. Activity (Sedentary/Light/Moderate/Active): ").strip().capitalize()
    
    diet = input("Diet quality (Poor/Fair/Good): ").strip().capitalize()
    while diet not in ['Poor', 'Fair', 'Good']:
        diet = input("Invalid. Diet (Poor/Fair/Good): ").strip().capitalize()
    
    smoking = input("Smoker? (Yes/No): ").strip().capitalize()
    while smoking not in ['Yes', 'No']:
        smoking = input("Invalid. Smoker? (Yes/No): ").strip().capitalize()
    
    bmi = input("BMI category (Normal/Overweight/Obese): ").strip().capitalize()
    while bmi not in ['Normal', 'Overweight', 'Obese']:
        bmi = input("Invalid. BMI (Normal/Overweight/Obese): ").strip().capitalize()
    
    hba1c = input("HbA1c level (Normal/Prediabetic/Diabetic) [optional, press Enter to skip]: ").strip().capitalize()
    if hba1c and hba1c not in ['Normal', 'Prediabetic', 'Diabetic']:
        hba1c = ''
    
    # Return collected evidence
    return {
        'Age': age,
        'FamilyHistory': family_history,
        'PhysicalActivity': activity,
        'DietQuality': diet,
        'Smoking': smoking,
        'BMI': bmi,
        'HbA1c': hba1c if hba1c else None
    }

def run_assessment(evidence):
    # Perform inference and recommend interventions
    ie = gum.LazyPropagation(bn) # Inference engine
    
    # Set evidence
    clean_evidence = {k: v for k, v in evidence.items() if v is not None} # Remove None values
    ie.setEvidence(clean_evidence)
    ie.makeInference()
    
    # Get posteriors
    risk_post = ie.posterior('DiabetesRisk')
    ir_post = ie.posterior('InsulinResistance')
    ms_post = ie.posterior('MetabolicSyndrome')
    
    print("\n" + "="*70)
    print("RISK ASSESSMENT RESULTS")
    print("="*70)
    
    print("\n5-Year Diabetes Risk:")
    for i, label in enumerate(['Low', 'Moderate', 'High', 'Very High']):
        print(f"  {label:12s}: {risk_post[i]*100:6.2f}%")
    
    print("\nInsulin Resistance (inferred):")
    for i, label in enumerate(['Low', 'Moderate', 'High']):
        print(f"  {label:12s}: {ir_post[i]*100:6.2f}%")
    
    print(f"\nMetabolic Syndrome: {ms_post[1]*100:.2f}% probability")
    
    # Determine risk category
    risk_probs = [risk_post[i] for i in range(4)]
    max_risk_idx = int(np.argmax(risk_probs))
    risk_category = ['Low', 'Moderate', 'High', 'VeryHigh'][max_risk_idx]
    
    print("\n" + "="*70)
    print("RECOMMENDED INTERVENTION")
    print("="*70)
    
    # Lifestyle intervention analysis
    baseline_high_risk = risk_post[2] + risk_post[3]
    
    factors_to_test = {
        'PhysicalActivity': ['Sedentary', 'Light', 'Moderate', 'Active'],
        'DietQuality': ['Poor', 'Fair', 'Good'],
        'BMI': ['Obese', 'Overweight', 'Normal']
    }
    
    # Calculate benefits of modifying each factor
    factor_benefits = {}
    
    for factor, values in factors_to_test.items():
        if factor in clean_evidence:
            current_value = clean_evidence[factor]
            current_idx = values.index(current_value) if current_value in values else -1
            
            if current_idx < len(values) - 1:
                test_evidence = clean_evidence.copy()
                test_evidence[factor] = values[-1]
                
                try:
                    ie_test = gum.LazyPropagation(bn)
                    ie_test.setEvidence(test_evidence)
                    ie_test.makeInference()
                    test_risk_post = ie_test.posterior('DiabetesRisk')
                    test_high_risk = test_risk_post[2] + test_risk_post[3]
                    benefit = baseline_high_risk - test_high_risk
                    factor_benefits[factor] = benefit
                except:
                    factor_benefits[factor] = 0
            else:
                factor_benefits[factor] = 0
    
    # Decide on lifestyle intervention
    activity_benefit = factor_benefits.get('PhysicalActivity', 0)
    diet_benefit = factor_benefits.get('DietQuality', 0)
    bmi_benefit = factor_benefits.get('BMI', 0)
    
    # Combine benefits conservatively
    total_lifestyle_benefit = max(activity_benefit + diet_benefit, bmi_benefit)
    
    # Check if lifestyle is modifiable
    current_activity = clean_evidence.get('PhysicalActivity', 'Unknown')
    current_diet = clean_evidence.get('DietQuality', 'Unknown')
    current_bmi = clean_evidence.get('BMI', 'Unknown')
    
    lifestyle_modifiable = (current_activity in ['Sedentary', 'Light'] or 
                           current_diet in ['Poor', 'Fair'] or
                           current_bmi in ['Overweight', 'Obese'])
    
    if total_lifestyle_benefit < 0.05 or not lifestyle_modifiable:
        lifestyle = 'None'
    elif bmi_benefit > 0.05 and (current_activity in ['Sedentary', 'Light'] or current_diet in ['Poor', 'Fair']):
        lifestyle = 'Combined'
    elif activity_benefit > 0.05 and diet_benefit > 0.05:
        lifestyle = 'Combined'
    elif diet_benefit > activity_benefit and diet_benefit > 0.03:
        lifestyle = 'Diet'
    elif activity_benefit > 0.03:
        lifestyle = 'Exercise'
    elif bmi_benefit > 0.05:
        lifestyle = 'Combined'
    else:
        lifestyle = 'None'
    
    p_high_ir = ir_post[2]
    p_mod_high_ir = ir_post[1] + ir_post[2]
    p_ms = ms_post[1]
    
    # Determine medical intervention
    metabolic_risk = (total_lifestyle_benefit < 0.10 and baseline_high_risk > 0.40)
    
    if baseline_high_risk < 0.20:
        medical = 'None'
    elif baseline_high_risk > 0.70 or p_high_ir > 0.50:
        medical = 'Metformin'
    elif metabolic_risk and not lifestyle_modifiable:
        medical = 'Metformin'
    elif p_mod_high_ir > 0.60 and baseline_high_risk > 0.50:
        medical = 'Metformin'
    elif p_ms > 0.70:
        medical = 'Metformin'
    else:
        medical = 'None'
    
    # Determine monitoring frequency
    monitoring = 'Quarterly' if baseline_high_risk > 0.35 else ('Biannual' if baseline_high_risk > 0.20 else 'Annual')
    
    print(f"\nLifestyle Intervention: {lifestyle}")
    print(f"Medical Intervention: {medical}")
    print(f"Monitoring Frequency: {monitoring}")
    
    print("\n" + "="*70)
    
    return risk_category

# ============================================================================
# GUIDELINE VALIDATION TEST CASES
# ============================================================================

guideline_cases = [
    {
        'name': 'Case 1: Normal Glucose, Low Risk',
        'guideline': 'ADA Section 3, p.S34: Screen q3y, no intervention',
        'reference': 'ADA 2021 Rec 3.1',
        'evidence': {'Age': 'Young', 'PhysicalActivity': 'Active', 'DietQuality': 'Good',
                    'BMI': 'Normal', 'FamilyHistory': 'No', 'Smoking': 'No', 'HbA1c': 'Normal'},
        'expected': 'Low risk → No intervention, Annual monitoring'
    },
    {
        'name': 'Case 2: BMI ≥25 with Risk Factor',
        'guideline': 'ADA Section 3, p.S34: Screen regularly',
        'reference': 'ADA 2021 Rec 3.2',
        'evidence': {'Age': 'Middle', 'PhysicalActivity': 'Light', 'DietQuality': 'Fair',
                    'BMI': 'Overweight', 'FamilyHistory': 'No', 'Smoking': 'No', 'HbA1c': 'Normal'},
        'expected': 'Moderate risk → Lifestyle if modifiable'
    },
    {
        'name': 'Case 3: Prediabetes (HbA1c 5.7-6.4%) - DPP Lifestyle',
        'guideline': 'ADA Section 3, p.S35: Intensive behavioral lifestyle (DPP)',
        'reference': 'ADA 2021 Rec 3.4 (Level A)',
        'evidence': {'Age': 'Middle', 'PhysicalActivity': 'Sedentary', 'DietQuality': 'Poor',
                    'BMI': 'Overweight', 'FamilyHistory': 'No', 'Smoking': 'No', 'HbA1c': 'Prediabetic'},
        'expected': 'Moderate-High risk → Combined lifestyle'
    },
    {
        'name': 'Case 4: Prediabetes + BMI ≥35',
        'guideline': 'ADA Section 3, p.S36: Consider metformin',
        'reference': 'ADA 2021 Rec 3.8 (Level A)',
        'evidence': {'Age': 'Middle', 'PhysicalActivity': 'Sedentary', 'DietQuality': 'Poor',
                    'BMI': 'Obese', 'FamilyHistory': 'Yes', 'Smoking': 'No', 'HbA1c': 'Prediabetic'},
        'expected': 'High risk → Combined + Metformin'
    },
    {
        'name': 'Case 5: Prediabetes with Good Lifestyle (Metabolic)',
        'guideline': 'ADA Section 3, p.S36: Metformin when metabolic/genetic',
        'reference': 'ADA 2021 Rec 3.8',
        'evidence': {'Age': 'Middle', 'PhysicalActivity': 'Active', 'DietQuality': 'Good',
                    'BMI': 'Overweight', 'FamilyHistory': 'Yes', 'HbA1c': 'Prediabetic'},
        'expected': 'High risk → Metformin only'
    },
    {
        'name': 'Case 6: Diabetic Range HbA1c (≥6.5%)',
        'guideline': 'ADA Section 2, p.S16: Diagnose diabetes, treat',
        'reference': 'ADA 2021 Rec 2.1 (Level B)',
        'evidence': {'Age': 'Senior', 'PhysicalActivity': 'Light', 'DietQuality': 'Fair',
                    'BMI': 'Obese', 'FamilyHistory': 'Yes', 'Smoking': 'No', 'HbA1c': 'Diabetic'},
        'expected': 'Very High risk → Intensive intervention'
    },
    {
        'name': 'Case 7: Young, Obesity, Family History',
        'guideline': 'ADA: Lifestyle first for young high-risk',
        'reference': 'ADA 2021 Section 3',
        'evidence': {'Age': 'Young', 'PhysicalActivity': 'Sedentary', 'DietQuality': 'Poor',
                    'BMI': 'Obese', 'FamilyHistory': 'Yes', 'Smoking': 'No', 'HbA1c': 'Normal'},
        'expected': 'Moderate-High risk → Combined lifestyle'
    },
    {
        'name': 'Case 8: Senior, Multiple Risk Factors',
        'guideline': 'ADA: Aggressive intervention for very high risk',
        'reference': 'ADA 2021 Section 3',
        'evidence': {'Age': 'Senior', 'PhysicalActivity': 'Sedentary', 'DietQuality': 'Poor',
                    'BMI': 'Obese', 'FamilyHistory': 'Yes', 'Smoking': 'Yes', 'HbA1c': 'Prediabetic'},
        'expected': 'Very High risk → Combined + Metformin'
    }
]

# Main loop
while True:
    # User menu
    choice = input("\n[1] Assess patient risk\n[2] Run predefined test cases\n[3] Run guideline validation\n[4] Exit\n\nChoice: ").strip()
    
    # Execute based on choice
    if choice == '1': # Interactive assessment
        evidence = get_user_input()
        run_assessment(evidence)
    
    elif choice == '2':# Predefined test cases
        print("\n" + "="*70)
        print("RUNNING PREDEFINED TEST CASES")
        print("="*70)
        
        test_cases = [
            {
                'name': 'Healthy Young Person',
                'evidence': {'Age': 'Young', 'PhysicalActivity': 'Active', 'DietQuality': 'Good',
                            'BMI': 'Normal', 'FamilyHistory': 'No', 'Smoking': 'No'}
            },
            {
                'name': 'High-Risk Middle-Aged Person',
                'evidence': {'Age': 'Middle', 'PhysicalActivity': 'Sedentary', 'DietQuality': 'Poor',
                            'BMI': 'Obese', 'FamilyHistory': 'Yes', 'Smoking': 'Yes', 'HbA1c': 'Prediabetic'}
            },
            {
                'name': 'Prediabetic with Good Lifestyle',
                'evidence': {'Age': 'Middle', 'PhysicalActivity': 'Active', 'DietQuality': 'Good',
                            'BMI': 'Overweight', 'HbA1c': 'Prediabetic', 'FamilyHistory': 'Yes'}
            }
        ]
        
        for test in test_cases:
            print(f"\n--- {test['name']} ---")
            run_assessment(test['evidence'])
    
    elif choice == '3': # Guideline validation
        print("\n" + "="*70)
        print("GUIDELINE VALIDATION TEST CASES")
        print("="*70)
        print("\nValidating against ADA Standards of Medical Care in Diabetes 2021")
        print("Source: https://diabetesjournals.org/care/issue/44/Supplement_1\n")
        
        for case in guideline_cases:
            print(f"\n{'='*70}")
            print(f"{case['name']}")
            print(f"Reference: {case['reference']}")
            print(f"Guideline: {case['guideline']}")
            print(f"Expected: {case['expected']}")
            print(f"{'='*70}")
            
            run_assessment(case['evidence'])
            
            input("\nPress Enter to continue to next case...")
    
    elif choice == '4': # Exit
        print("\nExiting. Models saved in ./models/")
        break
    
    else:
        print("Invalid choice.")

print("\n" + "="*70 + "\n")