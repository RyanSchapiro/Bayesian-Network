"""
Type 2 Diabetes Risk Assessment and Management Decision Network
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

print(f"✓ Network: {bn.size()} nodes, {bn.sizeArcs()} arcs")

print("✓ Filling probability tables...")

# Prior probabilities
bn.cpt('Age')[:] = [0.35, 0.40, 0.25]
bn.cpt('FamilyHistory')[:] = [0.70, 0.30]
bn.cpt('PhysicalActivity')[:] = [0.30, 0.35, 0.25, 0.10]
bn.cpt('DietQuality')[:] = [0.25, 0.50, 0.25]
bn.cpt('Smoking')[:] = [0.82, 0.18]

def normalize(probs):
    total = sum(probs)
    return [p/total for p in probs]

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
gum.saveBN(bn, 'models/diabetes_bn.bifxml')
print("✓ Bayesian Network complete\n")

# ============================================================================
# BUILD INFLUENCE DIAGRAM
# ============================================================================

print("="*70)
print("BUILDING INFLUENCE DIAGRAM")
print("="*70)

id_model = gum.InfluenceDiagram()

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
lifestyle_reduction = {'None': 0, 'Diet': 0.35, 'Exercise': 0.40, 'Combined': 0.58}
medical_reduction = {'None': 0, 'Metformin': 0.31, 'Intensive': 0.50}
risk_baseline = {'Low': 0.05, 'Moderate': 0.15, 'High': 0.30, 'VeryHigh': 0.45}

for risk in ['Low', 'Moderate', 'High', 'VeryHigh']:
    for lifestyle in ['None', 'Diet', 'Exercise', 'Combined']:
        for medical in ['None', 'Metformin', 'Intensive']:
            baseline_risk = risk_baseline[risk]
            total_reduction = (lifestyle_reduction[lifestyle] + medical_reduction[medical]) * 0.75
            p_no_diabetes = 1 - baseline_risk * (1 - total_reduction)
            p_no_diabetes = max(0.08, min(0.97, p_no_diabetes))
            p_uncontrolled = (1 - p_no_diabetes) * 0.35
            
            id_model.cpt('FutureDiabetesStatus')[{'DiabetesRisk': risk, 'LifestyleIntervention': lifestyle,
                                                   'MedicalIntervention': medical}] = \
                normalize([p_no_diabetes, 1-p_no_diabetes-p_uncontrolled, p_uncontrolled])

# Fill utilities
id_model.utility('HealthUtility')[{'FutureDiabetesStatus': 'NoDiabetes'}] = [4.50]
id_model.utility('HealthUtility')[{'FutureDiabetesStatus': 'Controlled'}] = [3.80]
id_model.utility('HealthUtility')[{'FutureDiabetesStatus': 'Uncontrolled'}] = [2.90]

lifestyle_cost = {'None': 0, 'Diet': -0.20, 'Exercise': -0.30, 'Combined': -0.45}
medical_cost = {'None': 0, 'Metformin': -0.25, 'Intensive': -1.00}
monitoring_cost = {'Annual': -0.015, 'Biannual': -0.030, 'Quarterly': -0.060, 'Monthly': -0.120}

for lifestyle in ['None', 'Diet', 'Exercise', 'Combined']:
    for medical in ['None', 'Metformin', 'Intensive']:
        for monitoring in ['Annual', 'Biannual', 'Quarterly', 'Monthly']:
            id_model.utility('CostUtility')[{'LifestyleIntervention': lifestyle,
                                             'MedicalIntervention': medical,
                                             'MonitoringFrequency': monitoring}] = \
                [lifestyle_cost[lifestyle] + medical_cost[medical] + monitoring_cost[monitoring]]

lifestyle_burden = {'None': 0, 'Diet': -0.12, 'Exercise': -0.10, 'Combined': -0.25}
medical_burden = {'None': 0, 'Metformin': -0.15, 'Intensive': -0.35}
monitoring_burden = {'Annual': -0.02, 'Biannual': -0.04, 'Quarterly': -0.08, 'Monthly': -0.15}

for lifestyle in ['None', 'Diet', 'Exercise', 'Combined']:
    for medical in ['None', 'Metformin', 'Intensive']:
        for monitoring in ['Annual', 'Biannual', 'Quarterly', 'Monthly']:
            id_model.utility('BurdenUtility')[{'LifestyleIntervention': lifestyle,
                                               'MedicalIntervention': medical,
                                               'MonitoringFrequency': monitoring}] = \
                [lifestyle_burden[lifestyle] + medical_burden[medical] + monitoring_burden[monitoring]]

id_model.saveBIFXML('models/diabetes_id.bifxml')
print("✓ Influence Diagram complete\n")

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
    os.system('open models/diabetes_bn.png 2>/dev/null')
    os.system('open models/diabetes_id.png 2>/dev/null')
    print("✓ Network visualizations created and opened\n")
except:
    print("⚠ Could not generate images (install graphviz)\n")

# ============================================================================
# INTERACTIVE RISK ASSESSMENT
# ============================================================================

print("="*70)
print("INTERACTIVE DIABETES RISK ASSESSMENT")
print("="*70)

def get_user_input():
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
    ie = gum.LazyPropagation(bn)
    
    # Set evidence
    clean_evidence = {k: v for k, v in evidence.items() if v is not None}
    ie.setEvidence(clean_evidence)
    ie.makeInference()
    
    # Get results
    risk_post = ie.posterior('DiabetesRisk')
    ir_post = ie.posterior('InsulinResistance')
    ms_post = ie.posterior('MetabolicSyndrome')
    
    print("\n" + "="*70)
    print("RISK ASSESSMENT RESULTS")
    print("="*70)
    
    print("\n5-Year Diabetes Risk:")
    for i, label in enumerate(['Low', 'Moderate', 'High', 'Very High']):
        print(f"  {label:12s}: {risk_post[i]*100:5.1f}%")
    
    print("\nInsulin Resistance (inferred):")
    for i, label in enumerate(['Low', 'Moderate', 'High']):
        print(f"  {label:12s}: {risk_post[i]*100:5.1f}%")
    
    print(f"\nMetabolic Syndrome: {ms_post[1]*100:.1f}% probability")
    
    # Determine risk category
    max_risk_idx = np.argmax(risk_post)
    risk_category = ['Low', 'Moderate', 'High', 'VeryHigh'][max_risk_idx]
    
    print("\n" + "="*70)
    print("RECOMMENDED INTERVENTION")
    print("="*70)
    
    # Get optimal intervention
    recommendations = {
        'Low': ('No intervention', 'Annual monitoring', 'Continue healthy lifestyle'),
        'Moderate': ('Combined lifestyle intervention', 'Biannual monitoring', 'Focus on diet and exercise'),
        'High': ('Metformin or intensive lifestyle', 'Quarterly monitoring', 'Consider pharmacological intervention'),
        'VeryHigh': ('Combined lifestyle + Metformin', 'Quarterly monitoring', 'Intensive intervention recommended')
    }
    
    rec = recommendations[risk_category]
    print(f"\nPrimary Recommendation: {rec[0]}")
    print(f"Monitoring: {rec[1]}")
    print(f"Notes: {rec[2]}")
    
    print("\n" + "="*70)
    
    return risk_category

# Main loop
while True:
    choice = input("\n[1] Assess patient risk\n[2] Run predefined test cases\n[3] Exit\n\nChoice: ").strip()
    
    if choice == '1':
        evidence = get_user_input()
        run_assessment(evidence)
    
    elif choice == '2':
        print("\n" + "="*70)
        print("RUNNING TEST CASES")
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
    
    elif choice == '3':
        print("\nExiting. Models saved in ./models/")
        break
    
    else:
        print("Invalid choice.")

print("\n" + "="*70 + "\n")