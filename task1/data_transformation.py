import pandas

from sklearn.preprocessing import StandardScaler

doc = pandas.read_csv("data_outliered.csv")

def encodeSex():
    global doc
    doc['sex_female'] = (doc['sex'] == 'Female').astype(int)
    doc = doc.drop(columns=["sex"])

def encodeDrink():
    global doc
    doc['DRINK'] = (doc["DRK_YN"] == "Y").astype(int)
    doc = doc.drop(columns=["DRK_YN"])

def standardScaling():
    global doc
    scaler = StandardScaler()

    cols_to_scale = ['age', 'height', 'weight', 'waistline', 'sight_left', 'sight_right', 'SBP', 'DBP',
                     'BLDS', 'tot_chole', 'HDL_chole', 'LDL_chole', 'triglyceride', 'hemoglobin',
                     'urine_protein', 'serum_creatinine', 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'SMK_stat_type_cd']


    doc[cols_to_scale] = scaler.fit_transform(doc[cols_to_scale])

def save():
    doc.to_csv("data_transformed.csv")

encodeSex()
encodeDrink()
standardScaling()
save()


