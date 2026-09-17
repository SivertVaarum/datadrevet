import pandas

document = pandas.read_csv("../smoking_driking_dataset_Ver01.csv")


#Identify missing categoricals
def missingCategorical(column):
    col = document[column]
    print("Unique values:", col.nunique())
    print("Missing values:", col.isna().sum())
    print("Value counts:")
    print(col.value_counts())


def numericalExplore(column):
    
    column = document[column]
    print(column.describe())
    print("not-null: ")
    print(column.notna())

numericalExplore("age")
numericalExplore("height")
numericalExplore("weight")
numericalExplore("waistline")
numericalExplore("sight_left")
numericalExplore("sight_right")
numericalExplore("SBP")
numericalExplore("DBP")
numericalExplore("BLDS")
numericalExplore("tot_chole")
numericalExplore("HDL_chole")
numericalExplore("LDL_chole")
numericalExplore("hemoglobin")
numericalExplore("serum_creatinine")
numericalExplore("SGOT_AST")
numericalExplore("SGOT_ALT")
numericalExplore("height")
numericalExplore("gamma_GTP")

    
missingCategorical("sex")
missingCategorical("hear_left")
missingCategorical("hear_right")
missingCategorical("DRK_YN")
missingCategorical("urine_protein")
missingCategorical("SMK_stat_type_cd")