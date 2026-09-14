import pandas as pd

doc = pd.read_csv("../smoking_driking_dataset_Ver01.csv")
print(len(doc))

#2. Choose appropriate methods to handle missing values (e.g., mean/median
#   imputation for numerical data, mode imputation for categorical data, or deletion of
#   rows/columns)

#Drops missing values
def deleteColumn(column):
    
    print(len(doc))
    doc[column] = doc[column].dropna()
    print(len(doc))
    
def replaceWithMedian(column):
    
    median = doc[column].median()
    doc[column] = doc[column].fillna(median)
    return

#Missing hearing values as flagged as "dont know" using -1
def hearingReplace(column):
    doc[column] = doc[column].fillna(-1)
    return
    
def save():
    print(len(doc))
    doc.to_csv("data_cleaned.csv", index=False)
    return



deleteColumn("DRK_YN") #Missing values for drinking deletion.
deleteColumn("SMK_stat_type_cd") #Missing smoke? -> Delete row.
deleteColumn("urine_protein")

hearingReplace("hear_left") #
hearingReplace("hear_right")

replaceWithMedian("age") #Replace missing with median
replaceWithMedian("height")
replaceWithMedian("weight")
replaceWithMedian("waistline")
#SBP	DBP	BLDS	tot_chole	HDL_chole	LDL_chole	triglyceride	hemoglobin
deleteColumn("SBP")
deleteColumn("DBP")
deleteColumn("BLDS")
deleteColumn("tot_chole")
deleteColumn("HDL_chole")
deleteColumn("LDL_chole")
deleteColumn("triglyceride")
deleteColumn("hemoglobin")
#serum_creatinine	SGOT_AST	SGOT_ALT	gamma_GTP
deleteColumn("serum_creatinine")
deleteColumn("SGOT_AST")
deleteColumn("SGOT_ALT")
deleteColumn("gamma_GTP")
