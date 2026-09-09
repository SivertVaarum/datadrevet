import pandas

document = pandas.read_csv("../smoking_driking_dataset_Ver01.csv")


#Identify missing categoricals
def missingCategorical(column):

    column = document[column]
    print("nunique: ")
    print(column.nunique())
    print("info: ")
    print(column.info())
    
missingCategorical("sex")
missingCategorical("DRK_YN")