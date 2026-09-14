import pandas
import numpy

doc = pandas.read_csv("data_cleaned.csv")


#removes outliers after iqr.
def outlierRemoval(column):

    doc[column] = numpy.log(doc[column])

    Q1 = doc[column].quantile(0.25)
    Q3 = doc[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print("log transform: " + lower +", " + upper)
    print("mg/dl: " + numpy.exp(lower), numpy.exp(upper))
    doc_clean = doc[(doc[column] >= lower) & (doc[column] <= upper)]

    doc_clean[column] = numpy.exp(doc_clean[column])

    print("lines before/after: " + len(doc), len(doc_clean))

def outlierCap():
    return
def save():
    doc.to_csv("data_cleaned.csv", index=False)
    
outlierRemoval("tot_chole")
outlierRemoval()