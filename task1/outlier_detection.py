import pandas
import numpy

doc = pandas.read_csv("smoking_driking_dataset_Ver01.csv")
clean_doc = None

def outliersCholesterol():

    doc["tot_chole"] = numpy.log(doc["tot_chole"])

    Q1 = doc['tot_chole'].quantile(0.25)
    Q3 = doc['tot_chole'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print("log transform: " + lower +", " + upper)
    print("mg/dl: " + numpy.exp(lower), numpy.exp(upper))
    doc_clean = doc[(doc['tot_chole'] >= lower) & (doc['tot_chole'] <= upper)]

    doc_clean["tot_chole"] = numpy.exp(doc_clean['tot_chole'])

    print("lines before/after: " + len(doc), len(doc_clean))

def handleMissing():
    #Regn ut median og insert? 
    return None

def encodingCatergorical():
    #Male = 0, Female = 1
    return None


outliersCholesterol()