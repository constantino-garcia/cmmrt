#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
@contents :  This module contains functions to generate fingerprints and descriptors using alvaDesc program from the SMRT database
@project :  cmmrt (CEU Mass Mediator Retention Time)
@program :  CEU Mass Mediator
@file :  build_data_smrt.py
@author :  Alberto Gil De la Fuente (alberto.gilf@gmail.com)
           Constantino García Martínez(constantino.garciama@ceu.es)
           

@version :  0.0.1, 27 September 2021
@information : A valid license of AlvaDesc is necessary to generate the descriptors and fingerprints of chemical structures. 

@copyright :  GNU General Public License v3.0
              Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, 
              which include larger works using a licensed work, under the same license. 
              Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
@end
"""
from alvadesccliwrapper.alvadesc import AlvaDesc
import csv
import os
import build_data

def main():
    # VARIABLES OF AlvaDesc Software
    aDescPath = 'C:/"Program Files"/Alvascience/alvaDesc/alvaDescCLI.exe'
    aDesc = AlvaDesc(aDescPath)
    # INPUT PATH CONTAINS PC IDS, RTs and INCHI of SMRT Database
    inputPath = 'C:/Users/alberto.gildelafuent/OneDrive - Fundación Universitaria San Pablo CEU/research/SMRT_in_CMM/'
    inputFileName = inputPath + "SMRT_dataset_test.csv"
    # IT WILL TAKE SDFs FROM PC IDS to create a CSV file containing the vector with fingerprints (ECFP, MACCSFP and PFP) of each SMRT compound
    outputPath = 'C:/Users/alberto.gildelafuent/Desktop/alberto/resources/SMRT/'
    outputFileName = outputPath + "vector_fingerprints/SMRT_vectorfingerprints.csv"
    sdfPath = outputPath + "SDF/"

    outputFile = open(outputFileName, 'w', newline='')
    outputFieldNames =['pid','rt']
    for i in range(1,2215):
        header_name = "V" + str(i)
        outputFieldNames.append(header_name)
    
    writer = csv.DictWriter(outputFile, fieldnames = outputFieldNames)
    writer.writeheader()
    with open(inputFileName) as csvfile:
        reader = csv.DictReader(csvfile,delimiter=';')
        for row in reader:
            pc_id = row["pubchem"]
            rt = row["rt"]
            sdffileName = sdfPath + str(pc_id) + ".sdf"
            vector_fingerprints = build_data.generate_vector_fingerprints(aDesc,sdffileName)
            partialDict = {'pid' : pc_id, 'rt' : rt}
            for i in range(1,2215):
                header_name = "V" + str(i)
                partialDict[header_name] = vector_fingerprints[i-1]
            writer.writerow(partialDict)

if __name__ == "__main__":
    main()