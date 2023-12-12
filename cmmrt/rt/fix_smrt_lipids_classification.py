#!/usr/bin/env python
# -*- coding: utf-8 -*-

 
"""
@contents :  This module contains functions to generate fingerprints and descriptors using alvaDesc program from the SMRT database
@project :  cmmrt (CEU Mass Mediator Retention Time)
@program :  CEU Mass Mediator
@file :  build_data_smrt.py
@author :  Alberto Gil De la Fuente (alberto.gilf@gmail.com)
           Constantino García Martínez(constantino.garciama@ceu.es)
           

@version :  0.0.2, 29 September 2021
@information : A valid license of AlvaDesc is necessary to generate the descriptors and fingerprints of chemical structures. 

@copyright :  GNU General Public License v3.0
              Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, 
              which include larger works using a licensed work, under the same license. 
              Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
@end
"""
import csv
import os
import build_data
import pandas as pd
import time

def main():
    inputPath = '/home/alberto/data/vector_fingerprints/'
    outputPath = '/home/alberto/data/vector_fingerprints/'
    #Constants
    NUMBER_FPVALUES=2214

    inputFileName = inputPath + "SMRT_vectorfingerprints_categorize_lipids.csv"
    # IT WILL TAKE SDFs FROM PC IDS to create a CSV file containing the vector with fingerprints (ECFP, MACCSFP and PFP) of each SMRT compound
    
    outputFileFingerprintsName = outputPath + "SMRT_vectorfingerprints_categorize_lipids_fixed.csv"
    if os.path.isfile(outputFileFingerprintsName):
        os.remove(outputFileFingerprintsName)


    # GET HEADERS FROM CLASSYFIRE
    classyfireFieldName ="lipid_from_classyfire"

    # CHECK IF THE COMPOUND IS IN LIPIDMAPS
    lipidMapsFieldName ="presence_in_lipidmaps"

    

    # WRITER FOR FINGERPRINTS
    outputFileFingerprints = open(outputFileFingerprintsName, 'w', newline='')
    FPFieldNames =['pid','rt']
    for i in range(0,NUMBER_FPVALUES):
        header_name = "V" + str(i+1)
        FPFieldNames.append(header_name)
    FPFieldNames.append(classyfireFieldName)
    FPFieldNames.append(lipidMapsFieldName)

    # Write headers
    writerFP = csv.DictWriter(outputFileFingerprints, fieldnames = FPFieldNames)
    writerFP.writeheader()

    with open(inputFileName) as csvfile:
        reader = csv.DictReader(csvfile,delimiter=',')
        for row in reader:
            classified_classyfire=row["lipid_from_classyfire"]
            classified_lipidmaps=row["presence_in_lipidmaps"]
            pc_id=0
            if not classified_classyfire in('0','1'):
                pc_id = row["pid"]
                print(pc_id)
                inchi, inchi_key = build_data.get_inchi_and_inchi_key_from_pubchem(pc_id)
                try:
                    is_lipid_from_classyfire = build_data.is_a_lipid_from_classyfire(inchi, inchi_key)
                    if is_lipid_from_classyfire:
                        is_lipid_from_classyfire = 1
                    else: 
                        is_lipid_from_classyfire = 0
                    row[classyfireFieldName] = is_lipid_from_classyfire
                except Exception as e:
                    print(e)
                    print("Check CLASSYFIRE classification for PC ID " + pc_id + " Inchi key: " + inchi_key + " inchi: " + inchi)
                    is_lipid_from_classyfire = "Not classified"
            if not classified_lipidmaps in('0','1'):
                if pc_id ==0:
                    pc_id = row["pid"]
                    print(pc_id)
                    inchi, inchi_key = build_data.get_inchi_and_inchi_key_from_pubchem(pc_id)
                try:
                    is_lipid_lipidMaps = build_data.is_in_lipidMaps(inchi_key)
                    if is_lipid_lipidMaps:
                        is_lipid_lipidMaps = 1
                    else:
                        is_lipid_lipidMaps = 0
                    row[lipidMapsFieldName] = is_lipid_lipidMaps
                except Exception:
                    print("Check LipidMaps classification for PC ID: " + pc_id + " Inchi key: " + inchi_key + " inchi: " + inchi)
                    is_lipid_lipidMaps = "Not classified"

            writerFP.writerow(row)


if __name__ == "__main__":
    main()