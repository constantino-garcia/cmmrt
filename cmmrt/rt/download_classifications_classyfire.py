#!/usr/bin/env python
# -*- coding: utf-8 -*-

 
"""
@contents :  This module contains functions to generate fingerprints and descriptors using alvaDesc program from the SMRT database
@project :  cmmrt (CEU Mass Mediator Retention Time)
@program :  CEU Mass Mediator
@file :  download_classification_classyfire.py
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
    outputPath = '/home/alberto/data/vector_fingerprints/classifications/'

    inputFileName = inputPath + "SMRT_vectorfingerprints_categorize_lipids.csv"
    # IT WILL TAKE SDFs FROM PC IDS to create a CSV file containing the vector with fingerprints (ECFP, MACCSFP and PFP) of each SMRT compound

    with open(inputFileName) as csvfile:
        reader = csv.DictReader(csvfile,delimiter=',')
        for row in reader:
            pc_id = row["pid"]
            inchi, inchi_key = build_data.get_inchi_and_inchi_key_from_pubchem(pc_id)
            
            try:
                build_data.download_classification_classyfire(inchi,inchi_key,pc_id, outputPath)
                
            except Exception:
                print("Classification not downloaded")

if __name__ == "__main__":
    main()
