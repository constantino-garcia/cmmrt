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
from alvadesccliwrapper.alvadesc import AlvaDesc
import csv
import os
import build_data

#ALVADESC_LOCATION = 'C:/"Program Files"/Alvascience/alvaDesc/alvaDescCLI.exe'
ALVADESC_LOCATION = '/usr/bin/alvaDescCLI'

def main():
    
    #Constants
    NUMBER_FPVALUES=2214
    # VARIABLES OF AlvaDesc Software

    aDesc = AlvaDesc(ALVADESC_LOCATION)
    # INPUT PATH CONTAINS PC IDS, RTs and INCHI of SMRT Database
    inputPath = 'C:/Users/alberto.gildelafuent/OneDrive - Fundación Universitaria San Pablo CEU/research/SMRT_in_CMM/'
    inputFileName = inputPath + "SMRT_dataset_test.csv"
    # IT WILL TAKE SDFs FROM PC IDS to create a CSV file containing the vector with fingerprints (ECFP, MACCSFP and PFP) of each SMRT compound
    outputPath = 'C:/Users/alberto.gildelafuent/Desktop/alberto/resources/SMRT/'
    sdfPath = outputPath + "SDF/"
    outputFileDescriptorsName = outputPath + "vector_fingerprints/SMRT_descriptors.csv"
    outputFileFingerprintsName = outputPath + "vector_fingerprints/SMRT_vectorfingerprints.csv"
    outputFileMergedName = outputPath + "vector_fingerprints/SMRT_descriptorsAndFingerprints.csv"
    if os.path.isfile(outputFileDescriptorsName):
        os.remove(outputFileDescriptorsName)
    if os.path.isfile(outputFileFingerprintsName):
        os.remove(outputFileFingerprintsName)
    if os.path.isfile(outputFileMergedName):
        os.remove(outputFileMergedName)

    # RUN A MOCK SDF TO OBTAIN DESCRIPTORS HEADERS
    aDesc.set_input_file(sdfPath + '1.sdf', 'MDL')
    aDesc.calculate_descriptors('ALL')
    listDescriptors = aDesc.get_output_descriptors()
    outputFileDescriptors = open(outputFileDescriptorsName, 'w', newline='')
    descriptorFieldNames =['pid','rt']
    descriptorFieldNames.extend(listDescriptors)
    writerDescriptors = csv.DictWriter(outputFileDescriptors, fieldnames = descriptorFieldNames)
    writerDescriptors.writeheader()

    # WRITER FOR FINGERPRINTS
    outputFileFingerprints = open(outputFileFingerprintsName, 'w', newline='')
    FPFieldNames =['pid','rt']
    for i in range(0,NUMBER_FPVALUES):
        header_name = "V" + str(i+1)
        FPFieldNames.append(header_name)
    writerFP = csv.DictWriter(outputFileFingerprints, fieldnames = FPFieldNames)
    writerFP.writeheader()

    # WRITER FOR MERGED
    mergedFieldNames = descriptorFieldNames[:]
    mergedFieldNames.extend(FPFieldNames[2:])
    outputFileMerged = open(outputFileMergedName, 'w', newline='')
    writerMerged = csv.DictWriter(outputFileMerged, fieldnames = mergedFieldNames)
    writerMerged.writeheader()
    
    with open(inputFileName) as csvfile:
        reader = csv.DictReader(csvfile,delimiter=';')
        for row in reader:
            pc_id = row["pubchem"]
            rt = row["rt"]
            sdffileName = sdfPath + str(pc_id) + ".sdf"
            partialDictMerged = {'pid' : pc_id, 'rt' : rt}
            descriptors = build_data.get_descriptors(aDesc,sdffileName)
            partialDictDescriptors = {'pid' : pc_id, 'rt' : rt}
            for i in range(0,len(listDescriptors)):
                descriptor_header = listDescriptors[i]
                partialDictDescriptors[descriptor_header] = descriptors[i]
            writerDescriptors.writerow(partialDictDescriptors)


            vector_fingerprints = build_data.generate_vector_fingerprints(aDesc,sdffileName)
            partialDictFP = {'pid' : pc_id, 'rt' : rt}
            for i in range(0,NUMBER_FPVALUES):
                header_name = "V" + str(i+1)
                partialDictFP[header_name] = vector_fingerprints[i]
                partialDictDescriptors[header_name] = vector_fingerprints[i]
            writerFP.writerow(partialDictFP)
            writerMerged.writerow(partialDictDescriptors)


if __name__ == "__main__":
    main()
