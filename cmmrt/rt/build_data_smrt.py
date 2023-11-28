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
    inputPath = '/home/alberto/OneDrive/research/SMRT_in_CMM/'
    outputPath = '/home/alberto/repos/cmm_rt_shared/'
    #Constants
    NUMBER_FPVALUES=2214
    # VARIABLES OF AlvaDesc Software

    aDesc = AlvaDesc(ALVADESC_LOCATION)
    # INPUT PATH CONTAINS PC IDS, RTs and INCHI of SMRT Database
    
    inputFileName = inputPath + "SMRT_dataset.csv"
    # IT WILL TAKE SDFs FROM PC IDS to create a CSV file containing the vector with fingerprints (ECFP, MACCSFP and PFP) of each SMRT compound
    
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


    # GET HEADERS FROM CLASSYFIRE
    classyfireFieldName ="lipid_from_classyfire"

    # CHECK IF THE COMPOUND IS IN LIPIDMAPS
    lipidMapsFieldName ="presence_in_lipidmaps"

    # RUN A MOCK SDF TO OBTAIN DESCRIPTORS HEADERS
    aDesc.set_input_file(sdfPath + '1.sdf', 'MDL')
    aDesc.calculate_descriptors('ALL')
    listDescriptors = aDesc.get_output_descriptors()
    outputFileDescriptors = open(outputFileDescriptorsName, 'w', newline='')
    descriptorFieldNames =['pid','rt']
    descriptorFieldNames.append(classyfireFieldName)
    descriptorFieldNames.append(lipidMapsFieldName)
    descriptorFieldNames.extend(listDescriptors)

    # Write headers
    writerDescriptors = csv.DictWriter(outputFileDescriptors, fieldnames = descriptorFieldNames)
    writerDescriptors.writeheader()

    # WRITER FOR FINGERPRINTS
    outputFileFingerprints = open(outputFileFingerprintsName, 'w', newline='')
    FPFieldNames =['pid','rt']
    FPFieldNames.append(classyfireFieldName)
    FPFieldNames.append(lipidMapsFieldName)
    for i in range(0,NUMBER_FPVALUES):
        header_name = "V" + str(i+1)
        FPFieldNames.append(header_name)

    # Write headers
    writerFP = csv.DictWriter(outputFileFingerprints, fieldnames = FPFieldNames)
    writerFP.writeheader()

    # WRITER FOR MERGED
    mergedFieldNames = descriptorFieldNames[:]
    mergedFieldNames.extend(FPFieldNames[4:])
    outputFileMerged = open(outputFileMergedName, 'w', newline='')

    # Write headers
    writerMerged = csv.DictWriter(outputFileMerged, fieldnames = mergedFieldNames)
    writerMerged.writeheader()
    
    with open(inputFileName) as csvfile:
        reader = csv.DictReader(csvfile,delimiter=';')
        for row in reader:
            pc_id = row["pubchem"]
            rt = row["rt"]
            sdffileName = sdfPath + str(pc_id) + ".sdf"
            descriptors = build_data.get_descriptors(aDesc,sdffileName)
            partialDictDescriptors = {'pid' : pc_id, 'rt' : rt}
            try:
                print(pc_id)
                inchi_key = build_data.get_inchi_key_from_pubchem(pc_id)
                print(inchi_key)
                is_lipid_lipidMaps = build_data.is_in_lipidMaps(inchi_key)
                print(is_lipid_lipidMaps)
                if is_lipid_lipidMaps:
                    is_lipid_lipidMaps = 1
                else:
                    is_lipid_lipidMaps = 0

                is_lipid_from_classyfire = build_data.is_a_lipid_from_classyfire(inchi_key)
                print(is_lipid_from_classyfire)
                if is_lipid_from_classyfire:
                    is_lipid_from_classyfire = 1
                else: 
                    is_lipid_from_classyfire = 0

            except Exception as e:
                print("CHECK THE PC ID " + pc_id + " and its inchi: " + inchi_key)
            
            partialDictDescriptors[classyfireFieldName] = is_lipid_from_classyfire
            partialDictDescriptors[lipidMapsFieldName] = is_lipid_lipidMaps
            for i in range(0,len(listDescriptors)):
                descriptor_header = listDescriptors[i]
                partialDictDescriptors[descriptor_header] = descriptors[i]
            writerDescriptors.writerow(partialDictDescriptors)


            vector_fingerprints = build_data.generate_vector_fingerprints(aDesc,sdffileName)
            partialDictFP = {'pid' : pc_id, 'rt' : rt}
            partialDictFP[classyfireFieldName] = 0
            partialDictFP[lipidMapsFieldName] = 0
            for i in range(0,NUMBER_FPVALUES):
                header_name = "V" + str(i+1)
                partialDictFP[header_name] = vector_fingerprints[i]
                partialDictDescriptors[header_name] = vector_fingerprints[i]
            writerFP.writerow(partialDictFP)
            writerMerged.writerow(partialDictDescriptors)


if __name__ == "__main__":
    main()
