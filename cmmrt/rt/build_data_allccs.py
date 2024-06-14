#!/usr/bin/env python
# -*- coding: utf-8 -*-

 
"""
@contents :  This module contains functions to generate fingerprints and descriptors using alvaDesc program from the ALL_CCS database
@project :  cmmrt (CEU Mass Mediator Retention Time)
@program :  CEU Mass Mediator
@file :  build_data_allccs.py
@author :  Alberto Gil De la Fuente (alberto.gilf@gmail.com)
           Constantino García Martínez(constantino.garciama@ceu.es)
           

@version :  0.0.3, 26 November 2023
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
    #inputPath = '/home/ceu/OneDrive/research/CCS_in_CMM/metlin_ccs/CCS-Publication-V3/'
    inputPath = '/home/ceu/OneDrive/research/CCS_in_CMM/metlin_ccs/'
    outputPath = '/home/ceu/research/repos/cmm_rt_shared/dataset_20240408/'
    sdf_path = f"{outputPath}SDF/"
    #Constants
    NUMBER_FPVALUES=2214
    # VARIABLES OF AlvaDesc Software

    aDesc = AlvaDesc(ALVADESC_LOCATION)
    # INPUT PATH CONTAINS SMILES, INCHIS, CCS_Information of compounds of ALL_CCS IMS
    
    inputFileName = inputPath + "AllCCS2_experimental_with_inchis.csv"
    # IT WILL TAKE SMILES to create a CSV file containing the vector with fingerprints (ECFP, MACCSFP and PFP) of each ALL_CCS compound
    
    outputFileDescriptorsName = outputPath + "vector_fingerprints/ALL_CCS_descriptors.csv"
    outputFileDescriptorsAndFingerprintsName = outputPath + "vector_fingerprints/ALL_CCS_descriptorsAndFingerprints.csv"
    outputFileFingerprintsVectorizedName = outputPath + "vector_fingerprints/ALL_CCS_vectorfingerprintsVectorized.csv"
    outputFileDescriptorsAndFingerPrintsVectorizedName = outputPath + "vector_fingerprints/ALL_CCS_descriptorsAndFingerprintsVectorized.csv"
    if os.path.isfile(outputFileDescriptorsName):
        os.remove(outputFileDescriptorsName)
    if os.path.isfile(outputFileDescriptorsAndFingerprintsName):
        os.remove(outputFileDescriptorsAndFingerprintsName)
    if os.path.isfile(outputFileFingerprintsVectorizedName):
        os.remove(outputFileFingerprintsVectorizedName)
    if os.path.isfile(outputFileDescriptorsAndFingerPrintsVectorizedName):
        os.remove(outputFileDescriptorsAndFingerPrintsVectorizedName)

    with open(inputFileName) as csvfile:
        #reader = csv.DictReader(csvfile,delimiter='\t')
        reader = csv.DictReader(csvfile,delimiter=',',quotechar='"')
        # RUN A MOCK SDF TO OBTAIN DESCRIPTORS HEADERS
        inchi="O=C(NCc1ccc(cc1)F)NCCCN1CCc2c1cccc2"
        aDesc.set_input_SMILES(inchi)
        aDesc.calculate_descriptors('ALL')
        listDescriptors = aDesc.get_output_descriptors()

        # Create here the headers from the input file and then add the descriptors
        descriptorFieldNames =reader.fieldnames.copy()
        descriptorFieldNames.extend(listDescriptors)
        descriptorsAndFingerPrintsFieldNames = descriptorFieldNames[:]
        descriptorsAndFingerPrintsVectorizedFieldNames = descriptorFieldNames[:]
        
        # Write headers in the output file
        outputFileDescriptors = open(outputFileDescriptorsName, 'w', newline='')
        writerDescriptors = csv.DictWriter(outputFileDescriptors, fieldnames = descriptorFieldNames)
        writerDescriptors.writeheader()

        # WRITER FOR FINGERPRINTS AND DESCRIPTORS
        descriptorsAndFingerPrintsFieldNames.append('ECFP')
        descriptorsAndFingerPrintsFieldNames.append('MACCSFP')
        descriptorsAndFingerPrintsFieldNames.append('PFP')
        descriptorsAndFingerPrintsFieldNames.append('MorganFP')

        outputFileDescriptorsAndFingerprints = open(outputFileDescriptorsAndFingerprintsName, 'w', newline='')
        writerDescriptorsAndFingerprints = csv.DictWriter(outputFileDescriptorsAndFingerprints, fieldnames = descriptorsAndFingerPrintsFieldNames)
        writerDescriptorsAndFingerprints.writeheader()

        # Create here the headers from the input file and then add the Fingerprints
        FPVectorizedFieldNames = reader.fieldnames.copy()
        for i in range(0,NUMBER_FPVALUES):
            header_name = "V" + str(i+1)
            FPVectorizedFieldNames.append(header_name)
            descriptorsAndFingerPrintsVectorizedFieldNames.append(header_name)

        # WRITER FOR FINGERPRINTS VECTORIZED
        outputFileFingerprintsVectorized = open(outputFileFingerprintsVectorizedName, 'w', newline='')
        writerFingerprintsVectorized = csv.DictWriter(outputFileFingerprintsVectorized, fieldnames = FPVectorizedFieldNames)
        writerFingerprintsVectorized.writeheader()

        # WRITER FOR MERGED
        outputFileDescriptorsAndFingerPrintsVectorized = open(outputFileDescriptorsAndFingerPrintsVectorizedName, 'w', newline='')
        writerDescriptorsAndFingerPrintsVectorized = csv.DictWriter(outputFileDescriptorsAndFingerPrintsVectorized, fieldnames = descriptorsAndFingerPrintsVectorizedFieldNames)
        writerDescriptorsAndFingerPrintsVectorized.writeheader()


        for row in reader:
            inchi = row["InChI"]
            try:
                pc_id = build_data.get_pubchemid_from_inchi(inchi)
                pc_id_sdf_path = f"{sdf_path}{pc_id}.sdf"
                if not os.path.exists(pc_id_sdf_path):
                    build_data.download_sdf_pubchem(pc_id,sdf_path, sdf_type=build_data.SDFType.THREE_D)
                
                
            except Exception as e:
                pc_id_sdf_path = None

            # Do directly the copy of all elements of the row
            descriptors = build_data.get_descriptors(aDesc,mol_structure_path=pc_id_sdf_path)
            partialDictDescriptorsRow = row.copy()
            
            for i in range(0,len(listDescriptors)):
                descriptor_header = listDescriptors[i]
                partialDictDescriptorsRow[descriptor_header] = descriptors[i]
            writerDescriptors.writerow(partialDictDescriptorsRow)
            partialDictDescriptorsAndFingerprintsRow = partialDictDescriptorsRow.copy()
            partialDictDescriptorsAndFingerprintsVectorizedRow = partialDictDescriptorsRow.copy()
            # Add fingerprints
            fingerprint_ecfp = build_data.get_fingerprint(aDesc,mol_structure_path=pc_id_sdf_path, fingerprint_type='ECFP')
            fingerprint_maccs = build_data.get_fingerprint(aDesc,mol_structure_path=pc_id_sdf_path, fingerprint_type='MACCSFP')
            fingerprint_pfp = build_data.get_fingerprint(aDesc,mol_structure_path=pc_id_sdf_path, fingerprint_type='PFP')
            try:
                fingerprint_morgan = build_data.get_morgan_fingerprint_rdkit(chemicalStructureFile=pc_id_sdf_path)
            except Exception as e: 
                fingerprint_morgan = "NA"
            partialDictDescriptorsAndFingerprintsRow['ECFP'] = fingerprint_ecfp
            partialDictDescriptorsAndFingerprintsRow['MACCSFP'] = fingerprint_maccs
            partialDictDescriptorsAndFingerprintsRow['PFP'] = fingerprint_pfp
            
            partialDictDescriptorsAndFingerprintsRow['MorganFP'] = fingerprint_morgan
            writerDescriptorsAndFingerprints.writerow(partialDictDescriptorsAndFingerprintsRow)

            vector_fingerprints = build_data.generate_vector_fingerprints(aDesc,mol_structure_path=pc_id_sdf_path)
            partialDictFP = row.copy()
            for i in range(0,NUMBER_FPVALUES):
                header_name = "V" + str(i+1)
                partialDictFP[header_name] = vector_fingerprints[i]
                partialDictDescriptorsAndFingerprintsVectorizedRow[header_name] = vector_fingerprints[i]
            writerFingerprintsVectorized.writerow(partialDictFP)
            writerDescriptorsAndFingerPrintsVectorized.writerow(partialDictDescriptorsAndFingerprintsVectorizedRow)


if __name__ == "__main__":
    main()
