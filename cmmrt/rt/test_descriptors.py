#!/usr/bin/env python
# -*- coding: utf-8 -*-

 
"""
@contents :  This module contains functions to generate fingerprints and descriptors using alvaDesc program from the METLIN IMS CCS database
@project :  cmmrt (CEU Mass Mediator Retention Time)
@program :  CEU Mass Mediator
@file :  build_data_metlin_ims.py
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
import math
import os
import build_data

#ALVADESC_LOCATION = 'C:/"Program Files"/Alvascience/alvaDesc/alvaDescCLI.exe'
ALVADESC_LOCATION = '/usr/bin/alvaDescCLI'

def main():
    aDesc = AlvaDesc(ALVADESC_LOCATION)

    # List of descriptors from a mock compound
    smiles="O=C(NCc1ccc(cc1)F)NCCCN1CCc2c1cccc2"
    aDesc.set_input_SMILES(smiles)
    aDesc.calculate_descriptors('ALL')
    listDescriptors = aDesc.get_output_descriptors()
    print(listDescriptors)

    outputPath = '/home/ceu/research/repos/cmm_rt_shared/metlin_ims/'
    sdf_path = f"{outputPath}SDF/"
    smiles = "O=C1CC(CN1c1ccc(c(c1)Cl)C)c1onc(n1)c1ccc2c(c1)OCO2"
    pc_id = 16955977
    try:
        pc_id_sdf_path = f"{sdf_path}{pc_id}.sdf"
        if not os.path.exists(pc_id_sdf_path):
            build_data.download_sdf_pubchem(pc_id,sdf_path)
        
    except Exception as e:
        pc_id_sdf_path = None

    descriptors_sdf = build_data.get_descriptors(aDesc,mol_structure_path=pc_id_sdf_path)
    descriptors_smiles = build_data.get_descriptors(aDesc,smiles=smiles)
    # Enumerate all descriptors

    for index, descriptor in enumerate(listDescriptors, start=0):
        if math.isnan(descriptors_sdf[index]) and math.isnan(descriptors_smiles[index]):
            continue
        # check if the descriptors are the same if they are float with a tolerance of 0.00001
        if isinstance(descriptors_sdf[index], float) and math.isclose(descriptors_sdf[index], descriptors_smiles[index], abs_tol=0.00001):
            continue

        if descriptors_sdf[index] != descriptors_smiles[index]:
            print(f"Error in descriptor: {descriptor}" )
            print(descriptors_sdf[index])
            print(descriptors_smiles[index])


    
if __name__ == "__main__":
    main()
