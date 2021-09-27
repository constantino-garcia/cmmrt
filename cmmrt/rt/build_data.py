#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

@contents :  This module contains functions to generate fingerprints and descriptors using alvaDesc program
@project :  cmmrt (CEU Mass Mediator Retention Time)
@program :  CEU Mass Mediator
@file :  build_data.py
@author :  Alberto Gil De la Fuente (alberto.gilf@gmail.com)
           Constantino García Martínez(constantino.garciama@ceu.es)
           

@version :  0.0.1, 27 September 2021
@information : A valid license of AlvaDesc is necessary to generate the descriptors and fingerprints of chemical structures. 

@copyright :  GNU General Public License v3.0
              Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, 
              which include larger works using a licensed work, under the same license. 
              Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
"""

from alvadesccliwrapper.alvadesc import AlvaDesc

def get_file_type(chemicalStructureFile):
    """ 
        Get the file type from a chemical structure file extension

        Syntax
        ------
          [str] = get_file_type(chemicalStructureFile)

        Parameters
        ----------
            [in] chemicalStructureFile: Chemical structure file with extension smiles, mol, sdf, mol2 or hin

        Returns
        -------
          str containing the molecule type format based on the file extension

        Exceptions
        ----------
          TypeError:
            If the file format is not smiles, mol, sdf, mol2 or hin

        Example
        -------
          >>> get_file_type = UserAccount("'C:/Users/alberto.gildelafuent/Desktop/alberto/resources/SMRT/1.sdf'")
    """
    if(chemicalStructureFile.lower().endswith(".smiles")):
        return "SMILES"
    elif(chemicalStructureFile.lower().endswith(".mol") or chemicalStructureFile.lower().endswith(".sdf")):
        return "MDL"
    elif(chemicalStructureFile.lower().endswith(".mol2")):
        return "SYBYL"
    elif(chemicalStructureFile.lower().endswith(".hin")):
        return "HYPERCHEM"
    else:
        raise TypeError("File formats recognized are smiles, mol, sdf, mol2 or hin")


def get_fingerprint(aDesc, chemicalStructureFile, fingerprint_type, fingerprint_size = 1024):
    """ 
        Generate the the specified type Fingerprint from a molecule structure file

        Syntax
        ------
          [str] = get_fingerprint(aDesc, chemicalStructureFile)

        Parameters
        ----------
            [in] aDesc: instance of the aDesc client
            [in] chemicalStructureFile: Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] fingerprint_type: 'ECFP' or 'PFP' or 'MACCSFP'
            [in] fingerprint_size: it's not used for MACCS and by default is 1024
        Returns
        -------
          str fingerprint

        Exceptions
        ----------
          TypeError:
            If the chemicalStructureFile is not smiles, mol, sdf, mol2 or hin
            If the fingerprint_type is not ECFP, PFP or MACCSFP

          RuntimeError:
            If aDesc gets an error calculating the fingerprints

        Example
        -------
          >>> pfp_fingerprint = get_fingerprint(AlvaDesc('C:/"Program Files"/Alvascience/alvaDesc/alvaDescCLI.exe'),"C:/Users/alberto.gildelafuent/Desktop/alberto/resources/SMRT/1.sdf". 'PFP')
    """
    file_type = get_file_type(chemicalStructureFile)
    if not fingerprint_type in ('ECFP','PFP','MACCSFP'):
        raise TypeError("Fingerprint format not valid. It should be ECFP or PFP or MACCSFP")
    aDesc.set_input_file(chemicalStructureFile, file_type)
    if not aDesc.calculate_fingerprint(fingerprint_type, fingerprint_size):
        raise RuntimeError('AlvaDesc Error ' + aDesc.get_error())
    else:
        fingerprint = aDesc.get_output()[0]
        return fingerprint



def get_descriptors(aDesc, chemicalStructureFile):
    """ 
        Generate all the descriptors from a molecule structure file

        Syntax
        ------
          [[obj]] = get_descriptors(aDesc, chemicalStructureFile)

        Parameters
        ----------
            [in] aDesc: instance of the aDesc client
            [in] chemicalStructureFile: Chemical structure represented by smiles, mol, sdf, mol2 or hin
        Returns
        -------
          [obj] descriptors

        Exceptions
        ----------
          TypeError:
            If the chemicalStructureFile is not smiles, mol, sdf, mol2 or hin

          RuntimeError:
            If aDesc gets an error calculating the descriptors

        Example
        -------
          >>> descriptors = get_descriptors(AlvaDesc('C:/"Program Files"/Alvascience/alvaDesc/alvaDescCLI.exe'),"C:/Users/alberto.gildelafuent/Desktop/alberto/resources/SMRT/1.sdf")
    """
    file_type = get_file_type(chemicalStructureFile)
    aDesc.set_input_file(chemicalStructureFile, file_type)
    if not aDesc.calculate_descriptors('ALL'):
        raise RuntimeError('AlvaDesc Error ' + aDesc.get_error())
    else:
        descriptors = aDesc.get_output()[0]
        return descriptors

def generate_vector_fingerprints_CSV(aDesc, chemicalStructureFile, sep=","):
    """ 
        Generate a string containing binary values of the fingerprints ECFP, MACCSFP and PFP in in that order. 

        Syntax
        ------
          [str] = generate_vector_fingerprints_CSV(aDesc, chemicalStructureFile, sep)

        Parameters
        ----------
            [in] aDesc: instance of the aDesc client
            [in] chemicalStructureFile: Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] sep: separator for the csv String

        Returns
        -------
          str containing the values of the fingerprints ECFP, MACCSFP and PFP in a csv format (by default the separator is ',')

        Exceptions
        ----------
          TypeError:
            If the chemicalStructureFile is not smiles, mol, sdf, mol2 or hin

          RuntimeError:
            If aDesc gets an error calculating the fingerprints

        Example
        -------
          >>> csv_fingerprints_pubchem1 = generate_vector_fingerprints_CSV(AlvaDesc('C:/"Program Files"/Alvascience/alvaDesc/alvaDescCLI.exe'),"C:/Users/alberto.gildelafuent/Desktop/alberto/resources/SMRT/1.sdf", ",")
    """
    ECFP_fingerprint = get_fingerprint(aDesc, chemicalStructureFile, 'ECFP')
    MACCSFP_fingerprint = get_fingerprint(aDesc, chemicalStructureFile, 'MACCSFP')
    PFP_fingerprint = get_fingerprint(aDesc, chemicalStructureFile, 'PFP')
    elements_ECFP = list(ECFP_fingerprint)
    elements_MACCSFP = list(MACCSFP_fingerprint)
    elements_PFP = list(PFP_fingerprint)
    str_fingerprints_csv = ""
    for element in elements_ECFP:
        element = int(element)
        if isinstance(element,int) and element in (0,1):
            str_fingerprints_csv = str_fingerprints_csv + str(element) + sep 
        else:
            raise RuntimeError("Fingerprint ECFP bad calculated: " + ECFP_fingerprint)
    for element in elements_MACCSFP:
        element = int(element)
        if isinstance(element,int) and element in (0,1):
            str_fingerprints_csv = str_fingerprints_csv + str(element) + sep 
        else:
            raise RuntimeError("Fingerprint ECFP bad calculated: " + ECFP_fingerprint)
    
    for element in elements_PFP:
        element = int(element)
        if isinstance(element,int) and element in (0,1):
            str_fingerprints_csv = str_fingerprints_csv + str(element) + sep 
        else:
            raise RuntimeError("Fingerprint ECFP bad calculated: " + ECFP_fingerprint)
    
    if(str_fingerprints_csv.endswith(sep)):
	    str_fingerprints_csv = str_fingerprints_csv[0:len(str_fingerprints_csv) - len(sep)]

    return str_fingerprints_csv


def generate_vector_descriptors_CSV(aDesc, chemicalStructureFile, sep=","):
    """ 
        Generate a string containing the descriptors in csv

        Syntax
        ------
          [str] = generate_vector_descriptors_CSV(aDesc, chemicalStructureFile, sep)

        Parameters
        ----------
            [in] aDesc: instance of the aDesc client
            [in] chemicalStructureFile: Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] sep: separator for the csv String

        Returns
        -------
          str containing the values of the descriptors in a csv format (by default the separator is ',')

        Exceptions
        ----------
          TypeError:
            If the chemicalStructureFile is not smiles, mol, sdf, mol2 or hin

          RuntimeError:
            If aDesc gets an error calculating the fingerprints

        Example
        -------
          >>> csv_descriptors_pubchem1 = generate_vector_descriptors_CSV(AlvaDesc('C:/"Program Files"/Alvascience/alvaDesc/alvaDescCLI.exe'),"C:/Users/alberto.gildelafuent/Desktop/alberto/resources/SMRT/1.sdf", ",")
    """
    descriptors = get_descriptors(aDesc,chemicalStructureFile)
    str_descriptors_csv = ""
    for element in descriptors:
        if (isinstance(element,(int,str))):
            str_descriptors_csv = str_descriptors_csv + str(element) + sep
        elif(isinstance(element,float)):
            str_descriptors_csv = str_descriptors_csv + f'{element:.18f}' + sep
        else:
            raise RuntimeError("Descriptor type not recognized: " + str(element) + " TYPE: " + type(element))
    if(str_descriptors_csv.endswith(sep)):
        str_descriptors_csv = str_descriptors_csv[0:len(str_descriptors_csv) - len(sep)]

    return str_descriptors_csv
    

def main():
	# VARIABLES FOR TESTINS. CHANGE THE PROGRAM AND FILE PATHS IF YOU RUN IT LOCALLY
    aDescPath = 'C:/"Program Files"/Alvascience/alvaDesc/alvaDescCLI.exe'
    aDesc = AlvaDesc(aDescPath)
    inputPath = 'C:/Users/alberto.gildelafuent/OneDrive - Fundación Universitaria San Pablo CEU/research/SMRT_in_CMM/'
    outputPath = 'C:/Users/alberto.gildelafuent/Desktop/alberto/resources/SMRT/'
    sdfPath = outputPath + "SDF/"
    inputFile ="1.sdf"

    print("=================================================================.")
    print("Test Case 1: File type of SDF")
    print("=================================================================.")
    expResult = "MDL"
    actualResult = get_file_type(sdfPath + inputFile)
    
    if expResult == actualResult:
        print("Test PASS. Format file extension SDF to MDL is correct")
    else:
        print("Test FAIL. Check the method get_file_type(inputPath)." + " RESULT: " + str(actualResult))
    
    print("=================================================================.")
    print("Test Case 2: ECFP of pubchem id1")
    print("=================================================================.")
    expResult = "0000001000000000000000000000000000000000000000000000000000000000000000000000010000100000000010000000010000000000001011000000000000010000000000000001000000000000000000000000000000000000000000000000000010000000000000000000000000000000000001000010000001000100000010000000000000001100010000000000000000100000000000000001000010000000000000000000000000000100000100000000000000000000000000010000010000000000000000001001000000000000100000000000000000000000100000001000000001000000000000000010000000010000000000000000000000000000000000000000000100000000000000000100000000000000000000000000000000000100000000000010000000000010001000001001000000010000000000000000000011100000000000000000010000010000000000001000001000000000000000000000000000000000000000100000000000000000000000000000000000000000000001110000000000000000000001000001000000000000000000000000000000010100000000000000000100000000000000000000000000000010000000000000001010000000001000101000000000000000000000100000000000000000000000000000000000010000000000000001000000000000"
    actualResult = get_fingerprint(aDesc, sdfPath + inputFile, 'ECFP')
    
    if expResult == actualResult:
        print("Test PASS. ECFP of pubchem id1 correctly calculated")
    else:
        print("Test FAIL. Check the method get_fingerprint(aDesc, inputPath, 'ECFP')." + " RESULT: " + str(actualResult))
    
    print("=================================================================.")
    print("Test Case 3: MACCSFP of pubchem id1")
    print("=================================================================.")
    expResult = "0000000000000000000000000000010000000000000000001000000000000000000000000100000000001100100010100001000000010001001000000110010000010001000110000101100111101111100100"
    actualResult = get_fingerprint(aDesc, sdfPath + inputFile, 'MACCSFP')
    
    if expResult == actualResult:
        print("Test PASS. MACCSFP of pubchem id1 correctly calculated")
    else:
        print("Test FAIL. Check the method get_fingerprint(aDesc, inputPath, 'MACCSFP')." + " RESULT: " + str(actualResult))

    print("=================================================================.")
    print("Test Case 4: PFP of pubchem id1")
    print("=================================================================.")
    expResult = "0000000000000000000000000000000000000000000000000000000000000000000000000000010000100000000010100000010000000000001010000001000000010010000000000000000000010000000000000000000000000100000000000001000000000000000000000000000000000000000000000010100001000000000010000100010000001100010000000000001000100000000100000000100011000000000000100000000001000100000000000000000000000000000000010000010000000000000000001001000000000000100000000000000000000000100000001010000001000000000000000000000101001000000000000000000000000000000000000000000100010000000000000100000000000000000000010010000000000100000000000010000000000010001000001000000000010000000000000000000111000000000000001000000000010000000000000000001000000000000000000000000000000010010000110100001000000000000000000000000000000000000000110000000000000000000001000001000000000000000000000000000000000100000000000000000000100000000010000100000000000010000000010000000010001000001000100000000000000000000000100000000000000001000000000100000000010000000000000001000000000000"
    actualResult = get_fingerprint(aDesc, sdfPath + inputFile, 'PFP')
    
    if expResult == actualResult:
        print("Test PASS. PFP of pubchem id1 correctly calculated")
    else:
        print("Test FAIL. Check the method get_fingerprint(aDesc, inputPath, 'PFP')." + " RESULT: " + str(actualResult))

    
    print("=================================================================.")
    print("Test Case 5: Descriptors of pubchem id1")
    print("=================================================================.")
    expResult_len = 5666
    actualResult = get_descriptors(aDesc, sdfPath + inputFile)
    if expResult_len == len(actualResult):
        print("Test PASS. number of descriptors of pubchem id1 correctly calculated")
    else:
        print("Test FAIL. Check the method get_descriptors(aDesc, inputPath)." + " RESULT: " + str(actualResult))

    
    print("=================================================================.")
    print("Test Case 6: CSV Fingerprints of pubchem id1")
    print("=================================================================.")
    expResult = "0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0"
    
    actualResult = generate_vector_fingerprints_CSV(aDesc, sdfPath + inputFile)
    
    if expResult == actualResult:
        print("Test PASS. The CSV Vector from fingerprints has been correctly implemented.")
    else:
        print("Test FAIL. Check the method generate_vector_fingerprints_CSV(aDesc, chemicalStructureFile, sep)." + " RESULT: " + str(actualResult))


    print("=================================================================.")
    print("Test Case 7: CSV Descriptors of pubchem id1. TEST NOT DONE")
    print("=================================================================.")
    

if __name__ == "__main__":
    main()

