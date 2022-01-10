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

NUMBER_FPVALUES = 2214
NUMBER_DESCRIPTORS = 5666
#ALVADESC_LOCATION = 'C:/"Program Files"/Alvascience/alvaDesc/alvaDescCLI.exe'
ALVADESC_LOCATION = '/usr/bin/alvaDescCLI'
outputPath = 'resources/'


def list_of_ints_from_str(big_int_str):
    ints_list = [int(d) for d in str(big_int_str)]
    return ints_list

def get_file_type(chemicalStructureFile):
    """ 
        Get the file type from a chemical structure file extension

        Syntax
        ------
          str = get_file_type(chemicalStructureFile)

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
          >>> get_file_type = get_file_type(outputPath + "1.sdf")
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

def check_fingerprint_type(fingerprintType):
    """ 
        Check the type of fingerprints belongs to ECFP, MACCSFP or PFP

        Syntax
        ------
          str = check_fingerprint_type(chemicalStructureFile)

        Parameters
        ----------
            [in] fingerprintType (str): type of finreprint

        Returns
        -------
          None

        Exceptions
        ----------
          ValueError:
            If the fingerprint Type is not ECFP, MACCSFP or PFP

        Example
        -------
          >>> check_fingerprint_type("PFP")
    """
    if not fingerprintType.lower() in ("ecfp", "maccsfp","pfp"):
        raise ValueError("Fingerprint Type not recognized. Currently ECFP, MACCSFP and PFP are available.")


def get_fingerprint_from_SMILES(aDesc, SMILES, fingerprint_type, fingerprint_size = 1024):
    """ 
        Generate the the specified type Fingerprint from a SMILES structure

        Syntax
        ------
          str = get_fingerprint_from_SMILES(aDesc, SMILES)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance): instance of the aDesc client
            [in] SMILES (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] fingerprint_type: 'ECFP' or 'PFP' or 'MACCSFP'
            [in] fingerprint_size: it's not used for MACCS and by default is 1024
        Returns
        -------
          str fingerprint

        Exceptions
        ----------
          TypeError:

          RuntimeError:
            If aDesc gets an error calculating the fingerprints

        Example
        -------
          >>> pfp_fingerprint = get_fingerprint_from_SMILES((ALVADESC_LOCATION),"CCCC"", 'PFP')
    """
    if not fingerprint_type in ('ECFP','PFP','MACCSFP'):
        raise TypeError("Fingerprint format not valid. It should be ECFP or PFP or MACCSFP")
    aDesc.set_input_SMILES(SMILES)
    # TESTING A REGULAR SMILES HARDCODED
    #aDesc.set_input_SMILES(['CC(=O)OC1=CC=CC=C1C(=O)O'])
    if not aDesc.calculate_fingerprint(fingerprint_type, fingerprint_size):
        raise RuntimeError('AlvaDesc Error ' + aDesc.get_error() + SMILES)
    else:
        fingerprint = aDesc.get_output()[0]
        return fingerprint

def get_fingerprint(aDesc, chemicalStructureFile, fingerprint_type, fingerprint_size = 1024):
    """ 
        Generate the the specified type Fingerprint from a molecule structure file

        Syntax
        ------
          str = get_fingerprint(aDesc, chemicalStructureFile, fingerprint_type, fingerprint_size)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance: instance of the aDesc client
            [in] chemicalStructureFile (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] fingerprint_type (str): 'ECFP' or 'PFP' or 'MACCSFP'
            [in] fingerprint_size (int): it's not used for MACCS and by default is 1024
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
          >>> pfp_fingerprint = get_fingerprint((ALVADESC_LOCATION),outputPath + "1.sdf", 'PFP')
    """
    file_type = get_file_type(chemicalStructureFile)
    if not fingerprint_type in ('ECFP','PFP','MACCSFP'):
        raise TypeError("Fingerprint format not valid. It should be ECFP or PFP or MACCSFP")
    aDesc.set_input_file(chemicalStructureFile, file_type)
    # TESTING A REGULAR SMILES HARDCODED
    #aDesc.set_input_SMILES(['CC(=O)OC1=CC=CC=C1C(=O)O'])
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
          [obj] = get_descriptors(aDesc, chemicalStructureFile)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance): instance of the aDesc client
            [in] chemicalStructureFile (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin
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
          >>> descriptors = get_descriptors(AlvaDesc(ALVADESC_LOCATION),outputPath + "1.sdf")
    """
    file_type = get_file_type(chemicalStructureFile)
    aDesc.set_input_file(chemicalStructureFile, file_type)
    if not aDesc.calculate_descriptors('ALL'):
        raise RuntimeError('AlvaDesc Error ' + aDesc.get_error())
    else:
        descriptors = aDesc.get_output()[0]
        return descriptors

def generate_vector_fingerprints(aDesc, chemicalStructureFile = None, smiles = None):
    """ 
        Generate an array containing binary values of the fingerprints ECFP, MACCSFP and PFP in in that order. 

        Syntax
        ------
          [obj] = generate_vector_fingerprints(aDesc, chemicalStructureFile)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance): instance of the aDesc client
            [in] chemicalStructureFile (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin.
            [in] SMILES (str): structure represented by SMILES instead of a file. If it is specified, chemicalStructureFile is ignored

        Returns
        -------
          array containing the values of the fingerprints ECFP, MACCSFP and PFP

        Exceptions
        ----------
          TypeError:
            If the chemicalStructureFile is not smiles, mol, sdf, mol2 or hin

          RuntimeError:
            If aDesc gets an error calculating the fingerprints

        Example
        -------
          >>> fingerprints_pubchem1 = generate_vector_fingerprints(AlvaDesc(ALVADESC_LOCATION),outputPath + "1.sdf")
    """
    if smiles == None:
        ECFP_fingerprint = get_fingerprint(aDesc, chemicalStructureFile, 'ECFP')
        MACCSFP_fingerprint = get_fingerprint(aDesc, chemicalStructureFile, 'MACCSFP')
        PFP_fingerprint = get_fingerprint(aDesc, chemicalStructureFile, 'PFP')
    else: 
        ECFP_fingerprint = get_fingerprint_from_SMILES(aDesc, smiles, 'ECFP')
        MACCSFP_fingerprint = get_fingerprint_from_SMILES(aDesc, smiles, 'MACCSFP')
        PFP_fingerprint = get_fingerprint_from_SMILES(aDesc, smiles, 'PFP')

    ECFP_ints_list = list_of_ints_from_str(ECFP_fingerprint)
    fingerprints = ECFP_ints_list
    print(smiles)
    print(fingerprints)
    
    MACCSFP_ints_list = list_of_ints_from_str(MACCSFP_fingerprint)
    fingerprints.extend(MACCSFP_ints_list)
    
    PFP_fingerprint = list_of_ints_from_str(PFP_fingerprint)
    fingerprints.extend(PFP_fingerprint)
    return fingerprints

def generate_vector_fps_descs(aDesc, chemicalStructureFile, fingerprint_types = ("ECFP", "MACCSFP", "PFP"), descriptors = True):
    """ 
        Generate an array containing binary values of the descriptors and fingerprints ECFP, MACCSFP and PFP in in that order. 

        Syntax
        ------
          [obj] = generate_vector_fps_descs(aDesc, chemicalStructureFile, fingerprint_types, descriptors)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance): instance of the aDesc client
            [in] chemicalStructureFile (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] fingerprints (tuple of Strings): Fingerprints to be calculated
            [in] descriptors (Boolean): include ALL descriptors


        Returns
        -------
          array containing the values of the fingerprints and descriptors specified 

        Exceptions
        ----------
          TypeError:
            If the chemicalStructureFile is not smiles, mol, sdf, mol2 or hin

          ValueError:
            If the fingerprints is not a tuple object or the elements of the tuple are not recognized (ECFP, MACCSFP, PFP)
            If descriptors is not a boolean

          RuntimeError:
            If aDesc gets an error calculating the fingerprints

        Example
        -------
          >>> descriptors_and_fingerprints = generate_vector_fps_descs(AlvaDesc(ALVADESC_LOCATION),outputPath + "1.sdf")
    """
    result_vector = []
    if isinstance(descriptors,bool) and descriptors:
        descriptors_list = get_descriptors(aDesc, chemicalStructureFile)
        result_vector.extend(descriptors_list)
    if isinstance(fingerprint_types,tuple):
        for fingerprint_type in fingerprint_types:
            check_fingerprint_type(fingerprint_type)
            fingerprint_str = get_fingerprint(aDesc, chemicalStructureFile, fingerprint_type)
            fingerprint_vector = list_of_ints_from_str(fingerprint_str)

            result_vector.extend(fingerprint_vector)
    else:
        raise ValueError("fingerprint_types should be a tuple of elements recognized (ECFP, MACCSFP, PFP)")

    return result_vector

def generate_vector_fingerprints_CSV(aDesc, chemicalStructureFile, sep=","):
    """ 
        Generate a string containing binary values of the fingerprints ECFP, MACCSFP and PFP in in that order. 

        Syntax
        ------
          str = generate_vector_fingerprints_CSV(aDesc, chemicalStructureFile, sep)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance): instance of the aDesc client
            [in] chemicalStructureFile (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] sep (str): separator for the csv String

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
          >>> csv_fingerprints_pubchem1 = generate_vector_fingerprints_CSV(AlvaDesc(ALVADESC_LOCATION),outputPath + "1.sdf", ",")
    """
    fingerprints = generate_vector_fingerprints(aDesc, chemicalStructureFile)
    str_fingerprints_csv = ""
    for element in fingerprints:
        element = int(element)
        if isinstance(element,int) and element in (0,1):
            str_fingerprints_csv = str_fingerprints_csv + str(element) + sep 
        else:
            raise RuntimeError("Fingerprints bad calculated: " + ECFP_fingerprint)
    
    if(str_fingerprints_csv.endswith(sep)):
        str_fingerprints_csv = str_fingerprints_csv[0:len(str_fingerprints_csv) - len(sep)]

    return str_fingerprints_csv



def generate_vector_descriptors_CSV(aDesc, chemicalStructureFile, sep=","):
    """ 
        Generate a string containing the descriptors in csv

        Syntax
        ------
          str = generate_vector_descriptors_CSV(aDesc, chemicalStructureFile, sep)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance): instance of the aDesc client
            [in] chemicalStructureFile (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] sep (str): separator for the csv String

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
          >>> csv_descriptors_pubchem1 = generate_vector_descriptors_CSV(AlvaDesc(ALVADESC_LOCATION),outputPath + "1.sdf", ",")
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
    aDescPath = ALVADESC_LOCATION
    aDesc = AlvaDesc(aDescPath)
    sdfPath = outputPath + "SDF/"
    inputFile ="1.sdf"
    '''
    print("=================================================================.")
    print("Test Case 1: File type of SDF")
    print("=================================================================.")
    expResult = "MDL"
    actualResult = get_file_type(sdfPath + inputFile)
    
    if expResult == actualResult:
        print("Test PASS. Format file extension SDF to MDL is correct")
    else:
        print("Test FAIL. Check the method get_file_type(structureFileName)." + " RESULT: " + str(actualResult))
    
    print("=================================================================.")
    print("Test Case 2: ECFP of pubchem id1")
    print("=================================================================.")
    expResult = "0000001000000000000000000000000000000000000000000000000000000000000000000000010000100000000010000000010000000000001011000000000000010000000000000001000000000000000000000000000000000000000000000000000010000000000000000000000000000000000001000010000001000100000010000000000000001100010000000000000000100000000000000001000010000000000000000000000000000100000100000000000000000000000000010000010000000000000000001001000000000000100000000000000000000000100000001000000001000000000000000010000000010000000000000000000000000000000000000000000100000000000000000100000000000000000000000000000000000100000000000010000000000010001000001001000000010000000000000000000011100000000000000000010000010000000000001000001000000000000000000000000000000000000000100000000000000000000000000000000000000000000001110000000000000000000001000001000000000000000000000000000000010100000000000000000100000000000000000000000000000010000000000000001010000000001000101000000000000000000000100000000000000000000000000000000000010000000000000001000000000000"
    actualResult = get_fingerprint(aDesc, sdfPath + inputFile, 'ECFP')
    
    if expResult == actualResult:
        print("Test PASS. ECFP of pubchem id1 correctly calculated")
    else:
        print("Test FAIL. Check the method get_fingerprint(aDesc, structureFileName, 'ECFP')." + " RESULT: " + str(actualResult))
    
    print("=================================================================.")
    print("Test Case 3: MACCSFP of pubchem id1")
    print("=================================================================.")
    expResult = "0000000000000000000000000000010000000000000000001000000000000000000000000100000000001100100010100001000000010001001000000110010000010001000110000101100111101111100100"
    actualResult = get_fingerprint(aDesc, sdfPath + inputFile, 'MACCSFP')
    
    if expResult == actualResult:
        print("Test PASS. MACCSFP of pubchem id1 correctly calculated")
    else:
        print("Test FAIL. Check the method get_fingerprint(aDesc, structureFileName, 'MACCSFP')." + " RESULT: " + str(actualResult))

    print("=================================================================.")
    print("Test Case 4: PFP of pubchem id1")
    print("=================================================================.")
    expResult = "0000000000000000000000000000000000000000000000000000000000000000000000000000010000100000000010100000010000000000001010000001000000010010000000000000000000010000000000000000000000000100000000000001000000000000000000000000000000000000000000000010100001000000000010000100010000001100010000000000001000100000000100000000100011000000000000100000000001000100000000000000000000000000000000010000010000000000000000001001000000000000100000000000000000000000100000001010000001000000000000000000000101001000000000000000000000000000000000000000000100010000000000000100000000000000000000010010000000000100000000000010000000000010001000001000000000010000000000000000000111000000000000001000000000010000000000000000001000000000000000000000000000000010010000110100001000000000000000000000000000000000000000110000000000000000000001000001000000000000000000000000000000000100000000000000000000100000000010000100000000000010000000010000000010001000001000100000000000000000000000100000000000000001000000000100000000010000000000000001000000000000"
    actualResult = get_fingerprint(aDesc, sdfPath + inputFile, 'PFP')
    
    if expResult == actualResult:
        print("Test PASS. PFP of pubchem id1 correctly calculated")
    else:
        print("Test FAIL. Check the method get_fingerprint(aDesc, structureFileName, 'PFP')." + " RESULT: " + str(actualResult))

    
    print("=================================================================.")
    print("Test Case 5: Descriptors of pubchem id1")
    print("=================================================================.")
    expResult_len = 5666
    actualResult = get_descriptors(aDesc, sdfPath + inputFile)
    if expResult_len == len(actualResult):
        print("Test PASS. number of descriptors of pubchem id1 correctly calculated")
    else:
        print("Test FAIL. Check the method get_descriptors(aDesc, structureFileName)." + " RESULT: " + str(actualResult))

    

    print("=================================================================.")
    print("Test Case 6: Fingerprints of pubchem id1")
    print("=================================================================.")
    expResult = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    
    actualResult = generate_vector_fingerprints(aDesc, sdfPath + inputFile)
    
    if expResult == actualResult:
        print("Test PASS. The CSV Vector from fingerprints has been correctly implemented.")
    else:
        print("Test FAIL. Check the method generate_vector_fingerprints_CSV(aDesc, chemicalStructureFile, sep)." + " RESULT: " + str(actualResult))
    '''
    print("=================================================================.")
    print("Test Case 7: Fingerprints of SMILES")
    print("=================================================================.")
    expResult = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    actualResult = generate_vector_fingerprints(aDesc, smiles = 'CC(C)=CCC/C(/C)=C\\C=O')
    
    '''
    if expResult == actualResult:
        print("Test PASS. The CSV Vector from fingerprints has been correctly implemented.")
    else:
        print("Test FAIL. Check the method generate_vector_fingerprints_CSV(aDesc, chemicalStructureFile, sep)." + " RESULT: " + str(actualResult))


    print("=================================================================.")
    print("Test Case 8: Descriptors and Fingerprints of pubchem id1")
    print("=================================================================.")

    expResult = NUMBER_DESCRIPTORS + NUMBER_FPVALUES
    descriptors_and_fingerprints = generate_vector_fps_descs(aDesc,sdfPath + inputFile, descriptors = True)
    
    if expResult == len(descriptors_and_fingerprints):
        print("Test PASS. Descriptors and Fingerprints of pubchem id1 has been correctly implemented.")
    else:
        print("EXP RESULT:", expResult)
        print("ACTUAL RESULT", descriptors_and_fingerprints)



    print("=================================================================.")
    print("Test Case 9: CSV Fingerprints of pubchem id1")
    print("=================================================================.")
    expResult = "0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0"
    
    actualResult = generate_vector_fingerprints_CSV(aDesc, sdfPath + inputFile)
    
    if expResult == actualResult:
        print("Test PASS. The CSV Vector from fingerprints has been correctly implemented.")
    else:
        print("Test FAIL. Check the method generate_vector_fingerprints_CSV(aDesc, chemicalStructureFile, sep)." + " RESULT: " + str(actualResult))


    print("=================================================================.")
    print("Test Case 10: CSV Descriptors of pubchem id1. TEST NOT DONE")
    print("=================================================================.")

    print("=================================================================.")
    print("Test Case 11: Checking fingerpints types")
    print("=================================================================.")
    try: 
        check_fingerprint_type("PFP")
    except Exception as e:
        print("Test FAIL. Check the method check_fingerprint_type(fingerprintType). It should accept PFP, ECFP and MACCSFP values" + e)
    try: 
        check_fingerprint_type("OPFP")
        print("Test FAIL. Check the method check_fingerprint_type(fingerprintType). It should not accept any other value than PFP, ECFP and MACCSFP" + e)
    except Exception as e:
        print("Test PASS Checking fingerpints types. ")
    '''

if __name__ == "__main__":
    main()

