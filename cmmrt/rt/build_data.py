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

import urllib.request,urllib.error, json 
import time
import os
import requests
from alvadesccliwrapper.alvadesc import AlvaDesc
from typing import Tuple
from typing import Any

NUMBER_FPVALUES = 2214
NUMBER_DESCRIPTORS = 5666
ALVADESC_LOCATION = '/usr/bin/alvaDescCLI'
outputPath = '/home/ceu/research/repos/cmm_rt_shared/metlin_ims/'

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FastFindRings
from enum import Enum


class SDFType(Enum):
    TWO_D = 2
    THREE_D = 3

class FingerprintType(Enum):
    '''
    Enum class to specify the type of fingerprint. ECFP, MACCSFP or PFP are from AlvaDesc
    MORGAN are RDKIT fingerprints
    MAP4 are from Reymond group (https://github.com/reymond-group/map4)
    MHFP6 are from Reymond group (https://github.com/reymond-group/mhfp)
    '''
    ECFP = "ECFP"
    MACCSFP = "MACCSFP"
    PFP = "PFP"
    MORGAN = "MORGAN"
    MAP4 = "MAP4"
    MHFP6 = "MHFP6"
    

def list_of_ints_from_str(big_int_str):
    ints_list = [int(d) for d in str(big_int_str)]
    return ints_list

def is_a_lipid_from_classyfire(inchi, inchi_key):
    """ 
        check if the inchi key is a lipid according to classyfire classification. First it uses the gnps2 classyfire endpoint. If it is not there, it goes to the classyfire one. It uses inchi key first, if not classified, inchi. 

        Syntax
        ------
          boolean = is_a_lipid_from_classyfire(inchi, inchi_key)

        Parameters
        ----------

            [in] inchi: string with the inchi of a compound
            [in] inchi_key: string with the inchi key of a compound

        Returns
        -------
          boolean stating wether this inchi key is a lipid according to classyifire classification

        Exceptions
        ----------
          None

        Example
        -------
          >>> is_a_lipid_from_classyfire = is_a_lipid_from_classyfire("RDHQFKQIGNGIED-UHFFFAOYSA-N")
    """

    # http://classyfire.wishartlab.com/entities/BSYNRYMUTXBXSQ-UHFFFAOYSA-N.json
    # https://structure.gnps2.org/classyfire?inchikey=InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N

    url_gnps2_classyfire = "https://structure.gnps2.org/classyfire?inchikey=" + inchi_key
    tried_gnsp2_inchi_key = False
    tried_gnsp2_inchi = False
    retries = 0
    while True:
        try:
            with urllib.request.urlopen(url_gnps2_classyfire) as jsonclassyfire:
                response_code = jsonclassyfire.getcode()

                if response_code == 200:
                    data = json.load(jsonclassyfire)
                    superclass = data["superclass"]["name"]
                    if superclass == "Lipids and lipid-like molecules":
                        return True
                    else:
                        return False
                    #print(data)
                # If the data is not in gnps2, then we hit the classyfire endpoint
                else:
                    if not tried_gnsp2_inchi_key:
                        tried_gnsp2_inchi_key = True
                        url_gnps2_classyfire = "https://structure.gnps2.org/classyfire?inchi=" + inchi
                    elif not tried_gnsp2_inchi:
                        tried_gnsp2_inchi = True
                        url_gnps2_classyfire = "http://classyfire.wishartlab.com/entities/" + inchi_key + ".json"
        except urllib.error.HTTPError as exception:
            if exception.code == 400 or exception.code == 500:
                if not tried_gnsp2_inchi_key:
                    tried_gnsp2_inchi_key = True
                    url_gnps2_classyfire = "https://structure.gnps2.org/classyfire?inchi=" + inchi
                elif not tried_gnsp2_inchi:
                    tried_gnsp2_inchi = True
                    url_gnps2_classyfire = "http://classyfire.wishartlab.com/entities/" + inchi_key + ".json"
                else:
                    raise exception
            elif exception.code == 429:
                if retries < 3:
                    print("Too many requests. Try in 5 seconds")
                    print(exception)
                    time.sleep(5)
            else:
                raise exception


def is_in_lipidMaps(inchi_key):
    """ 
        check if the inchi key is present in lipidmaps

        Syntax
        ------
          boolean = is_in_lipidMaps(inchi_key)

        Parameters
        ----------
            [in] inchi_key: string with the inchi key of a compound

        Returns
        -------
          boolean stating wether the inchi key is present in lipidmaps or not

        Exceptions
        ----------
          Exception if the lipidmaps endpoint is not accesible

        Example
        -------
          >>> is_in_lipidMaps2 = is_in_lipidMaps("RDHQFKQIGNGIED-UHFFFAOYSA-N")
    """
    try:
        lm_id = get_lm_id_from_inchi_key(inchi_key)
        return True
    except ValueError as ve:
        return False
    except Exception as e:
        raise e

def download_sdf_pubchem(pc_id, output_path, sdf_type:SDFType = SDFType.THREE_D):
    """ 
        Get SDF file from the pubchem identifier. It retries the call 3 times if the request is not responded.

        Syntax
        ------
          str = download_sdf_pubchem(pc_id, output_path)

        Parameters
        ----------
            [in] pc_id: PC_ID integer corresponding to the pubchem identifier
            [out] output_path: file path to save the corresponding {pc_id}.sdf file
            [in] sdf_type: SDFType enum to specify the type of SDF file [TWO_D, THREE_D]

        Returns
        -------
            None

        Exceptions
        ----------
          Exception:
            If the pubchem identifier is not present in pubchem database it will reraise the exception

        Example
        -------
          >>> inchi_key = get_inchi_key_from_pubchem(1,'.')
    """
    url_pubchem="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + str(pc_id) + "/SDF"
    if sdf_type == SDFType.TWO_D:
        pass
    else:
        url_pubchem = url_pubchem + "?record_type=3d"

    with urllib.request.urlopen(url_pubchem) as response:
        content = response.read().decode("utf-8")

        file_path = f"{output_path}/{pc_id}.sdf"

        with open(file_path, "w") as file:
            file.write(content)
            return

def get_inchi_from_smiles(smiles):
    """
    Generates the InChI (International Chemical Identifier) from a given SMILES string.

    Syntax:
        inchi = get_inchi_from_smiles(smiles)

    Parameters:
        smiles (str): A SMILES (Simplified Molecular Input Line Entry System) string representing the chemical structure.

    Returns:
        str: The corresponding InChI representation of the chemical structure.

    Exceptions:
        None

    Example:
        >>> get_inchi_from_smiles("CCO")
        'InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3'

    Note:
        This function relies on the RDKit library for chemical structure manipulation.
    """
    mol = Chem.MolFromSmiles(smiles)

    inchi = Chem.MolToInchi(mol, options='-SNon')
    return inchi

def get_pubchemid_from_inchi(inchi):
    """ 
        Get SDF file from the inchi. It retries the call 3 times if the request is not responded.

        Syntax
        ------
          str = download_sdf_pubchem_from_inchi(inchi)

        Parameters
        ----------
            [in] pc_id: PC_ID integer corresponding to the pubchem identifier
            [out] output_path: file path to save the corresponding {pc_id}.sdf file
            [in] sdf_type: SDFType enum to specify the type of SDF file [TWO_D, THREE_D]

        Returns
        -------
            None

        Exceptions
        ----------
          Exception:
            If the inchi is not found in pubchem

        Example
        -------
          >>> inchi_key = get_inchi_key_from_pubchem("InChI=1S/C7H6O2/c8-7(9)6-4-2-1-3-5-6/h1-5H,(H,8,9)")
    """
    # URL for the POST request
    url_pubchem = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/cids/TXT"

    # Body of the POST request
    post_body_data = {'inchi': inchi}

    # Making the POST request
    response = requests.post(url_pubchem, data=post_body_data)

    # Extracting CID from the response
    cid = response.text.strip()  # Removing leading/trailing whitespace
    if cid:
        return int(cid)
    else:
        raise Exception('INCHI NOT FOUND: ' + inchi + ' in the url ' + url_pubchem)

def get_inchi_and_inchi_key_from_pubchem(pc_id):
    """ 
        Get inchi and inchi key from the pubchem identifier. It retries the call 3 times if the request is not responded.

        Syntax
        ------
          str = get_inchi_and_inchi_key_from_pubchem(pc_id)

        Parameters
        ----------
            [in] pc_id: PC_ID integer corresponding to the pubchem identifier

        Returns
        -------
          str containing the inchi 
          str containing the inchi key

        Exceptions
        ----------
          Exception:
            If the pubchem identifier is not present in pubchem database

        Example
        -------
          >>> inchi_key = get_inchi_key_from_pubchem(1)
    """
    url_pubchem="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + str(pc_id) + "/property/InChI,InChIKey/JSON"
    retries = 0
    while retries < 5:
        try:
            with urllib.request.urlopen(url_pubchem) as jsonpubchem:
                data = json.load(jsonpubchem)
                data_compound = data["PropertyTable"]["Properties"][0]
                #print(data)
                try:
                    if 'InChI' not in data_compound or 'InChIKey' not in data_compound:
                        raise Exception('INCHI NOT FOUND')
                    else:
                        inchi = data_compound["InChI"]
                        inchi_key = data_compound["InChIKey"]
                        return inchi, inchi_key
                except Exception as e:
                    raise Exception('INCHI or INCHI KEY NOT FOUND at ' + url_pubchem)
        except urllib.error.HTTPError as e:
            print(e)
            time.sleep(2)
            retries +=1
        except Exception as e:
            print("Connection error to PUBCHEM" + str(e))
            time.sleep(5)
            retries +=1
    print("NOT FOUND URL")
    raise Exception('HTTP NOT FOUND: ' + url_pubchem)

def get_lm_id_from_inchi_key(inchi_key):
    """ 
        Get lm_id from inchi key

        Syntax
        ------
          str = get_lm_id_from_inchi_key(inchi_key)

        Parameters
        ----------
            [in] inchi_key: string with the inchi key of a compound

        Returns
        -------
          str containing the lm id

        Exceptions
        ----------
          Exception:
            ValueError: If the inchi key is not in the lipid maps database
            HttpError: if the url cannot be resolved or the lipidmaps server is down

        Example
        -------
          >>> lm_id = get_lm_id_from_inchi_key("RDHQFKQIGNGIED-UHFFFAOYSA-N")
    """
    url_lipidmaps="https://www.lipidmaps.org/rest/compound/inchi_key/" + inchi_key + "/all"
    while True:
        with urllib.request.urlopen(url_lipidmaps) as jsonLipidMaps:
            data = json.load(jsonLipidMaps)
            if 'lm_id' not in data:
                raise ValueError('LM ID from INCHI KEY ' + inchi_key + ' NOT FOUND')
            else:
                lm_id = data["lm_id"]
                return lm_id
        



def get_file_type(mol_structure_path):
    """ 
        Get the file type from a chemical structure file extension

        Syntax
        ------
          str = get_file_type(mol_structure_path)

        Parameters
        ----------
            [in] mol_structure_path: Chemical structure file with extension smiles, mol, sdf, mol2 or hin

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
    if(mol_structure_path.lower().endswith(".smiles")):
        return "SMILES"
    elif(mol_structure_path.lower().endswith(".mol") or mol_structure_path.lower().endswith(".sdf")):
        return "MDL"
    elif(mol_structure_path.lower().endswith(".mol2")):
        return "SYBYL"
    elif(mol_structure_path.lower().endswith(".hin")):
        return "HYPERCHEM"
    else:
        raise TypeError("File formats recognized are smiles, mol, sdf, mol2 or hin")

def check_fingerprint_type(fingerprintType):
    """ 
        Check the type of fingerprints belongs to ECFP, MACCSFP or PFP

        Syntax
        ------
          str = check_fingerprint_type(fingerprintType)

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


def get_morgan_fingerprint_rdkit(smiles):
    """
    Generates the Morgan fingerprint from a given SMILES string.
    
    Syntax:
        fingerprint = get_morgan_fingerprint_rdkit(smiles)
    Parameters:
        smiles (str): A SMILES (Simplified Molecular Input Line Entry System) string representing the chemical structure.
    Returns:
        str: The corresponding Morgan fingerprint of the chemical structure.
    Exceptions:
        ValueError if smiles does not correspond to a molecule.

    """

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol.UpdatePropertyCache()
    FastFindRings(mol)

    generator = AllChem.GetMorganGenerator(radius=2, fpSize=1024) 
    fp = generator.GetFingerprint(mol)

    return fp.ToBitString()

def get_map4_fingerprint(smiles):
    """
    NOT WORKING BECAUSE OF THE IMPORT OF TMAP (https://github.com/reymond-group/tmap)
    Generates the MAP4 fingerprint from a given SMILES string.

    Syntax:
        fingerprint = get_map4_fingerprint(smiles)

    Parameters:
        smiles (str): A SMILES (Simplified Molecular Input Line Entry System) string representing the chemical structure.

    Returns:
        str: The corresponding MAP4 fingerprint of the chemical structure.

    Exceptions:
        ValueError if smiles does not correspond to a molecule.

    Example:
        >>> get_map4_fingerprint("CCO")
        
    """

    from map4 import MAP4Calculator
    dim = 1024

    MAP4 = MAP4Calculator(dimensions=dim)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol.UpdatePropertyCache()
    FastFindRings(mol)

    map4 = MAP4.calculate(mol)
    fp = MAP4.calculate(mol)
    return fp

def get_mhfp6_fingerprint(smiles):
    """
    Generates the MHFP6 fingerprint from a given SMILES string.

    Syntax:
        fingerprint = get_mhfp6_fingerprint(smiles)

    Parameters:
        smiles (str): A SMILES (Simplified Molecular Input Line Entry System) string representing the chemical structure.

    Returns:
        str: The corresponding MHFP6 fingerprint of the chemical structure.

    Exceptions:
        ValueError if smiles does not correspond to a molecule.

    Example:
        >>> get_mhfp6_fingerprint("CCO")
        
    """

    from mhfp.encoder import MHFPEncoder
    mhfp_encoder = MHFPEncoder()
    
    fp = mhfp_encoder.encode(smiles)
    return fp


def get_fingerprint(aDesc, mol_structure_path=None, smiles=None, fingerprint_type: FingerprintType = FingerprintType.ECFP, fingerprint_size = 1024):
    """ 
        Generate the the specified type Fingerprint from a molecule structure file

        Syntax
        ------
          str = get_fingerprint(aDesc, mol_structure_path, smiles, fingerprint_type, fingerprint_size)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance: instance of the aDesc client
            [in] mol_structure_path (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] smiles (str): SMILES representing the molecule
            [in] fingerprint_type (str): 'ECFP' or 'PFP' or 'MACCSFP'
            [in] fingerprint_size (int): it's not used for MACCS and by default is 1024
        Returns
        -------
          str fingerprint

        Exceptions
        ----------
          TypeError:
            If the mol_structure_path is not smiles, mol, sdf, mol2 or hin
            If the fingerprint_type is not ECFP, PFP or MACCSFP

          RuntimeError:
            If aDesc gets an error calculating the fingerprints

        Example
        -------
          >>> pfp_fingerprint = get_fingerprint((ALVADESC_LOCATION),outputPath + "1.sdf", None, 'PFP')
    """
    if mol_structure_path==None and smiles != None:
        aDesc.set_input_SMILES(smiles)
    elif mol_structure_path != None:
        file_type = get_file_type(mol_structure_path)
        aDesc.set_input_file(mol_structure_path, file_type)
    else:
        raise ValueError("SDF or SMILES should be specified")
    if not aDesc.calculate_fingerprint(fingerprint_type.value, fingerprint_size):
        raise RuntimeError('AlvaDesc Error ' + aDesc.get_error())
    else:
        fingerprint = aDesc.get_output()[0]
        return fingerprint

def get_descriptors(aDesc, mol_structure_path=None, smiles=None):
    """ 
        Generate all the descriptors from a molecule structure file

        Syntax
        ------
          [obj] = get_descriptors(aDesc, mol_structure_path)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance): instance of the aDesc client
            [in] mol_structure_path (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] smiles (str): SMILES representing the molecule
        Returns
        -------
          [obj] descriptors

        Exceptions
        ----------
          TypeError:
            If the mol_structure_path is not smiles, mol, sdf, mol2 or hin

          RuntimeError:
            If aDesc gets an error calculating the descriptors

        Example
        -------
          >>> descriptors = get_descriptors(AlvaDesc(ALVADESC_LOCATION),outputPath + "1.sdf")
    """
    if mol_structure_path==None and smiles != None:
        aDesc.set_input_SMILES(smiles)
    elif mol_structure_path != None:
        file_type = get_file_type(mol_structure_path)
        aDesc.set_input_file(mol_structure_path, file_type)
    else:
        raise ValueError("SDF or SMILES should be specified")
    if not aDesc.calculate_descriptors('ALL'):
        raise RuntimeError('AlvaDesc Error ' + aDesc.get_error())
    else:
        descriptors = aDesc.get_output()[0]
        return descriptors

def generate_vector_fingerprints(aDesc, mol_structure_path = None, smiles = None):
    """ 
        Generate an array containing binary values of the fingerprints ECFP, MACCSFP and PFP in in that order. 

        Syntax
        ------
          [obj] = generate_vector_fingerprints(aDesc, mol_structure_path)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance): instance of the aDesc client
            [in] mol_structure_path (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin.
            [in] smiles (str): structure represented by SMILES instead of a file. If it is specified, mol_structure_path is ignored

        Returns
        -------
          array containing the values of the fingerprints ECFP, MACCSFP and PFP

        Exceptions
        ----------
          TypeError:
            If the mol_structure_path is not smiles, mol, sdf, mol2 or hin

          RuntimeError:
            If aDesc gets an error calculating the fingerprints

        Example
        -------
          >>> fingerprints_pubchem1 = generate_vector_fingerprints(AlvaDesc(ALVADESC_LOCATION),outputPath + "1.sdf")
    """

    ECFP_fingerprint = get_fingerprint(aDesc, mol_structure_path, smiles, FingerprintType.ECFP)
    MACCSFP_fingerprint = get_fingerprint(aDesc, mol_structure_path, smiles, FingerprintType.MACCSFP)
    PFP_fingerprint = get_fingerprint(aDesc, mol_structure_path, smiles, FingerprintType.PFP)


    ECFP_ints_list = list_of_ints_from_str(ECFP_fingerprint)
    fingerprints = ECFP_ints_list
    
    MACCSFP_ints_list = list_of_ints_from_str(MACCSFP_fingerprint)
    fingerprints.extend(MACCSFP_ints_list)
    
    PFP_fingerprint = list_of_ints_from_str(PFP_fingerprint)
    fingerprints.extend(PFP_fingerprint)
    return fingerprints

def generate_vector_fps_descs(
    aDesc, 
    mol_structure_path=None, 
    smiles = None, 
    fingerprint_types: Tuple[FingerprintType, ...] = (FingerprintType.ECFP, FingerprintType.MACCSFP, FingerprintType.PFP), 
    descriptors = True):
    """ 
        Generate an array containing binary values of the descriptors and fingerprints ECFP, MACCSFP and PFP in in that order. 

        Syntax
        ------
          [obj] = generate_vector_fps_descs(aDesc, mol_structure_path, fingerprint_types, descriptors)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance): instance of the aDesc client
            [in] mol_structure_path (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] smiles (str): SMILES representing the molecule
            [in] fingerprints (tuple of FingerprintType): Fingerprints to be calculated
            [in] descriptors (Boolean): include ALL descriptors


        Returns
        -------
          array containing the values of the fingerprints and descriptors specified 

        Exceptions
        ----------
          TypeError:
            If the mol_structure_path is not smiles, mol, sdf, mol2 or hin

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
        descriptors_list = get_descriptors(aDesc, mol_structure_path, smiles)
        result_vector.extend(descriptors_list)
    if isinstance(fingerprint_types,tuple):
        for fingerprint_type in fingerprint_types:
            
            fingerprint_str = get_fingerprint(aDesc, mol_structure_path, smiles, fingerprint_type)
            fingerprint_vector = list_of_ints_from_str(fingerprint_str)

            result_vector.extend(fingerprint_vector)
    else:
        raise ValueError("fingerprint_types should be a tuple of elements recognized (ECFP, MACCSFP, PFP)")

    return result_vector

def generate_vector_fingerprints_CSV(aDesc, mol_structure_path, smiles = None, sep=","):
    """ 
        Generate a string containing binary values of the fingerprints ECFP, MACCSFP and PFP in in that order. 

        Syntax
        ------
          str = generate_vector_fingerprints_CSV(aDesc, mol_structure_path, smiles, sep)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance): instance of the aDesc client
            [in] mol_structure_path (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] smiles (str): SMILES representing the molecule
            [in] sep (str): separator for the csv String

        Returns
        -------
          str containing the values of the fingerprints ECFP, MACCSFP and PFP in a csv format (by default the separator is ',')

        Exceptions
        ----------
          TypeError:
            If the mol_structure_path is not smiles, mol, sdf, mol2 or hin

          RuntimeError:
            If aDesc gets an error calculating the fingerprints

        Example
        -------
          >>> csv_fingerprints_pubchem1 = generate_vector_fingerprints_CSV(AlvaDesc(ALVADESC_LOCATION),outputPath + "1.sdf", None, ",")
    """
    fingerprints = generate_vector_fingerprints(aDesc, mol_structure_path)
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



def generate_vector_descriptors_CSV(aDesc, mol_structure_path, smiles = None, sep=","):
    """ 
        Generate a string containing the descriptors in csv

        Syntax
        ------
          str = generate_vector_descriptors_CSV(aDesc, mol_structure_path, sep)

        Parameters
        ----------
            [in] aDesc (AlvaDesc instance): instance of the aDesc client
            [in] mol_structure_path (str): File name containing the Chemical structure represented by smiles, mol, sdf, mol2 or hin
            [in] smiles (str): SMILES representing the molecule
            [in] sep (str): separator for the csv String

        Returns
        -------
          str containing the values of the descriptors in a csv format (by default the separator is ',')

        Exceptions
        ----------
          TypeError:
            If the mol_structure_path is not smiles, mol, sdf, mol2 or hin

          RuntimeError:
            If aDesc gets an error calculating the fingerprints

        Example
        -------
          >>> csv_descriptors_pubchem1 = generate_vector_descriptors_CSV(AlvaDesc(ALVADESC_LOCATION),outputPath + "1.sdf", None, ",")
    """
    descriptors = get_descriptors(aDesc,mol_structure_path, smiles)
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
    

def inchi_to_3d_structure_rdkit(inchi):
    # Convert InChI to RDKit molecule object
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        raise ValueError("Invalid InChI string")
    
    # Add hydrogen atoms to the molecule
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if result == -1:
        result = AllChem.EmbedMolecule(mol, useRandomCoords=True)
    
    # Optimize the 3D structure
    AllChem.UFFOptimizeMolecule(mol)
    
    return mol


def main():
    # VARIABLES FOR TESTINS. CHANGE THE PROGRAM AND FILE PATHS IF YOU RUN IT LOCALLY
    aDescPath = ALVADESC_LOCATION
    aDesc = AlvaDesc(aDescPath)
    sdfPath = '/home/ceu/research/repos/cmm_rt_shared/SDF/'
    sdfPath = sdfPath + "3D/"
    inputFile ="1.sdf"

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
    actualResult = get_fingerprint(aDesc, sdfPath + inputFile, fingerprint_type=FingerprintType.ECFP)
    
    if expResult == actualResult:
        print("Test PASS. ECFP of pubchem id1 correctly calculated")
    else:
        print("Test FAIL. Check the method get_fingerprint(aDesc, structureFileName, 'ECFP')." + " RESULT: " + str(actualResult))
    
    print("=================================================================.")
    print("Test Case 3: MACCSFP of pubchem id1")
    print("=================================================================.")
    expResult = "0000000000000000000000000000010000000000000000001000000000000000000000000100000000001100100010100001000000010001001000000110010000010001000110000101100111101111100100"
    actualResult = get_fingerprint(aDesc, sdfPath + inputFile, fingerprint_type=FingerprintType.MACCSFP)
    
    if expResult == actualResult:
        print("Test PASS. MACCSFP of pubchem id1 correctly calculated")
    else:
        print("Test FAIL. Check the method get_fingerprint(aDesc, structureFileName, 'MACCSFP')." + " RESULT: " + str(actualResult))

    print("=================================================================.")
    print("Test Case 4: PFP of pubchem id1")
    print("=================================================================.")
    expResult = "0000000000000000000000000000000000000000000000000000000000000000000000000000010000100000000010100000010000000000001010000001000000010010000000000000000000010000000000000000000000000100000000000001000000000000000000000000000000000000000000000010100001000000000010000100010000001100010000000000001000100000000100000000100011000000000000100000000001000100000000000000000000000000000000010000010000000000000000001001000000000000100000000000000000000000100000001010000001000000000000000000000101001000000000000000000000000000000000000000000100010000000000000100000000000000000000010010000000000100000000000010000000000010001000001000000000010000000000000000000111000000000000001000000000010000000000000000001000000000000000000000000000000010010000110100001000000000000000000000000000000000000000110000000000000000000001000001000000000000000000000000000000000100000000000000000000100000000010000100000000000010000000010000000010001000001000100000000000000000000000100000000000000001000000000100000000010000000000000001000000000000"
    actualResult = get_fingerprint(aDesc, sdfPath + inputFile, fingerprint_type=FingerprintType.PFP)
    
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
        print("Test FAIL. Check the method generate_vector_fingerprints_CSV(aDesc, mol_structure_path, sep)." + " RESULT: " + str(actualResult))
    
    print("=================================================================.")
    print("Test Case 7A: Fingerprints of SMILES")
    print("=================================================================.")
    smiles = 'CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C'
    expResult = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    
    actualResult = generate_vector_fingerprints(aDesc, smiles = smiles)
    
    
    if expResult == actualResult:
        print("Test PASS. The CSV Vector from fingerprints has been correctly implemented.")
    else:
        print("Test FAIL. Check the method generate_vector_fingerprints_CSV(aDesc,  smiles = smiles, sep)." + " RESULT: " + str(actualResult))


    print("=================================================================.")
    print("Test Case 7B: Morgan Fingerprints of SMILES")
    print("=================================================================.")
    expResult = "0100000000010000000000000000000001000000000000000000000000000000000001000000000010000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000100000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000001100000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000010000000000000000000000000000000000000000000010000000000000000000100000000000000000000000000000000000000000000000000000000000000000000010000000000000000000001000000000000000000000000000000100000010000000000000000000000000000001000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000001000000"
    
    actualResult = get_morgan_fingerprint_rdkit(smiles = smiles)
    
    
    if expResult == actualResult:
        print("Test PASS. The Morgan Fingerprint has been correctly implemented.")
    else:
        print("Test FAIL. Check the method get_morgan_fingerprint_rdkit(smiles)." + " RESULT: " + str(actualResult))
    
    
    print("=================================================================.")
    print("Test Case 7C: MAP4 Fingerprints of SMILES")
    print("=================================================================.")
    
    expResult = ""

    actualResult = get_map4_fingerprint(smiles = smiles)
    
    if expResult == actualResult:
        print("Test PASS. The MAP4 Fingerprint has been correctly implemented.")
    else:
        print("Test FAIL. Check the method get_map4_fingerprint_rdkit(smiles)." + " RESULT: " + str(actualResult))
    '''
    print("=================================================================.")
    print("Test Case 7D: MHFP6 Fingerprints of SMILES")
    print("=================================================================.")
    expResult = ""

    actualResult = get_mhfp6_fingerprint(smiles = smiles)
    print(actualResult)
    if expResult == actualResult:
        print("Test PASS. The MAP4 Fingerprint has been correctly implemented.")
    else:
        print("Test FAIL. Check the method get_map4_fingerprint_rdkit(smiles)." + " RESULT: " + str(actualResult))
    '''
    print("=================================================================.")
    print("Test Case 8: Descriptors and Fingerprints of pubchem id1")
    print("=================================================================.")

    expResult = NUMBER_DESCRIPTORS + NUMBER_FPVALUES
    descriptors_and_fingerprints = generate_vector_fps_descs(aDesc,sdfPath + inputFile, descriptors = True)
    
    if expResult == len(descriptors_and_fingerprints):
        print("Test PASS. Descriptors and Fingerprints of pubchem id1 has been correctly implemented.")
    else:
        print("EXP RESULT:", expResult)
        print("ACTUAL RESULT", len(descriptors_and_fingerprints))



    print("=================================================================.")
    print("Test Case 9: CSV Fingerprints of pubchem id1")
    print("=================================================================.")
    expResult = "0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0"
    
    actualResult = generate_vector_fingerprints_CSV(aDesc, sdfPath + inputFile)
    
    if expResult == actualResult:
        print("Test PASS. The CSV Vector from fingerprints has been correctly implemented.")
    else:
        print("Test FAIL. Check the method generate_vector_fingerprints_CSV(aDesc, mol_structure_path, sep)." + " RESULT: " + str(actualResult))


    print("=================================================================.")
    print("Test Case 10: CSV Descriptors of pubchem id1. TEST NOT DONE")
    print("=================================================================.")

    print("=================================================================.")
    print("Test Case 11: Checking fingerpints types")
    print("=================================================================.")

    print("=================================================================.")
    print("Test Case 12: Checking INCHI KEY from Pubchem ID ")
    print("=================================================================.")
    try: 
        inchi, inchi_key = get_inchi_and_inchi_key_from_pubchem(1)
        if inchi_key == "RDHQFKQIGNGIED-UHFFFAOYSA-N":
            print("Test PASS Checking inchi key from pubchem ")
        else: 
            print("Test FAIL. Check the INCHI KEY of pubchem 1" + e)
    except Exception as e:
        print("Test FAIL. Check the call to pubchem API" + e)
    try: 
        inchi, inchi_key = get_inchi_and_inchi_key_from_pubchem("asd")
        print("Test FAIL. Check the method get_inchi_and_inchi_key_from_pubchem(inchi_key)")
    except Exception as e:
        print("Test PASS Checking wrong inchi keys. ")
    
    print("=================================================================.")
    print("Test Case 13: Checking LM_ID from INCHI KEY")
    print("=================================================================.")
    try: 
        lm_id = get_lm_id_from_inchi_key("RDHQFKQIGNGIED-UHFFFAOYSA-N")
        if lm_id == "LMFA07070060":
            print("Test PASS Checking fingerpints types. ")
        else: 
            print("Test FAIL. Check the LM ID of inchi key RDHQFKQIGNGIED-UHFFFAOYSA-N" + e)
    except Exception as e:
        print("Test FAIL. Check the LM ID of inchi key RDHQFKQIGNGIED-UHFFFAOYSA-N" + e)
    try: 
        lm_id = get_lm_id_from_inchi_key("asd")
        print("Test FAIL. Check the LM ID of inchi key RDHQFKQIGNGIED-UHFFFAOYSA-N" + e)
    except Exception as e:
        print("Test PASS Checking wrong inchi keys in LM ID. ")
    
    print("=================================================================.")
    print("Test Case 14: Checking Classyfire classification is a lipid")
    print("=================================================================.")
    try: 
        is_lipid_from_classyfire = is_a_lipid_from_classyfire("InChI=1S/C9H17NO4/c1-7(11)14-8(5-9(12)13)6-10(2,3)4/h8H,5-6H2,1-4H3","RDHQFKQIGNGIED-UHFFFAOYSA-N")
        if is_lipid_from_classyfire:
            print("Test PASS Checking lipids in classyfire")
        else: 
            print("Test FAIL. Check lipids in classyfire for INCHI KEY RDHQFKQIGNGIED-UHFFFAOYSA-N")
    except Exception as e:
        print("Test FAIL. Check lipids in classyfire" + str(e))
    try: 
        is_lipid_from_classyfire = is_a_lipid_from_classyfire("InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)","BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        if is_lipid_from_classyfire:
            print("Test FAIL Checking lipids in classyfire with a lipid where is not. Check BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        else: 
            print("Test PASS. Check lipids in classyfire")
    except Exception as e:
        print("Test PASS Checking wrong inchi keys in CLASSYFIRE ")
    try: 
        is_a_lipid = is_a_lipid_from_classyfire("asd","asd")
        print("Test FAIL. Check the classifcation of inchi key" + str(e))
    except Exception as e:
        print("Test PASS Checking wrong inchi keys in CLASSYFIRE ")
    try: 
        is_a_lipid = is_a_lipid_from_classyfire("InChI=1S/C6H9N3S/c1-3-4-8-6(10-2)9-5-7/h3H,1,4H2,2H3,(H,8,9)", "QTNZEFGUDULPSY-UHFFFAOYSA-N")
        print("Test FAIL. Check the classification of inchi key" + str(e))
    except Exception as e:
        if e.code == 500:
            print("Test PASS Checking wrong inchi keys in CLASSYFIRE of a compound with inchi key QTNZEFGUDULPSY-UHFFFAOYSA-N")
        else:
            print("Test FAIL. Check the LM ID of inchi key" + str(e))
    try: 
        is_a_lipid = is_a_lipid_from_classyfire("InChI=1S/C16H20FN3O3S/c1-12(2)24(21,22)20-10-8-16(17,9-11-20)15-18-14(19-23-15)13-6-4-3-5-7-13/h3-7,12H,8-11H2,1-2H3", "DSMCTAYHDQSAIU-UHFFFAOYSA-N")
        if is_a_lipid:
            print("Test FAIL. Check the classification of inchi key: ")
        else:
            print("Test PASS Checking not a lipid DSMCTAYHDQSAIU-UHFFFAOYSA-N")
    except Exception as e:
        if e.code == 500:
            print("Test WRONG checking if DSMCTAYHDQSAIU-UHFFFAOYSA-N is a lipid")
        else:
            print("Test FAIL. Check the LM ID of inchi key" + + str(e))
    print("=================================================================.")
    print("Test Case 15: Checking download SDF from pubchem")
    print("=================================================================.")
    try: 
        
        download_sdf_pubchem(2,sdfPath)
        print("Test PASS Download SDF of Pubchem id: 2")
    except Exception as e:
        print("Test FAIL. Check lipids in classyfire" + str(e))
    
    print("=================================================================.")
    print("Test Case 16: Get PC ID From INCHI")
    print("=================================================================.")
    try: 
        
        pc_id = get_pubchemid_from_inchi("InChI=1S/C7H6O2/c8-7(9)6-4-2-1-3-5-6/h1-5H,(H,8,9)")
        if int(pc_id) == 243:
            print("Test PASS Download SDF of Pubchem id: 2")
        else:
            print("Test FAIL. Check the pubchem id of inchi key")
    except Exception as e:
        print("Test FAIL. Check lipids in classyfire" + str(e))


    print("=================================================================")
    print("Test Case 17: Get InChI From SMILES")
    print("=================================================================")
    try:
        smiles = "CCO"
        expected_inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        obtained_inchi = get_inchi_from_smiles(smiles)
        
        if obtained_inchi == expected_inchi:
            print("Test PASS: Obtained InChI matches the expected InChI.")
        else:
            print("Test FAIL: Obtained InChI does not match the expected InChI.")
            print("Expected InChI:", expected_inchi)
            print("Obtained InChI:", obtained_inchi)
    except Exception as e:
        print("Test FAIL: An exception occurred.")
        print(e)
    

if __name__ == "__main__":
    main()











