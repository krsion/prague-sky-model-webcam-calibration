import numpy as np
from PIL import Image
import functools
import numpy as np
from PIL import Image
from coordinates import CoordinateConvertor
import json
import jsonlines

def iterable_argument_cache(func):
    """Cache decorator for functions with argument being a list (mostly of strings).
    """
    memo = {}

    @functools.wraps(func)
    def wrapper(*args):
        key = tuple(args)  # Convert list of strings to a tuple
        if key not in memo:
            memo[key] = func(*args)
        return memo[key]

    return wrapper

def read_image_greyscale(path: str, width:int, height:int) -> np.ndarray:
    """Unified way of reading greyscale images. 

    Args:
        path (str): path to image
        width (int): width to resize image to
        height (int): height to resize image to

    Returns:
        np.ndarray: HxW matrix of greyscale values
    """
    return np.array(Image.open(path).convert('L').resize((width, height)))

    

                
def latex_table():
    names = ['Brno & 45\\textdegree', 'Č. Budějovice & 270\\textdegree', 'Dukovany & 135\\textdegree', 'Holešov & 270\\textdegree', 'Nedvězí & 180\\textdegree',
             'Olomouc & 180\\textdegree', 'Polom & 270\\textdegree', 'Přibyslav & 225\\textdegree', 'Přimda & 90\\textdegree', 'Temelín & 135\\textdegree']

    
    truth = [45, 270, 135, 270, 180, 180, 270, 225, 90, 135]
        
    def print_results_from_jsonlines(file_path1, file_path2, W, H):
        convertor = CoordinateConvertor(W, H)
        with jsonlines.open(file_path1) as reader1, jsonlines.open(file_path2) as reader2:
            
                for document1, document2, i in zip(reader1, reader2, range(10)):
                    try:
                        fov1 = np.rad2deg(convertor.f_to_fov(document1["f"]))%360
                        print(f'{names[i]:<16}  & {document1["phi"]%360:.2f}\\textdegree  \t & {document1["theta"]:.2f}\\textdegree  \t & {fov1:.2f}\\textdegree', end='')  
                    except:
                        print(f"{names[i]:<16} &&&", end='')
                    try:    
                        fov2 = np.rad2deg(convertor.f_to_fov(document2["f"]))%360
                        print(f'\t & {document2["phi"]%360:.2f}\\textdegree  \t & {document2["theta"]:.2f}\\textdegree  \t & {fov2:.2f}\\textdegree  \\\\ ')
                    except:
                        print("&&& \\\\")
    

    p4p_results = {'dukovany': {'theta': 1.4710038110135613, 'phi': 2.330354043400501, 'f': 1488.7844647180918},
                            'brno': {'theta': 1.1833720772413656, 'phi': 1.2933783429191128, 'f': 1458.200979566561},
                            'ceske_budejovice': {'theta': 1.5005772575612908, 'phi': 4.715353977779227, 'f': 1547.703516624164},
                            'holesov': {'theta': 1.4996441707720347, 'phi': 4.662868837784603, 'f': 1701.44801136993},
                            'nedvezi': {'theta': 1.338972649124866, 'phi': 1.5283634662991692, 'f': 2410.5099293982266},
                            'polom': {'theta': 1.44823176585931, 'phi': 2.521434438279009, 'f': 1702.8024823405835},
                            'pribyslav': {'theta': 1.4794873452241486, 'phi': 4.574500859792659, 'f': 1684.317345643372},
                            'primda': {'theta': 1.34178267177191, 'phi': 1.355306554833905, 'f': 4002.438040107163},
                            'olomouc': {'theta': 1.389273629807438, 'phi': 2.863550646590424, 'f': 1371.6869611268887},
                            'temelin': {'theta': 1.4220596645194596, 'phi': 2.4789882569454122, 'f': 1607.9499169483042}
                            }



    print_results_from_jsonlines('chmu-prague-log.jsonl','chmu-prague-clean-log.jsonl', 720, 540)
    #print_results_from_jsonlines('../results/matlab-large-corrected.jsonl','../results/matlab-small-corrected.jsonl', 1280, 720)


if __name__ == "__main__":
    latex_table()