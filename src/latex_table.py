from coordinates import CoordinateConvertor
import numpy as np
import json


def print_results(results, W, H, input_radians):
    
    
    
    names = ['Brno', 'České Budějovice', 'Dukovany', 'Holešov', 'Nedvězí', 'Olomouc', 'Polom', 'Přibyslav', 'Přimda', 'Temelín']

    convertor = CoordinateConvertor(W, H)
    for i, pos in enumerate(sorted(results)):
        phi, theta = results[pos]["phi"], results[pos]["theta"]
        if input_radians:
            phi, theta = np.rad2deg(phi), np.rad2deg(theta)
        phi, theta = phi%360, theta%360
        fov = np.rad2deg(convertor.f_to_fov(results[pos]["f"]))%360
        print(f'{names[i]:<15} \t & {phi:.2f}\\textdegree  \t & {theta:.2f}\\textdegree  \t & {fov:.2f}\\textdegree  \\\\ \\hline')
    
    print()

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


matlab_results = json.loads(open('../results/matlab-n1000w720h540f0500.json').read())

perez_results = json.loads(open('../results/python-n1000w720h540f0500.json').read())

print_results(p4p_results, 1600, 1200, True)
print_results(matlab_results, 720, 540, False)
print_results(perez_results, 720, 540, False)