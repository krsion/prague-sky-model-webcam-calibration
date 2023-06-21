from calibrator import focal_length_and_zenith, focal_length_and_zenith_on_dataset
import json
import numpy as np


np.random.seed(42)

sun_calibrations = {'dukovany': {'theta': 1.4710038110135613, 'phi': 2.330354043400501, 'f': 1488.7844647180918},
                    'brno': {'theta': 1.1833720772413656, 'phi': 1.2933783429191128, 'f': 1458.200979566561},
                    'ceske_budejovice': {'theta': 1.5005772575612908, 'phi': 4.715353977779227, 'f': 1547.703516624164},
                    'holesov': {'theta': 1.4996441707720347, 'phi': 4.662868837784603, 'f': 1701.44801136993},
                    'nedvezi': {'theta': 1.338972649124866, 'phi': 1.5283634662991692, 'f': 2410.5099293982266},
                    'polom': {'theta': 1.44823176585931, 'phi': 2.521434438279009, 'f': 1702.8024823405835},
                    'pribyslav': {'theta': 1.4794873452241486, 'phi': 4.574500859792659, 'f': 1684.317345643372},
                    'primda': {'theta': 1.34178267177191, 'phi': 1.355306554833905, 'f': 4002.438040107163},
                    'olomouc': {'theta': 1.389273629807438, 'phi': 2.863550646590424, 'f': 1371.6869611268887},
                    'temelin': {'theta': 1.4220596645194596, 'phi': 2.4789882569454122, 'f': 1607.9499169483042}}

results_psm = focal_length_and_zenith_on_dataset(
    'I-train', focal_length_and_zenith, zenith_degrees=True, mode='psm')
print(json.dumps(results_psm))


results_perez = {
    'churanov': {'focalLength': 3091.4525196480818, 'zenithAngle': 87.26953190734883},
    'ceske_budejovice': {'focalLength': 3252.533789688667, 'zenithAngle': 79.42874996645419},
    'belotin': {'focalLength': 3131.8556922880575, 'zenithAngle': 87.93865609846428},
    'cheb': {'focalLength': 2558.049061339774, 'zenithAngle': 83.05320224028507},
    'brno': {'focalLength': 3715.8890320852524, 'zenithAngle': 82.99466257007896},
    'broumov': {'focalLength': 2646.7115778883062, 'zenithAngle': 88.01905888089844},
    'dukovany': {'focalLength': 2676.0056489102717, 'zenithAngle': 87.62474718043924},
    'dylen': {'focalLength': 2979.6668980430704, 'zenithAngle': 89.99999999999895},
    'doksany': {'focalLength': 2844.0300636802885, 'zenithAngle': 89.99999999999999},
    'frydlant': {'focalLength': 2654.4955676930167, 'zenithAngle': 88.75820652204087}
}

azimuths = {
    'churanov': 45.0, 'ceske_budejovice': 270.0, 'belotin': 90.0, 'cheb': 45.0, 'brno': 45.0,
    'broumov': 45.0, 'dukovany': 135.0, 'dylen': 292.5, 'doksany': 225.0, 'frydlant': 225.0
}
