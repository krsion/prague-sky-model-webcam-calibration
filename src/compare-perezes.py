from sky_models import PerezSkyModel, PerezAzimuthIndependentSkyModel, PragueSkyModel
import numpy as np
from PIL import Image

phi_c, theta_c, f, phi_s, theta_s = None, 0.5, 600, None, None
im1 = PerezSkyModel(720, 540).generate_image(phi_c, theta_c, f, phi_s, theta_s)
im2 = PerezAzimuthIndependentSkyModel(720, 540).generate_image(theta_c, f)
im3 = PragueSkyModel(720, 540).generate_image(phi_c, theta_c, f, phi_s, theta_s) 

scale1 = 255 / np.max(im1)
scale2 = 255 / np.max(im2)
scale3 = 255 / np.max(im3)

im1 = Image.fromarray(np.uint8(np.clip(im1 * scale1, None, 255)))
im2 = Image.fromarray(np.uint8(np.clip(im2 * scale2, None, 255)))
im3 = Image.fromarray(np.uint8(np.clip(im3 * scale3, None, 255)))

im1.save('im1.png')
im2.save('im2.png')
im3.save('im3.png')