import os
import shutil
import json
import numpy as np
import cv2

from utils import read_image_greyscale


def is_blue_sky(image, sky_mask, hue_range=(90, 140), sat_range=(50, 255), val_range=(100, 255), blue_sky_threshold=0.8, blue_ground_threshold=0.2, edge_threshold=100):
    ground_mask = cv2.bitwise_not(sky_mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([hue_range[0], sat_range[0], val_range[0]])
    upper_blue = np.array([hue_range[1], sat_range[1], val_range[1]])

    blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_sky = blue & sky_mask
    blue_ground = blue & ground_mask
    blue_sky_percentage = np.sum(blue_sky > 0) / np.sum(sky_mask > 0)
    blue_ground_percentage = np.sum(blue_ground > 0) / np.sum(ground_mask > 0)

    masked_image_basic = cv2.bitwise_and(image, image, mask=sky_mask)
    masked_image = cv2.bitwise_and(image, image, mask=blue)
    gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_masked_image, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edge_percentage = np.sum(edges > 0) / edges.size

    # print(blue_sky_percentage, blue_ground_percentage, blue_sky_threshold, edge_percentage, edge_threshold)

    combined_image = np.concatenate(
        (image, masked_image_basic, masked_image, edges_colored), axis=1)
    is_blue_sky = blue_sky_percentage > blue_sky_threshold and edge_percentage < edge_threshold and blue_ground_percentage < blue_ground_threshold
    return is_blue_sky, combined_image


def fit_alpha_beta(img, mask):
    y_sky, x_sky = (mask == 255).nonzero()
    A = np.stack([(y_sky-np.min(y_sky))**2, np.ones(y_sky.shape)]).T
    b = img[y_sky, x_sky]
    x = np.linalg.inv(A.T@A)@A.T@b
    error = np.linalg.norm(A@x-b)
    return x[0], x[1], error


def grad(img, mask):
    y_sky, x_sky = (mask == 255).nonzero()
    g = np.gradient(img)
    g0 = g[0][y_sky, x_sky]
    g1 = g[1][y_sky, x_sky]
    return np.sum(g0**2)/g0.size + np.sum(g1**2)/g1.size


days = ['20210331', '20220803', '20210614', '20210619', '20210811',
        '20210909', '20220111', '20220321', '20220322', '20220323', '20220324', '20230301', '20230302', '20230303']


def select_clearskies2_from_clearskies():
    """From folder clearskies copies selected images to folder clearskies2 ... on Linux server
    """
    PATH = '/home/krsickao/bakalarka/data/clearskies'
    times = ['0700', '0730', '0800', '0830', '0900', '0930', '1000', '1030', '1100', '1130',
             '1200', '1230', '1300', '1330', '1400', '1430', '1500', '1530', '1600', '1630', '1700', '1730', '1800', '1830', '1900']
    locations = os.listdir(PATH)
    blue_sky_images = []
    target_dir = '/home/krsickao/bakalarka/data/clearskies2'

    for loci, location in enumerate(locations):
        print(f'{loci}/{len(locations)} - {location}:')
        mask = read_image_greyscale(
            '/home/krsickao/bakalarka/data/mask/'+location+'.jpg')
        webcam_path = os.path.join(PATH, location)
        for dayi, day in enumerate(days):
            print(f'\t{dayi}/{len(days)} - {day}:')
            webcam_day_path = os.path.join(webcam_path, day)
            if not os.path.isdir(webcam_day_path):
                # print('ERROR', location, day, 'does not exist')
                continue
            for timei, time in enumerate(times):
                img_path = os.path.join(webcam_day_path, time+'.jpg')
                if not os.path.isfile(img_path):
                    # print("ERROR:", img_path, "neexistuje")
                    continue
                image = read_image_greyscale(img_path)
                is_blue, visualization = is_blue_sky(image, mask)
                if is_blue:
                    dir = os.path.join(target_dir, location, day)
                    if not os.path.isdir(dir):
                        os.makedirs(dir)
                    shutil.copy(img_path, os.path.join(dir, time+'.jpg'))

                    blue_sky_images.append(img_path)


def calculate_derived_numbers(path: str = '/projects/SkyGAN/webcams/chmi.cz/sky_webcams', results_path: str = '../data/numbers'):
    """Calculates numbers for selecting clear sky images

    Args:
        path (str, optional): Input. Defaults to '/projects/SkyGAN/webcams/chmi.cz/sky_webcams'.
        results_path (str, optional): Output. Defaults to '../data/numbers'.
    """
    times = ['0700', '0730', '0800', '0830', '0900', '0930', '1000', '1030', '1100', '1130',
             '1200', '1230', '1300', '1330', '1400', '1430', '1500', '1530', '1600', '1630', '1700', '1730', '1800', '1830', '1900']
    locations = os.listdir(path)
    data = {}
    for loci, location in enumerate(locations):
        print(f'{loci}/{len(locations)} - {location}:')
        mask = read_image_greyscale(
            '/home/krsickao/bakalarka/data/mask/'+location+'.jpg')
        webcam_path = os.path.join(path, location)
        data[location] = {}
        for dayi, day in enumerate(days):
            print(f'\t{dayi}/{len(days)} - {day}:')
            webcam_day_path = os.path.join(webcam_path, day)
            if not os.path.isdir(webcam_day_path):
                # print('ERROR', location, day, 'does not exist')
                continue
            data[location][day] = {}
            for timei, time in enumerate(times):
                img_path = os.path.join(webcam_day_path, time+'.jpg')
                if not os.path.isfile(img_path):
                    # print("ERROR:", img_path, "neexistuje")
                    continue
                img = read_image_greyscale(img_path)
                alpha, beta, error = fit_alpha_beta(img, mask)
                gradient = grad(img, mask)
                data[location][day][time] = {
                    'filename': img_path, 'alpha': alpha, 'beta': beta, 'error': error, 'gradient': gradient
                }

        with open(f'{results_path}/data-{location}.json', 'w') as f:
            json.dump(data[location], f)
    # with open(f'../data/data.json', 'w') as f:
    #    json.dump(data, f)


def dataset_I_filenames(path='../data/numbers'):
    result = {}
    for json_filename in os.listdir(path):
        # loading
        with open(os.path.join(path, json_filename), 'r') as f:
            data = json.load(f)
        filenames, alphas, betas, errors = [], [], [], []
        for day in data:
            for time in data[day]:
                filenames.append(data[day][time]['filename'])
                alphas.append(data[day][time]['alpha'])
                betas.append(data[day][time]['beta'])
                errors.append(data[day][time]['error'])
        filenames, alphas, betas, errors = map(
            np.array, (filenames, alphas, betas, errors))
        # filtering
        alpha_mask = alphas < 0.001
        filenames, alphas, betas, errors = filenames[alpha_mask], alphas[
            alpha_mask], betas[alpha_mask], errors[alpha_mask]
        k10 = alphas.size//10
        top10percent_idx = np.argpartition(errors, k10)[:k10]
        filenames, alphas, betas, errors = filenames[top10percent_idx], alphas[
            top10percent_idx], betas[top10percent_idx], errors[top10percent_idx]
        N = 40
        if filenames.size > N:
            topNidx = np.argpartition(errors, N)[:N]
            filenames = filenames[topNidx]
        result[json_filename.split('-')[1].split('.')[0]] = filenames.tolist()
    return result


def dataset_J_filenames(path):
    result = {}
    for location in os.listdir(path):
        with open(os.path.join(path, location), 'r') as f:
            data = json.load(f)
        Ns = np.zeros(len(data))
        days = np.array(list(data.keys()))
        for dayi, day in enumerate(data):
            for time in data[day]:
                if data[day][time]['alpha'] < 0:
                    Ns[dayi] += 1

        idx = np.argsort(Ns)
        sorted_days = days[idx]
        top4_days = sorted_days[-4:]
        top4_Ns = Ns[idx][-4:]
        print(top4_days, top4_Ns)

        grads = []
        filenames = []
        for day in top4_days:
            for time in data[day]:
                grads.append(data[day][time]['gradient'])
                filenames.append(data[day][time]['filename'])

        grads, filenames = np.array(grads), np.array(filenames)

        N = min(40, len(grads))
        top_imgs = filenames[np.argsort(grads)][:N]

        result[location.split('-')[1].split('.')[0]] = top_imgs.tolist()
    return result


def copy_dataset(filenames, target_folder):
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
    for location in filenames:
        location_folder = os.path.join(target_folder, location)
        if not os.path.isdir(location_folder):
            os.mkdir(location_folder)
        for filei, file in enumerate(filenames[location]):
            _, year, month, day, hour, minute = parse_filepath(file)
            day_folder = os.path.join(
                location_folder, f"{year:4d}{month:02d}{day:02d}")
            if not os.path.isdir(day_folder):
                os.makedirs(day_folder)
            shutil.copyfile(file, os.path.join(
                day_folder, f'{hour:02d}{minute:02d}.jpg'))


if __name__ == '__main__':
    print('SELECT CLEARSKIES2 FROM CLEARSKIES')
    select_clearskies2_from_clearskies()
    print('CALCULATE DERIVED NUMBERS')
    calculate_derived_numbers(path='../data/clearskies',
                              results_path='../data/clearskies-numbers')
    print('COPY DATASET')
    copy_dataset(dataset_I_filenames(path='../data/clearskies-numbers'),
                 target_folder='../data/clearskies-I')
