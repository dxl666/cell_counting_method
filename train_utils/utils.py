import os
import torch
from skimage import measure
import cv2
import numpy as np
import math
import shutil
from PIL import Image
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.segmentation import watershed
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

def remove_small_points(image, threshold_point):
    original_img = cv2.imread(image)
    img = original_img[:, :, 0] // 255 if len(original_img.shape) == 3 else original_img[:, :] // 255
    img_label, num = measure.label(img, connectivity=2, return_num=True)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等

    resMatrix = np.zeros(img_label.shape)  # 创建一个原图像大小的全零数组
    for i in range(0, len(props)):
        if props[i].area > threshold_point:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix += tmp  # 组合所有符合条件的连通域
    resMatrix[resMatrix > 0] = 1
    resMatrix *= 255
    result = np.zeros(original_img.shape)
    if len(original_img.shape) == 3:
        result[:, :, 0] = resMatrix
        result[:, :, 1] = resMatrix
        result[:, :, 2] = resMatrix
    else:
        result = resMatrix
    return result



def calculation_cell_counting_accuracy(original_img_path, predict_img_path):
    original_img = cv2.imread(original_img_path)

    # A549: 40  MCF7: 10
    predict_img = remove_small_points(predict_img_path, 10)
    # predict_img = cv2.imread(predict_img_path)
    original_img = original_img[:, :, 0] // 255
    predict_img = predict_img[:, :, 0] // 255 if len(predict_img.shape) == 3 else predict_img[:, :] / 255

    original_img_label, original_num = measure.label(original_img, connectivity=2, return_num=True)  # 输出二值图像中所有的连通域
    predict_img_label, predict_num = measure.label(predict_img, connectivity=2, return_num=True)
    original_props = measure.regionprops(original_img_label)
    predict_props = measure.regionprops(predict_img_label)
    tp = 0
    fp = 0
    original_flags = np.zeros(original_num)
    for i in range(len(predict_props)):
        center_y, center_x = predict_props[i].centroid
        center_x = int(center_x)
        center_y = int(center_y)
        flag = original_img_label[center_y, center_x]
        if flag >= 1 and original_flags[flag - 1] == 0:
            tp += 1
            original_flags[flag - 1] = 1
        else:
            fp += 1
    fn = original_num - tp
    counting_accuracy = ((tp + 0.01) / (tp + fp + fn + 0.01)) * 100
    if counting_accuracy < 70:
        print('{} accuracy: {:.2f}%'.format(predict_img_path, counting_accuracy))
    return counting_accuracy


def run_delete_mask(mask_path):
    mask_list = os.listdir(mask_path)
    for mask_name in mask_list:
        if 'enhance' in mask_name:
            msk_path = os.path.join(mask_path, mask_name)
            os.remove(msk_path)
            print(mask_name + ' is deleted')


# A549/MCF7数据集增强
def data_annotation_enhance(mask_path):
    original_mask = cv2.imread(mask_path)
    enhance_mask = np.zeros(original_mask.shape)
    original_mask = original_mask[:, :, 0] // 255
    cell_name = mask_path.split('/')[-1].split('_')[0]
    original_mask_label, original_num = measure.label(original_mask, connectivity=2, return_num=True)
    original_props = measure.regionprops(original_mask_label)
    areas = []
    for i in range(len(original_props)):
        areas.append(original_props[i].area)
    sort_areas = sorted(areas)
    threshold = sort_areas[original_num // 2]
    for i in range(len(original_props)):
        if original_props[i].area > threshold:
            if cell_name == 'A549':
                radius = 10
            else:
                radius = 4
        else:
            if cell_name == "A549":
                if original_props[i].area <= 50:
                    radius = 5
                else:
                    radius = 7
            else:
                if original_props[i].area <= 15:
                    radius = 2
                else:
                    radius = 3

        cv2.circle(enhance_mask, (int(original_props[i].centroid[1]), int(original_props[i].centroid[0])), radius,
                   (255, 255, 255), -1)
    enhance_mask_name = mask_path.split('/')[-1].split('.')[0] + '_enhance.png'
    enhance_mask_path = '/'.join(mask_path.split('/')[0:-2]) + '/enhanceMasks'
    if not os.path.exists(enhance_mask_path):
        os.mkdir(enhance_mask_path)
    enhance_mask_path = os.path.join(enhance_mask_path, enhance_mask_name)
    cv2.imwrite(enhance_mask_path, enhance_mask)
    print(enhance_mask_name + ' save to ' + enhance_mask_path)


# BM数据集增强
def data_annotation_enhance_BM(mask_path, num):
    original_mask = cv2.imread(mask_path)
    enhance_mask = np.zeros(original_mask.shape)
    original_mask = original_mask[:, :, 0] // 255
    original_mask_label, original_num = measure.label(original_mask, connectivity=2, return_num=True)
    original_props = measure.regionprops(original_mask_label)
    for i in range(len(original_props)):
        cv2.circle(enhance_mask, (int(original_props[i].centroid[1]), int(original_props[i].centroid[0])), 5,
                   (255, 255, 255), -1)
    enhance_mask_name = "BM_" + str(num) + "_enhance.png"
    enhance_mask_path = '/'.join(mask_path.split('/')[0:-1]) + '/enhanceMasks'
    if not os.path.exists(enhance_mask_path):
        os.mkdir(enhance_mask_path)
    enhance_mask_path = os.path.join(enhance_mask_path, enhance_mask_name)
    cv2.imwrite(enhance_mask_path, enhance_mask)
    print(enhance_mask_name + ' save to ' + enhance_mask_path)


# A549/MCF7增强
def run_data_annotation_enhance():
    mask_root_path = '../DRIVE224_3/'
    flags = ["training", "val", "test"]
    for name in flags:
        mask_path = os.path.join(mask_root_path, name, "masks")
        image_list = os.listdir(mask_path)
        for mask_name in image_list:
            cell = mask_name.split('_')[0]
            if cell == "MCF7" or cell == "A549":
                msk_path = os.path.join(mask_path, mask_name)
                data_annotation_enhance(msk_path)
            else:
                continue


# BM增强
def run_data_annotation_enhance_BM():
    mask_root_path = '../DRIVE2/'
    mask_path = os.path.join(mask_root_path, "masks")
    image_list = os.listdir(mask_path)
    num = 1
    for mask_name in image_list:
        msk_path = os.path.join(mask_path, mask_name)
        data_annotation_enhance_BM(msk_path, num)
        num += 1


def copy_label_files(source_dir, target_dir):
    # 遍历源目录下的所有一级子目录
    for subdir in os.listdir(source_dir):
        sub_path = os.path.join(source_dir, subdir)
        if os.path.isdir(sub_path):
            # 在子目录下查找 label.png 文件
            label_file = os.path.join(sub_path, 'label.png')
            if os.path.isfile(label_file):
                # 将 label.png 复制到目标位置，并以子目录名字命名
                image_name = '_'.join(subdir.split('_')[0:-1])
                target_filename = os.path.join(target_dir, image_name + '.png')
                shutil.copy(label_file, target_filename)
                print(f"复制 {label_file} 到 {target_filename}")


def convertAnn(mask_path):
    origin_mask = cv2.imread(mask_path)
    mask = np.zeros(origin_mask.shape, dtype="uint8")
    for i in range(origin_mask.shape[0]):
        for j in range(origin_mask.shape[1]):
            if origin_mask[i, j, 0] > 0 or origin_mask[i, j, 1] > 0 or origin_mask[i, j, 2] > 0:
                mask[i, j, :] = 255
    new_path = os.path.join('/'.join(mask_path.split('/')[0:-1]), "label_convert.png")
    cv2.imwrite(new_path, mask)


def compute_mean_std(root_path):
    img_channels = 3
    img_dir = root_path
    # roi_dir = "./DRIVE/training/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    # assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".png")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        # ori_path = os.path.join(roi_dir, img_name.replace(".tif", "_mask.gif"))
        img = np.array(Image.open(img_path)) / 255.
        # roi_img = np.array(Image.open(ori_path).convert('L'))
        img = img.reshape(-1, 3)
        # img = img[roi_img == 255]
        # test = img.mean(axis=0)
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    mean = np.round(mean, 3)
    std = np.round(std, 3)
    print(f"mean: {mean}")
    print(f"std: {std}")



# 针对MBM
def drawResultFigureMBM(image_path, mask_path, enhance_path, model):
    image = cv2.imread(image_path)
    true_mask = cv2.imread(mask_path)
    # origin_predict = remove_small_points(origin_path, 40)
    enhance_predict = cv2.imread(enhance_path)
    true_mask = true_mask[:, :, 0] // 255 if len(true_mask.shape) == 3 else true_mask // 255
    enhance_predict = enhance_predict[:, :, 0] // 255 if len(enhance_predict.shape) == 3 else enhance_predict // 255
    true_mask_label, true_num = measure.label(true_mask, connectivity=2, return_num=True)
    enhance_predict_label, enhance_num = measure.label(enhance_predict, connectivity=2, return_num=True)
    true_props = measure.regionprops(true_mask_label)
    enhance_props = measure.regionprops(enhance_predict_label)
    flag_arr = np.zeros(true_num)
    enhance_image = image.copy()
    enhance_tp = 0
    for i in range(len(enhance_props)):
        center_y, center_x = enhance_props[i].centroid
        center_y = int(center_y)
        center_x = int(center_x)
        flag = true_mask_label[center_y, center_x]
        if flag != 0 and flag_arr[flag - 1] != 1:
            flag_arr[flag - 1] = 1
            enhance_tp += 1
            true_y, true_x = true_props[flag - 1].centroid
            true_y = int(true_y)
            true_x = int(true_x)
            cv2.circle(enhance_image, (true_x, true_y), 3, (0, 255, 0), -1)

    enhance_fp = enhance_num - enhance_tp
    enhance_fn = true_num - enhance_tp

    for i in range(len(flag_arr)):
        if flag_arr[i] == 0:
            true_y, true_x = true_props[i].centroid
            true_y = int(true_y)
            true_x = int(true_x)
            cv2.circle(enhance_image, (true_x, true_y), 3, (0, 0, 255), -1)
    save_root_path = "../predict/"
    save_name = enhance_path.split("/")[-1]
    enhance_save_path = os.path.join(save_root_path, "enhance_MBM", model + "_counting")
    if not os.path.exists(enhance_save_path):
        os.mkdir(enhance_save_path)
    enhance_save_path = os.path.join(enhance_save_path, save_name)
    cv2.imwrite(enhance_save_path, enhance_image)
    print("enhance predict image save to " + enhance_save_path)


def run_drawResultFigureMBM():
    image_root_path = "../DRIVE224_4/test/images"
    mask_root_path = "../DRIVE224_4/test/enhanceMasks"
    enhance_root_path = "../predict/enhance_MBM/"
    model_list = ['UNet', 'SAUNet', 'CellUNet', 'CResUNet', 'SCTransUNet']
    image_list = os.listdir(image_root_path)

    for image in image_list:
        cell = image.split('_')[0]
        if cell == "A549" or cell == "MCF7" or cell == 'BMB':
            image_path = os.path.join(image_root_path, image)
            mask = image.split('.')[0] + "_mask_enhance.png"
            mask_path = os.path.join(mask_root_path, mask)
            predict = image.split('.')[0] + '_mask_predict.png'
            for model in model_list:
                enhance_path = os.path.join(enhance_root_path, model)
                enhance_path = os.path.join(enhance_path, predict)
                drawResultFigureMBM(image_path, mask_path, enhance_path, model)


# 针对MBM
def drawGTMBM(image_path, mask_path):
    image = cv2.imread(image_path)
    true_mask = cv2.imread(mask_path)
    # origin_predict = remove_small_points(origin_path, 40)

    true_mask = true_mask[:, :, 0] // 255 if len(true_mask.shape) == 3 else true_mask // 255
    true_mask_label, true_num = measure.label(true_mask, connectivity=2, return_num=True)
    true_props = measure.regionprops(true_mask_label)
    true_image = image.copy()
    enhance_tp = 0
    for i in range(len(true_props)):
        center_y, center_x = true_props[i].centroid
        center_y = int(center_y)
        center_x = int(center_x)
        cv2.circle(true_image, (center_x, center_y), 3, (0, 255, 0), -1)
    save_root_path = "../predict/"
    save_name = image_path.split("/")[-1]
    true_save_path = os.path.join(save_root_path, "enhance_liveCell", 'GT')
    if not os.path.exists(true_save_path):
        os.mkdir(true_save_path)
    true_save_path = os.path.join(true_save_path, save_name)
    cv2.imwrite(true_save_path, true_image)
    print("enhance predict image save to " + true_save_path)


def run_drawGTMBM():
    image_root_path = "../DRIVE224_2/test/images"
    mask_root_path = "../DRIVE224_2/test/enhanceMasks"
    image_list = os.listdir(image_root_path)

    for image in image_list:
        cell = image.split('_')[0]
        if cell == "A549" or cell == "MCF7" or cell == 'BMB':
            image_path = os.path.join(image_root_path, image)
            mask = image.split('.')[0] + "_mask_enhance.png"
            mask_path = os.path.join(mask_root_path, mask)
            drawGTMBM(image_path, mask_path)


def convertDot(mask_path):
    original_mask = cv2.imread(mask_path)
    enhance_mask = np.zeros(original_mask.shape)
    original_mask = original_mask[:, :, 0] // 255
    original_mask_label, original_num = measure.label(original_mask, connectivity=2, return_num=True)
    original_props = measure.regionprops(original_mask_label)
    for i in range(len(original_props)):
        # cv2.circle(enhance_mask, (int(original_props[i].centroid[1]), int(original_props[i].centroid[0])), 5,
        #            (255, 255, 255), -1)
        center_x = int(original_props[i].centroid[1])
        center_y = int(original_props[i].centroid[0])
        enhance_mask[center_y, center_x, :] = 255
    enhance_mask_name = '_'.join(os.path.splitext(mask_path.split("/")[-1])[0].split('_')[0:-1]) + '_dot.png'
    enhance_mask_path = '../predict/test'
    if not os.path.exists(enhance_mask_path):
        os.mkdir(enhance_mask_path)
    enhance_mask_path = os.path.join(enhance_mask_path, enhance_mask_name)
    cv2.imwrite(enhance_mask_path, enhance_mask)
    print(enhance_mask_name + ' save to ' + enhance_mask_path)


if __name__ == "__main__":
    run_drawGTMBM()