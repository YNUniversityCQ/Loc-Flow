
# 2d heatmap decoding

import cv2
import torch
import numpy as np
from tqdm import tqdm
from lib import loader, modules
from torch.utils.data import Dataset, DataLoader

def get_max_preds(batch_heatmaps):

    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
            + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx],[dy]])
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border))
            dr[border: -border, border: -border] = hm[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[i, j] = dr[border: -border, border: -border].copy()
            hm[i, j] *= origin_max / np.max(hm[i, j])
    return hm


def los(output, target):
    loss = torch.sum((output - target) ** 2)
    loss = torch.sqrt(loss)
    return loss

def mean_loc_error(outputs, targets):
    def loc_error(coord1, coord2):

        return np.linalg.norm(coord1 - coord2)

    dis = [loc_error(coord1, coord2) for coord1, coord2 in zip(outputs, targets)]
    dis = np.mean(dis)
    return dis

# loading model
device = torch.device("cuda:0")
model = modules.LocUNet()
model.load_state_dict(torch.load('results/DPMBestModel.pt'))
model.to(device)

def main_worker():

    # loading test data
    test_data = loader.locDL(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)

    interation = 0
    loss = []
    for inputs, target_map, target_loc in tqdm(test_dataloader):
        interation += 1

        inputs = inputs.cuda()

        with torch.no_grad():
            pre = model(inputs)
            pre = pre.cpu().numpy()

            # pre = np.where(pre < 0, 0, pre)
            #
            # pre1 = pre.squeeze(0)
            # pre1 = pre1.squeeze(0)
            #
            # print(pre1)
            #
            # out_img = Image.fromarray((pre1 * 255).astype('uint8'))
            # out_img.show()

            coords, maxvals = get_max_preds(pre)

            hm = gaussian_blur(pre, 11)
            hm = np.maximum(hm, 1e-10)
            hm = np.log(hm)
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    coords[n, p] = taylor(hm[n][p], coords[n][p])

            pre = coords.copy()

            target = target_loc.cpu().numpy()
            # print(pre[0][:, ::-1] * 4, target)

            # back the init space
            d = mean_loc_error(pre[0][:, ::-1] * 4, target)
            # print(d)

            if interation >= 1000:
                break

        loss.append(d)

    d_err = sum(loss)/len(loss)

    print('测试集平均欧氏距离：', d_err)

if __name__ == '__main__':
 main_worker()


# # 1D heatmap regression
#
# import torch
# import numpy as np
# from tqdm import tqdm
# from lib import loader, modules
# from torch.utils.data import Dataset, DataLoader
#
# def los(output, target):
#     loss = np.sum((output - target) ** 2)
#     loss = np.sqrt(loss)
#     return loss
#
# def mean_loc_error(outputs, targets):
#     def loc_error(coord1, coord2):
#
#         return np.linalg.norm(coord1 - coord2)
#
#     dis = [loc_error(coord1, coord2) for coord1, coord2 in zip(outputs, targets)]
#     dis = np.mean(dis)
#     return dis
#
# # loading model
# device = torch.device("cuda:0")
# model = modules.LocUNet()
# model.load_state_dict(torch.load('results/DPMBestModel.pt'))
# model.to(device)
#
# def main_worker():
#
#     # loading test data
#     test_data = loader.locDL(phase='test')
#     test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)
#
#     interation = 0
#     loss = []
#     for inputs, target_map, target_loc in tqdm(test_dataloader):
#         interation += 1
#
#         inputs = inputs.cuda()
#
#         with torch.no_grad():
#             pre = model(inputs)
#             pre = pre.squeeze(0)
#
#             # target output
#             loc_r = torch.argmax(pre[0, :])
#             loc_c = torch.argmax(pre[1, :])
#
#             pre = torch.stack([loc_r, loc_c], dim=0)
#
#             pre = pre.cpu().numpy()
#             target_loc = target_loc.cpu().numpy()
#
#             d = los(pre / 2, target_loc)
#
#             # print(pre / 2, target_loc, d)
#
#             # if interation >= 1000:
#             #     break
#
#         loss.append(d)
#
#     d_err = sum(loss)/len(loss)
#
#     print('测试集平均欧氏距离：', d_err)
#
# if __name__ == '__main__':
#  main_worker()