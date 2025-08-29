from __future__ import print_function, division

import os
import re
import math
import torch
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

warnings.filterwarnings("ignore")

def rss_noise_add(image, val):
    dB_image = -127 + (image / 255) * 80
    noise = np.random.normal(0, val, image.shape)
    dB_image += noise
    image = ((dB_image + 127) / 80) * 255
    return image

class locDL(Dataset):
    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="dataset/",
                 numTx=2,
                 numTrials=50,
                 numRx=200,
                 simulation="DPM",
                 cityMap="true",
                 carsMap="false",
                 TxMaps="true",
                 noise="false",
                 val=0,
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom".
            dir_dataset: directory of the dataset.
            numTx: Number of transmitters per map. Default and maximum numTx = 5.
            numTrials: Number of sets of numTx transmitters per map
            simulation:"DPM", "IRT2", "DPMtoIRT2", "DPMcars". Default= "DPM"
            cityMap: . Default cityMap="false"
            TxMaps: Images of Tx. Defaul TxMaps="false"
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())

        Output:
            inputs: The LocUNet inputs.
            RXlocr: Pixel row of true location
            RXlocc: Pixel column of true location

        """

        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 99, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 68
        elif phase == "val":
            self.ind1 = 69
            self.ind2 = 83
        elif phase == "test":
            self.ind1 = 84
            self.ind2 = 98
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.numTrials = numTrials
        self.numRx = numRx
        self.simulation = simulation
        self.cityMap = cityMap
        self.carsMap = carsMap
        self.TxMaps = TxMaps
        self.transform = transform
        self.noise = noise
        self.val = val

        self.height = 256
        self.width = 256

        if simulation == "DPM":
            self.dir_gainTrue = self.dir_dataset + "DPM/true/"
            self.dir_gainEst = self.dir_dataset + "DPM/estimate/"
        elif simulation == "DPMcars":
            self.dir_gainTrue = self.dir_dataset + "DPMcars/true/"
            self.dir_gainEst = self.dir_dataset + "DPMcars/estimate/"
        elif simulation == "IRT2":
            self.dir_gainTrue = self.dir_dataset + "IRT2/true/"
            self.dir_gainEst = self.dir_dataset + "DPM/estimate/"
        elif simulation == "IRT2cars":
            self.dir_gainTrue = self.dir_dataset + "IRT2cars/true/"
            self.dir_gainEst = self.dir_dataset + "DPMcars/estimate/"


        self.dir_buildings = self.dir_dataset + "buildings/"
        self.dir_cars = self.dir_dataset + "cars/"
        self.dir_Tx = self.dir_dataset + "png/antennas/"

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTrials * self.numRx

    def __getitem__(self, idx):
        numMapPhase = self.ind2 - self.ind1 + 1
        idxMap, idxTrial, idxRx = np.unravel_index(idx, (numMapPhase, self.numTrials, self.numRx))
        dataset_map_ind = self.maps_inds[idxMap + self.ind1]
        # names of files that depend only on the map:
        nameMap = str(dataset_map_ind) + ".png"

        # Load true (reported) radio maps for Txs:
        mat = np.load(r'C:\Users\陈琦\Desktop\python\LocUNet-main\lib\my_fileCorr.npy', allow_pickle='TRUE').item()
        rxx = mat['rxx']
        rxy = mat['rxy']
        RXr = rxx[dataset_map_ind, idxRx] - 1
        RXc = rxy[dataset_map_ind, idxRx] - 1
        antList = mat['antList']
        TXlist = antList[idxTrial, dataset_map_ind, :]

        inputEstMaps = []
        for m in range(self.numTx):
            name2 = str(dataset_map_ind) + "_" + str(TXlist[m] - 1) + ".png"
            img_name_gainTrue = os.path.join(self.dir_gainTrue, name2)
            image_gainTrue = np.asarray(io.imread(img_name_gainTrue)) / 255
            img_name_gainEst = os.path.join(self.dir_gainEst, name2)
            image_gainEst = np.asarray(io.imread(img_name_gainEst))

            if self.noise == 'true':
                image_gainEst = rss_noise_add(image_gainEst, self.val) / 255
            else:
                image_gainEst = image_gainEst / 255

            inputEstMaps.append(image_gainEst)
            gainTrue = image_gainTrue[RXr, RXc]
            imgGainTrue = gainTrue * np.ones(np.shape(image_gainEst))
            inputEstMaps.append(imgGainTrue)
        inputs = inputEstMaps

        # Load Tx maps
        if self.TxMaps == "true":
            antX = mat['antX']
            antY = mat['antY']
            TXr = antX[idxTrial, :, dataset_map_ind]
            TXc = antY[idxTrial, :, dataset_map_ind]
            inputTxMaps = []
            for m in range(self.numTx):
                imTx = np.zeros((256, 256))

                imTx[TXr[m], TXc[m]] = 1
                # add noise of BS
                # sigma = 15
                # imTx[TXr[m] + int(np.random.normal(0, sigma, 1)), TXc[m] + int(np.random.normal(0, sigma, 1))] = 1

                inputs.append(imTx)

        # Load buildings:
        if self.cityMap == "true":
            img_name_buildings = os.path.join(self.dir_buildings, nameMap)
            image_buildings = np.asarray(io.imread(img_name_buildings)) / 255
            inputs.append(image_buildings)

        # Load cars:
        if self.carsMap == "true":
            img_name_cars = os.path.join(self.dir_cars, nameMap)
            image_cars = np.asarray(io.imread(img_name_cars)) / 255
            inputs.append(image_cars)
        inputs = np.asarray(inputs, dtype=np.float32)
        inputs = np.transpose(inputs, (1, 2, 0))

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)

        # True coordinates
        # RXlocr = torch.from_numpy(np.asarray(RXr/255, dtype=np.float32))
        # RXlocc = torch.from_numpy(np.asarray(RXc/255, dtype=np.float32))

        RXlocr = torch.from_numpy(np.asarray(RXr / 255, dtype=np.float32))
        RXlocc = torch.from_numpy(np.asarray(RXc / 255, dtype=np.float32))

        # target coordinate add noise
        # sigma = 5
        # RXlocr = torch.from_numpy(np.asarray(int(RXr + np.random.normal(0, sigma, 1)) / 255, dtype=np.float32))
        # RXlocc = torch.from_numpy(np.asarray(int(RXc + np.random.normal(0, sigma, 1)) / 255, dtype=np.float32))

        return inputs, torch.stack((RXlocr, RXlocc), dim=0)


class locDL2(Dataset):
    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="dataset/",
                 numTx=5,
                 numTrials=50,
                 numRx=200,
                 simulation="DPM",
                 cityMap="true",
                 carsMap="false",
                 TxMaps="true",
                 mean=0,
                 std=5,
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom".
            dir_dataset: directory of the dataset.
            numTx: Number of transmitters per map. Default and maximum numTx = 5.
            numTrials: Number of sets of numTx transmitters per map
            simulation:"DPM", "IRT2", "DPMtoIRT2", "DPMcars". Default= "DPM"
            cityMap: . Default cityMap="false"
            TxMaps: Images of Tx. Defaul TxMaps="false"
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())

        Output:
            inputs: The LocUNet inputs.
            RXlocr: Pixel row of true location
            RXlocc: Pixel column of true location

        """

        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 99, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 68
        elif phase == "val":
            self.ind1 = 69
            self.ind2 = 83
        elif phase == "test":
            self.ind1 = 84
            self.ind2 = 98
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.numTrials = numTrials
        self.numRx = numRx
        self.simulation = simulation
        self.cityMap = cityMap
        self.carsMap = carsMap
        self.TxMaps = TxMaps
        self.transform = transform
        self.mean = mean
        self.std = std

        self.height = 256
        self.width = 256

        if simulation == "DPM":
            self.dir_gainTrue = self.dir_dataset + "DPM/true/"
            self.dir_gainEst = self.dir_dataset + "DPM/estimate/"
        elif simulation == "DPMcars":
            self.dir_gainTrue = self.dir_dataset + "DPMcars/true/"
            self.dir_gainEst = self.dir_dataset + "DPMcars/estimate/"
        elif simulation == "IRT2":
            self.dir_gainTrue = self.dir_dataset + "IRT2/true/"
            self.dir_gainEst = self.dir_dataset + "DPM/estimate/"
        elif simulation == "IRT2cars":
            self.dir_gainTrue = self.dir_dataset + "IRT2cars/true/"
            self.dir_gainEst = self.dir_dataset + "DPM/estimate/"


        self.dir_buildings = self.dir_dataset + "buildings/"
        self.dir_cars = self.dir_dataset + "cars/"
        self.dir_Tx = self.dir_dataset + "png/antennas/"

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTrials * self.numRx

    def __getitem__(self, idx):
        numMapPhase = self.ind2 - self.ind1 + 1
        idxMap, idxTrial, idxRx = np.unravel_index(idx, (numMapPhase, self.numTrials, self.numRx))
        dataset_map_ind = self.maps_inds[idxMap + self.ind1]
        # names of files that depend only on the map:
        nameMap = str(dataset_map_ind) + ".png"

        # Load true (reported) radio maps for Txs:
        mat = np.load(r'C:\Users\陈琦\Desktop\python\LocUNet-main\lib\my_fileCorr.npy', allow_pickle='TRUE').item()
        rxx = mat['rxx']
        rxy = mat['rxy']
        RXr = rxx[dataset_map_ind, idxRx] - 1
        RXc = rxy[dataset_map_ind, idxRx] - 1
        antList = mat['antList']
        TXlist = antList[idxTrial, dataset_map_ind, :]

        inputEstMaps = []

        img_name_buildings = os.path.join(self.dir_buildings, nameMap)
        image_buildings = np.asarray(io.imread(img_name_buildings)) / 255

        for m in range(self.numTx):
            name2 = str(dataset_map_ind) + "_" + str(TXlist[m] - 1) + ".png"
            img_name_gainTrue = os.path.join(self.dir_gainTrue, name2)
            image_gainTrue = np.asarray(io.imread(img_name_gainTrue)) / 255
            img_name_gainEst = os.path.join(self.dir_gainEst, name2)
            image_gainEst = np.asarray(io.imread(img_name_gainEst)) / 255
            inputEstMaps.append(image_gainEst)
            RXrN = min(max(0, RXr + np.random.normal(self.mean, self.std)), 255)
            RXcN = min(max(0, RXc + np.random.normal(self.mean, self.std)), 255)
            RxrNround = np.rint(RXrN)
            RxcNround = np.rint(RXcN)
            while image_buildings[int(RxrNround), int(RxcNround)] == 1:
                RXrN = min(max(0, RXr + np.random.normal(self.mean, self.std)), 255)
                RXcN = min(max(0, RXc + np.random.normal(self.mean, self.std)), 255)
                RxrNround = np.rint(RXrN)
                RxcNround = np.rint(RXcN)
            gainTrue = image_gainTrue[int(RxrNround), int(RxcNround)]
            imgGainTrue = gainTrue * np.ones(np.shape(image_gainEst))
            inputEstMaps.append(imgGainTrue)

        inputs = inputEstMaps

        # Load Tx maps
        if self.TxMaps == "true":
            antX = mat['antX']
            antY = mat['antY']
            TXr = antX[idxTrial, :, dataset_map_ind]
            TXc = antY[idxTrial, :, dataset_map_ind]
            inputTxMaps = []
            for m in range(self.numTx):
                imTx = np.zeros((256, 256))

                imTx[TXr[m], TXc[m]] = 1
                # sigma = 15
                # imTx[TXr[m] + int(np.random.normal(0, sigma, 1)), TXc[m] + int(np.random.normal(0, sigma, 1))] = 1

                inputs.append(imTx)

        # Load buildings:
        if self.cityMap == "true":
            img_name_buildings = os.path.join(self.dir_buildings, nameMap)
            image_buildings = np.asarray(io.imread(img_name_buildings)) / 255
            inputs.append(image_buildings)

        # Load cars:
        if self.carsMap == "true":
            img_name_cars = os.path.join(self.dir_cars, nameMap)
            image_cars = np.asarray(io.imread(img_name_cars)) / 255
            inputs.append(image_cars)
        inputs = np.asarray(inputs, dtype=np.float32)
        inputs = np.transpose(inputs, (1, 2, 0))

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)

        # True coordinates
        RXlocr = torch.from_numpy(np.asarray(RXr, dtype=np.float32))
        RXlocc = torch.from_numpy(np.asarray(RXc, dtype=np.float32))

        return inputs, torch.stack((RXlocr, RXlocc), dim=0)


# class locDL(Dataset):
#     def __init__(self, maps_inds=np.zeros(1), phase="train",
#                  ind1=0, ind2=0,
#                  dir_dataset="dataset/",
#                  numTx=5,
#                  num_user=1,
#                  numTrials=50,
#                  numRx=200,
#                  width=256,
#                  height=256,
#                  simulation="DPM_car",  # DPM_car, DPM
#                  cityMap="true",
#                  carsMap="true",
#                  TxMaps="true",
#                  transform=transforms.ToTensor()):
#
#         if maps_inds.size == 1:

#             self.maps_inds = np.arange(0, 99, 1, dtype=np.int16)

#             np.random.seed(42)
#             np.random.shuffle(self.maps_inds)
#         else:
#             self.maps_inds = maps_inds
#
#         if phase == "train":
#             self.ind1 = 0
#             self.ind2 = 68
#         elif phase == "val":
#             self.ind1 = 69
#             self.ind2 = 83
#         elif phase == "test":
#             self.ind1 = 84
#             self.ind2 = 98
#
#         else:
#             self.ind1 = ind1
#             self.ind2 = ind2
#
#         self.dir_dataset = dir_dataset
#         self.numTx = numTx
#         self.num_user = num_user
#         self.numTrials = numTrials
#         self.numRx = numRx
#         self.simulation = simulation
#         self.cityMap = cityMap
#         self.carsMap = carsMap
#         self.TxMaps = TxMaps
#         self.transform = transform
#
#         self.height = width
#         self.width = height
#
#         if simulation == "DPM":
#             self.dir_gainTrue = self.dir_dataset + "DPM/true/"
#             self.dir_gainEst = self.dir_dataset + "DPM/estimate/"
#
#         elif simulation == "DPM_car":
#             self.dir_gainTrue = self.dir_dataset + "DPMcars/true/"
#             self.dir_gainEst = self.dir_dataset + "DPMcars/estimate/"
#
#         self.dir_buildings = self.dir_dataset + "buildings/"
#         self.dir_cars = self.dir_dataset + "cars/"
#         self.dir_Tx = self.dir_dataset + "png/antennas/"
#
#     def __len__(self):
#
#         return (self.ind2 - self.ind1 + 1) * self.numTrials * self.numRx
#
#     def __getitem__(self, idx):
#
#         numMapPhase = self.ind2 - self.ind1 + 1
#         idxMap, idxTrial, idxRx = np.unravel_index(idx, (numMapPhase, self.numTrials, self.numRx))
#         dataset_map_ind = self.maps_inds[idxMap + self.ind1]
#         nameMap = str(dataset_map_ind) + ".png"
#

#         mat = np.load(r'C:\Users\陈琦\Desktop\python\LocUNet-main\lib\my_fileCorr.npy', allow_pickle='TRUE').item()
#
#         rxx = mat['rxx']  # 99*200
#         # print(rxx[62, :])
#         rxy = mat['rxy']  # 99*200
#         # print(rxy[62, :])
#
#         RXr = rxx[dataset_map_ind, idxRx] - 1
#         RXc = rxy[dataset_map_ind, idxRx] - 1
#
#         # shape: (50, 99, 5)
#         antList = mat['antList']
#         TXlist = antList[idxTrial, dataset_map_ind, :]
#
#         inputEstMaps = []

#         for m in range(self.numTx):

#             name2 = str(dataset_map_ind) + "_" + str(TXlist[m] - 1) + ".png"
#             img_name_gainTrue = os.path.join(self.dir_gainTrue, name2)
#             image_gainTrue = np.asarray(io.imread(img_name_gainTrue)) / 255
#

#             img_name_gainEst = os.path.join(self.dir_gainEst, name2)
#             image_gainEst = np.asarray(io.imread(img_name_gainEst)) / 255
#
#             inputEstMaps.append(image_gainEst)
#

#             gainTrue = image_gainTrue[RXr, RXc]
#
#             imgGainTrue = gainTrue * np.ones(np.shape(image_gainEst))
#             inputEstMaps.append(imgGainTrue)
#
#         inputs = inputEstMaps
#

#         if self.TxMaps == "true":
#             antX = mat['antX']
#             antY = mat['antY']
#             TXr = antX[idxTrial, :, dataset_map_ind]
#             # print(TXr)
#             TXc = antY[idxTrial, :, dataset_map_ind]
#             # print(TXc)
#             for m in range(self.numTx):
#                 imTx = np.zeros((256, 256))
#                 imTx[TXr[m], TXc[m]] = 1
#                 inputs.append(imTx)
#

#         # print(len(inputs))
#

#         if self.cityMap == "true":
#             img_name_buildings = os.path.join(self.dir_buildings, nameMap)
#             image_buildings = np.asarray(io.imread(img_name_buildings)) / 255
#             inputs.append(image_buildings)
#
#         inputs = np.asarray(inputs, dtype=np.float32)
#         inputs = np.transpose(inputs, (1, 2, 0))

#         if self.transform:

#             inputs = self.transform(inputs).type(torch.float32)
#

#
#         RXlocr = torch.from_numpy(np.asarray(RXr/255, dtype=np.float32))
#         RXlocc = torch.from_numpy(np.asarray(RXc/255, dtype=np.float32))
#         loc = torch.stack((RXlocr, RXlocc), dim=0)
#
#         return inputs, loc

# RSS add noise
def rss_noise_add(image, val):
    dB_image = -127 + (image / 255) * 80
    noise = np.random.normal(0, val, image.shape)
    dB_image += noise
    image = ((dB_image + 127) / 80) * 255
    return image

class locDL5(Dataset):
    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="dataset/",
                 numTx=5,
                 num_user=1,
                 numTrials=50,
                 numRx=200,
                 width=256,
                 height=256,
                 simulation="DPM",
                 cityMap="true",
                 carsMap="false",
                 TxMaps="true",
                 noise="true",
                 noise_val=5,
                 transform=transforms.ToTensor()):

        if maps_inds.size==1:

            self.maps_inds=np.arange(0, 99, 1, dtype=np.int16)

            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 68
        elif phase == "val":
            self.ind1 = 69
            self.ind2 = 83
        elif phase == "test":
            self.ind1 = 84
            self.ind2 = 98

        else:
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.num_user = num_user
        self.numTrials = numTrials
        self.numRx = numRx
        self.simulation = simulation
        self.cityMap = cityMap
        self.carsMap = carsMap
        self.TxMaps = TxMaps
        self.transform = transform
        self.noise = noise
        self.val = noise_val

        self.height = width
        self.width = height

        if simulation == "DPM":
            self.dir_gainTrue = self.dir_dataset + "DPM/true/"
            self.dir_gainEst = self.dir_dataset + "DPM/estimate/"

        self.dir_buildings = self.dir_dataset + "buildings/"
        self.dir_Tx = self.dir_dataset + "png/antennas/"

    def __len__(self):

        return (self.ind2-self.ind1+1)*self.numTrials*self.numRx

    def __getitem__(self, idx):

        numMapPhase = self.ind2-self.ind1+1
        idxMap,idxTrial,idxRx = np.unravel_index(idx,(numMapPhase, self.numTrials, self.numRx))
        dataset_map_ind = self.maps_inds[idxMap+self.ind1]

        nameMap = str(dataset_map_ind) + ".png"

        mat = np.load(r'C:\Users\陈琦\Desktop\python\LocUNet-main\lib\my_fileCorr.npy', allow_pickle='TRUE').item()

        rxx = mat['rxx']                  # 99*200
        rxy = mat['rxy']                  # 99*200
        # print(rxy[dataset_map_ind, :])

        # get the ue localization (single or multi)
        multi_row = []
        multi_column = []
        if self.num_user > 1:
            np.random.seed(idxRx)
            indices = np.random.choice(rxx.shape[1], self.num_user, replace=False)
            multi_row = rxx[dataset_map_ind, indices] - 1
            multi_column = rxy[dataset_map_ind, indices] - 1
        else:
            multi_row.append(rxx[dataset_map_ind, idxRx] - 1)
            multi_column.append(rxy[dataset_map_ind, idxRx] - 1)

        antList = mat['antList']
        TXlist = antList[idxTrial, dataset_map_ind, :]

        inputEstMaps = []
        for m in range(self.numTx):

            name2 = str(dataset_map_ind) + "_" + str(TXlist[m]-1) + ".png"
            img_name_gainTrue = os.path.join(self.dir_gainTrue, name2)
            image_gainTrue = np.asarray(io.imread(img_name_gainTrue))/255


            img_name_gainEst = os.path.join(self.dir_gainEst, name2)
            image_gainEst = np.asarray(io.imread(img_name_gainEst))
            if self.noise == 'true':
                image_gainEst = rss_noise_add(image_gainEst, self.val)/255
            else:
                image_gainEst = image_gainEst / 255

            inputEstMaps.append(image_gainEst)


            for r, c in zip(multi_row, multi_column):
                gainTrue = image_gainTrue[r, c]
                imgGainTrue = gainTrue*np.ones(np.shape(image_gainEst))
                inputEstMaps.append(imgGainTrue)

        inputs = inputEstMaps


        if self.TxMaps == "true":
            antX = mat['antX']
            antY = mat['antY']
            TXr = antX[idxTrial, :, dataset_map_ind]
            TXc = antY[idxTrial, :, dataset_map_ind]
            for m in range(self.numTx):
                imTx = np.zeros((256, 256))
                imTx[TXr[m], TXc[m]] = 1
                inputs.append(imTx)


        if self.cityMap == "true":
            img_name_buildings = os.path.join(self.dir_buildings, nameMap)
            image_buildings = np.asarray(io.imread(img_name_buildings))/255
            inputs.append(image_buildings)

        inputs = np.asarray(inputs, dtype=np.float32)
        inputs = np.transpose(inputs, (1, 2, 0))

        if self.transform:

            inputs = self.transform(inputs).type(torch.float32)


        Loc = []
        for rr, cc in zip(multi_row, multi_column):
            RXlocr = torch.from_numpy(np.asarray(rr, dtype=np.float32).reshape(-1, 1))
            RXlocc = torch.from_numpy(np.asarray(cc, dtype=np.float32).reshape(-1, 1))
            loc = torch.cat((RXlocr, RXlocc), dim=1)
            Loc.append(loc)

        Loc = torch.cat(Loc, 0)

        return inputs, Loc


# RSS 2d heatmap

# define gaussian heatmap
def GaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map

class locDL3(Dataset):
    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="dataset/",
                 numTx=5,
                 numTrials=50,
                 numRx=200,
                 width=256,
                 height=256,
                 simulation="DPM",
                 cityMap="true",
                 carsMap="false",
                 TxMaps="true",
                 transform=transforms.ToTensor()):


        if maps_inds.size == 1:

            self.maps_inds = np.arange(0, 99, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 68
        elif phase == "val":
            self.ind1 = 69
            self.ind2 = 83
        elif phase == "test":
            self.ind1 = 84
            self.ind2 = 98

        else:
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.numTrials = numTrials
        self.numRx = numRx
        self.simulation = simulation
        self.cityMap = cityMap
        self.carsMap = carsMap
        self.TxMaps = TxMaps
        self.transform = transform

        self.height = width
        self.width = height

        if simulation == "DPM":
            self.dir_gainTrue = self.dir_dataset + "DPM/true/"
            self.dir_gainEst = self.dir_dataset + "DPM/estimate/"

        self.dir_buildings = self.dir_dataset + "buildings/"
        self.dir_Tx = self.dir_dataset + "png/antennas/"

    def __len__(self):

        return (self.ind2 - self.ind1 + 1) * self.numTrials * self.numRx

    def __getitem__(self, idx):


        numMapPhase = self.ind2 - self.ind1 + 1

        idxMap, idxTrial, idxRx = np.unravel_index(idx, (numMapPhase, self.numTrials, self.numRx))
        # print('idx = ',idx,'idxmap=',idxMap,'idxtrial=',idxTrial,'idxRx=',idxRx)
        # print('self.ind1=',self.ind1)
        dataset_map_ind = self.maps_inds[idxMap + self.ind1]
        # print(self.maps_inds)

        # print('dataset_map_ind=',dataset_map_ind)
        nameMap = str(dataset_map_ind) + ".png"


        mat = np.load(r'C:\Users\陈琦\Desktop\python\LocUNet-main\lib\my_fileCorr.npy', allow_pickle='TRUE').item()

        rxx = mat['rxx']  # 99*200
        # print(rxx[62, :])
        rxy = mat['rxy']  # 99*200
        # print(rxy[62, :])

        RXr = rxx[dataset_map_ind, idxRx]
        # print(dataset_map_ind, idxRx)
        RXc = rxy[dataset_map_ind, idxRx]

        loc_x = np.round(RXr / 4)
        loc_y = np.round(RXc / 4)

        # print(loc_x, loc_y)

        antList = mat['antList']
        TXlist = antList[idxTrial, dataset_map_ind, :]

        # gaussian heatmap
        var = self.width / 64 * 2

        heatmap = GaussianHeatMap(64, 64, loc_y, loc_x, var)

        # visualization heatmap
        # out_img = Image.fromarray((heatmap * 255).astype('uint8'))
        # out_img.show()

        inputEstMaps = []

        for m in range(self.numTx):

            name2 = str(dataset_map_ind) + "_" + str(TXlist[m] - 1) + ".png"
            img_name_gainTrue = os.path.join(self.dir_gainTrue, name2)
            image_gainTrue = np.asarray(io.imread(img_name_gainTrue)) / 255


            img_name_gainEst = os.path.join(self.dir_gainEst, name2)
            image_gainEst = np.asarray(io.imread(img_name_gainEst)) / 255

            inputEstMaps.append(image_gainEst)


            gainTrue = image_gainTrue[RXr, RXc]

            imgGainTrue = gainTrue * np.ones(np.shape(image_gainEst))

            inputEstMaps.append(imgGainTrue)

        inputs = inputEstMaps

        # # 10维
        # print(len(inputs))


        if self.TxMaps == "true":
            antX = mat['antX']
            antY = mat['antY']
            TXr = antX[idxTrial, :, dataset_map_ind]
            # print(TXr)
            TXc = antY[idxTrial, :, dataset_map_ind]
            # print(TXc)
            inputTxMaps = []
            for m in range(self.numTx):
                imTx = np.zeros((256, 256))
                imTx[TXr[m], TXc[m]] = 1
                inputs.append(imTx)


        # print(len(inputs))


        if self.cityMap == "true":
            img_name_buildings = os.path.join(self.dir_buildings, nameMap)
            image_buildings = np.asarray(io.imread(img_name_buildings)) / 255
            inputs.append(image_buildings)


        # print(len(inputs))

        inputs = np.asarray(inputs, dtype=np.float32)
        inputs = np.transpose(inputs, (1, 2, 0))

        if self.transform:

            inputs = self.transform(inputs).type(torch.float32)

        RXlocr = torch.from_numpy(np.asarray(RXr, dtype=np.float32))
        RXlocc = torch.from_numpy(np.asarray(RXc, dtype=np.float32))
        loc = torch.stack((RXlocr, RXlocc), dim=0)

        heatmap = self.transform(heatmap).type(torch.float32)

        return inputs, heatmap, loc

class locDL3(Dataset):
    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="dataset/",
                 numTx=3,
                 numTrials=50,
                 numRx=200,
                 simulation="DPM",
                 cityMap="true",
                 carsMap="false",
                 TxMaps="true",
                 noise="false",
                 val=25,
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom".
            dir_dataset: directory of the dataset.
            numTx: Number of transmitters per map. Default and maximum numTx = 5.
            numTrials: Number of sets of numTx transmitters per map
            simulation:"DPM", "IRT2", "DPMtoIRT2", "DPMcars". Default= "DPM"
            cityMap: . Default cityMap="false"
            TxMaps: Images of Tx. Defaul TxMaps="false"
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())

        Output:
            inputs: The LocUNet inputs.
            RXlocr: Pixel row of true location
            RXlocc: Pixel column of true location

        """

        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 99, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 68
        elif phase == "val":
            self.ind1 = 69
            self.ind2 = 83
        elif phase == "test":
            self.ind1 = 84
            self.ind2 = 98
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.numTrials = numTrials
        self.numRx = numRx
        self.simulation = simulation
        self.cityMap = cityMap
        self.carsMap = carsMap
        self.TxMaps = TxMaps
        self.transform = transform
        self.noise = noise
        self.val = val

        self.height = 256
        self.width = 256

        if simulation == "DPM":
            self.dir_gainTrue = self.dir_dataset + "DPM/ToA/"
            self.dir_gainEst = self.dir_dataset + "DPM/ToA/"

        self.dir_buildings = self.dir_dataset + "buildings/"
        self.dir_cars = self.dir_dataset + "cars/"
        self.dir_Tx = self.dir_dataset + "png/antennas/"

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTrials * self.numRx

    def __getitem__(self, idx):
        numMapPhase = self.ind2 - self.ind1 + 1
        idxMap, idxTrial, idxRx = np.unravel_index(idx, (numMapPhase, self.numTrials, self.numRx))
        dataset_map_ind = self.maps_inds[idxMap + self.ind1]
        # names of files that depend only on the map:
        nameMap = str(dataset_map_ind) + ".png"

        # Load true (reported) radio maps for Txs:
        mat = np.load(r'C:\Users\陈琦\Desktop\python\LocUNet-main\lib\my_fileCorr.npy', allow_pickle='TRUE').item()
        rxx = mat['rxx']
        rxy = mat['rxy']
        RXr = rxx[dataset_map_ind, idxRx] - 1
        RXc = rxy[dataset_map_ind, idxRx] - 1
        antList = mat['antList']
        TXlist = antList[idxTrial, dataset_map_ind, :]

        inputEstMaps = []
        for m in range(self.numTx):
            name2 = str(dataset_map_ind) + "_" + str(TXlist[m] - 1) + ".png"
            img_name_gainTrue = os.path.join(self.dir_gainTrue, name2)
            image_gainTrue = np.asarray(io.imread(img_name_gainTrue)) / 255
            img_name_gainEst = os.path.join(self.dir_gainEst, name2)
            image_gainEst = np.asarray(io.imread(img_name_gainEst))

            if self.noise == 'true':
                image_gainEst = rss_noise_add(image_gainEst, self.val) / 255
            else:
                image_gainEst = image_gainEst / 255

            inputEstMaps.append(image_gainEst)
            gainTrue = image_gainTrue[RXr, RXc]
            imgGainTrue = gainTrue * np.ones(np.shape(image_gainEst))
            inputEstMaps.append(imgGainTrue)
        inputs = inputEstMaps

        # Load Tx maps
        if self.TxMaps == "true":
            antX = mat['antX']
            antY = mat['antY']
            TXr = antX[idxTrial, :, dataset_map_ind]
            TXc = antY[idxTrial, :, dataset_map_ind]
            inputTxMaps = []
            for m in range(self.numTx):
                imTx = np.zeros((256, 256))
                imTx[TXr[m], TXc[m]] = 1
                inputs.append(imTx)

        # Load buildings:
        if self.cityMap == "true":
            img_name_buildings = os.path.join(self.dir_buildings, nameMap)
            image_buildings = np.asarray(io.imread(img_name_buildings)) / 255
            inputs.append(image_buildings)

        # Load cars:
        if self.carsMap == "true":
            img_name_cars = os.path.join(self.dir_cars, nameMap)
            image_cars = np.asarray(io.imread(img_name_cars)) / 255
            inputs.append(image_cars)
        inputs = np.asarray(inputs, dtype=np.float32)
        inputs = np.transpose(inputs, (1, 2, 0))

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)

        # True coordinates
        RXlocr = torch.from_numpy(np.asarray(RXr, dtype=np.float32))
        RXlocc = torch.from_numpy(np.asarray(RXc, dtype=np.float32))

        return inputs, torch.stack((RXlocr, RXlocc), dim=0)


# # TOA 无噪
# class locDL(Dataset):
#     def __init__(self, maps_inds=np.zeros(1), phase="train",
#                  ind1=0, ind2=0,
#                  dir_dataset="dataset/",
#                  numTx=5,
#                  num_user=1,
#                  numTrials=50,
#                  numRx=200,
#                  width=256,
#                  height=256,
#                  simulation="DPM",
#                  cityMap="true",
#                  carsMap="false",
#                  TxMaps="true",
#                  noise="false",
#                  noise_val=5,
#                  transform=transforms.ToTensor()):
#
#         if maps_inds.size == 1:

#             self.maps_inds = np.arange(0, 99, 1, dtype=np.int16)

#             np.random.seed(42)
#             np.random.shuffle(self.maps_inds)
#         else:
#             self.maps_inds = maps_inds
#
#         if phase == "train":
#             self.ind1 = 0
#             self.ind2 = 68
#         elif phase == "val":
#             self.ind1 = 69
#             self.ind2 = 83
#         elif phase == "test":
#             self.ind1 = 84
#             self.ind2 = 98
#
#         else:  # 自定义范围
#             self.ind1 = ind1
#             self.ind2 = ind2
#
#         self.dir_dataset = dir_dataset
#         self.numTx = numTx
#         self.num_user = num_user
#         self.numTrials = numTrials
#         self.numRx = numRx
#         self.simulation = simulation
#         self.cityMap = cityMap
#         self.carsMap = carsMap
#         self.TxMaps = TxMaps
#         self.transform = transform
#         self.noise = noise
#         self.val = noise_val
#
#         self.height = width
#         self.width = height
#
#         if simulation == "DPM":
#             self.dir_gainTrue = self.dir_dataset + "DPM/ToA/"
#             self.dir_gainEst = self.dir_dataset + "DPM/ToA/"
#
#         self.dir_buildings = self.dir_dataset + "buildings/"
#         self.dir_Tx = self.dir_dataset + "png/antennas/"
#
#     def __len__(self):
#
#         return (self.ind2 - self.ind1 + 1) * self.numTrials * self.numRx
#
#     def __getitem__(self, idx):
#
#         numMapPhase = self.ind2 - self.ind1 + 1
#         idxMap, idxTrial, idxRx = np.unravel_index(idx, (numMapPhase, self.numTrials, self.numRx))
#         dataset_map_ind = self.maps_inds[idxMap + self.ind1]
#
#         nameMap = str(dataset_map_ind) + ".png"
#
#         mat = np.load(r'C:\Users\陈琦\Desktop\python\LocUNet-main\lib\my_fileCorr.npy', allow_pickle='TRUE').item()
#
#         rxx = mat['rxx']  # 99*200
#         # print(rxx[dataset_map_ind, :])
#         rxy = mat['rxy']  # 99*200
#         # print(rxy[dataset_map_ind, :])
#
#         multi_row = []
#         multi_column = []
#         if self.num_user > 1:
#             np.random.seed(idxRx)
#             indices = np.random.choice(rxx.shape[1], self.num_user, replace=False)
#             multi_row = rxx[dataset_map_ind, indices] - 1
#             multi_column = rxy[dataset_map_ind, indices] - 1
#         else:
#             multi_row.append(rxx[dataset_map_ind, idxRx] - 1)
#             multi_column.append(rxy[dataset_map_ind, idxRx] - 1)
#
#         antList = mat['antList']
#         TXlist = antList[idxTrial, dataset_map_ind, :]
#
#         inputEstMaps = []
#         for m in range(self.numTx):

#             name2 = str(dataset_map_ind) + "_" + str(TXlist[m] - 1) + ".png"
#             img_name_gainTrue = os.path.join(self.dir_gainTrue, name2)
#             image_gainTrue = np.asarray(io.imread(img_name_gainTrue)) / 255
#
#             img_name_gainEst = os.path.join(self.dir_gainEst, name2)
#             image_gainEst = np.asarray(io.imread(img_name_gainEst))
#             if self.noise == 'true':
#                 image_gainEst = rss_noise_add(image_gainEst, self.val) / 255
#             else:
#                 image_gainEst = image_gainEst / 255
#
#             inputEstMaps.append(image_gainEst)
#
#             for r, c in zip(multi_row, multi_column):
#                 gainTrue = image_gainTrue[r, c]
#                 imgGainTrue = gainTrue * np.ones(np.shape(image_gainEst))
#                 inputEstMaps.append(imgGainTrue)
#
#         inputs = inputEstMaps
#
#         if self.TxMaps == "true":
#             antX = mat['antX']
#             antY = mat['antY']
#             TXr = antX[idxTrial, :, dataset_map_ind]
#             TXc = antY[idxTrial, :, dataset_map_ind]
#             for m in range(self.numTx):
#                 imTx = np.zeros((256, 256))
#                 imTx[TXr[m], TXc[m]] = 1
#                 inputs.append(imTx)
#
#         if self.cityMap == "true":
#             img_name_buildings = os.path.join(self.dir_buildings, nameMap)
#             image_buildings = np.asarray(io.imread(img_name_buildings)) / 255
#             inputs.append(image_buildings)
#
#         inputs = np.asarray(inputs, dtype=np.float32)
#         inputs = np.transpose(inputs, (1, 2, 0))

#         if self.transform:
#             inputs = self.transform(inputs).type(torch.float32)

#         Loc = []
#         for rr, cc in zip(multi_row, multi_column):
#             RXlocr = torch.from_numpy(np.asarray(rr, dtype=np.float32).reshape(-1, 1))
#             RXlocc = torch.from_numpy(np.asarray(cc, dtype=np.float32).reshape(-1, 1))
#             loc = torch.cat((RXlocr, RXlocc), dim=1)
#             Loc.append(loc)
#
#         Loc = torch.cat(Loc, 0)
#
#         return inputs, Loc


class locDL9(Dataset):
    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="dataset/",
                 numTx=2,
                 numTrials=50,
                 numRx=200,
                 width=256,
                 height=256,
                 simulation="DPM",
                 cityMap="true",
                 carsMap="false",
                 TxMaps="true",
                 k=2,
                 sigma=2,
                 transform=transforms.ToTensor()):

        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 99, 1, dtype=np.int16)

            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 68
        elif phase == "val":
            self.ind1 = 69
            self.ind2 = 83
        elif phase == "test":
            self.ind1 = 84
            self.ind2 = 98

        else:  # define range
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.numTrials = numTrials
        self.numRx = numRx
        self.simulation = simulation
        self.cityMap = cityMap
        self.carsMap = carsMap
        self.TxMaps = TxMaps
        self.transform = transform

        self.height = width
        self.width = height

        self.k = k
        self.sigma = sigma

        if simulation == "DPM":
            self.dir_gainTrue = self.dir_dataset + "DPM/ToA/"
            self.dir_gainEst = self.dir_dataset + "DPM/ToA/"

        self.dir_buildings = self.dir_dataset + "buildings/"
        self.dir_Tx = self.dir_dataset + "png/antennas/"

    def __len__(self):

        return (self.ind2 - self.ind1 + 1) * self.numTrials * self.numRx

    def __getitem__(self, idx):
        numMapPhase = self.ind2 - self.ind1 + 1

        idxMap, idxTrial, idxRx = np.unravel_index(idx, (numMapPhase, self.numTrials, self.numRx))
        # print('idx = ',idx,'idxmap=',idxMap,'idxtrial=',idxTrial,'idxRx=',idxRx)
        # print('self.ind1=',self.ind1)
        dataset_map_ind = self.maps_inds[idxMap + self.ind1]

        # print(self.maps_inds)
        # print('dataset_map_ind=',dataset_map_ind)

        nameMap = str(dataset_map_ind) + ".png"

        mat = np.load(r'C:\Users\陈琦\Desktop\python\LocUNet-main\lib\my_fileCorr.npy', allow_pickle='TRUE').item()

        rxx = mat['rxx']  # 99*200
        # print(rxx[62, :])
        rxy = mat['rxy']  # 99*200
        # print(rxy[62, :])

        RXr = rxx[dataset_map_ind, idxRx]
        # print(dataset_map_ind, idxRx)
        RXc = rxy[dataset_map_ind, idxRx]

        mu_r = RXr * self.k
        mu_c = RXc * self.k

        x = np.arange(0, self.width * self.k, 1, np.float32)
        y = np.arange(0, self.height * self.k, 1, np.float32)

        # Standard normal with denominator terms
        antenna_r = (np.exp(- ((x - mu_r) ** 2) / (2 * self.sigma ** 2))) / (self.sigma * np.sqrt(np.pi * 2))
        antenna_c = (np.exp(- ((y - mu_c) ** 2) / (2 * self.sigma ** 2))) / (self.sigma * np.sqrt(np.pi * 2))

        # # Standard normal with no denominator terms
        # antenna_r = np.exp(- ((x - mu_r) ** 2) / (2 * self.sigma ** 2))
        # antenna_c = np.exp(- ((y - mu_c) ** 2) / (2 * self.sigma ** 2))

        antList = mat['antList']
        TXlist = antList[idxTrial, dataset_map_ind, :]

        inputEstMaps = []
        #  numTx=5
        for m in range(self.numTx):

            # loading true map
            name2 = str(dataset_map_ind) + "_" + str(TXlist[m] - 1) + ".png"
            img_name_gainTrue = os.path.join(self.dir_gainTrue, name2)
            image_gainTrue = np.asarray(io.imread(img_name_gainTrue)) / 255

            # loading estimate map
            img_name_gainEst = os.path.join(self.dir_gainEst, name2)
            image_gainEst = np.asarray(io.imread(img_name_gainEst)) / 255

            inputEstMaps.append(image_gainEst)

            # index RXr and RXc rss
            gainTrue = image_gainTrue[RXr, RXc]
            # index rss * (256×256)
            imgGainTrue = gainTrue * np.ones(np.shape(image_gainEst))

            inputEstMaps.append(imgGainTrue)

        inputs = inputEstMaps

        # # 10 dimension
        # print(len(inputs))

        # loading Tx map (5 TX)
        if self.TxMaps == "true":
            antX = mat['antX']
            antY = mat['antY']
            TXr = antX[idxTrial, :, dataset_map_ind]
            # print(TXr)
            TXc = antY[idxTrial, :, dataset_map_ind]
            # print(TXc)
            inputTxMaps = []
            for m in range(self.numTx):
                imTx = np.zeros((256, 256))
                imTx[TXr[m], TXc[m]] = 1
                inputs.append(imTx)

        # # 15维
        # print(len(inputs))

        # loading building
        if self.cityMap == "true":
            img_name_buildings = os.path.join(self.dir_buildings, nameMap)
            image_buildings = np.asarray(io.imread(img_name_buildings)) / 255
            inputs.append(image_buildings)

        # # 16 dimension
        # print(len(inputs))

        inputs = np.asarray(inputs, dtype=np.float32)
        inputs = np.transpose(inputs, (1, 2, 0))
        # to tensor
        if self.transform:
            # numpy to tensor
            inputs = self.transform(inputs).type(torch.float32)

        # target coordinate
        RXlocr = torch.from_numpy(np.asarray(RXr, dtype=np.float32))
        RXlocc = torch.from_numpy(np.asarray(RXc, dtype=np.float32))
        loc = torch.stack((RXlocr, RXlocc), dim=0)

        # 1d sequence
        antenna_r = torch.from_numpy(np.asarray(antenna_r, dtype=np.float32))
        antenna_c = torch.from_numpy(np.asarray(antenna_c, dtype=np.float32))

        target = torch.stack([antenna_r, antenna_c], dim=0)

        return inputs, target, loc

#  debug
def test():
    dataset = locDL(phase='train')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for x, y, z in loader:
        print(x.shape, y.shape, z)


if __name__ == "__main__":
    test()

