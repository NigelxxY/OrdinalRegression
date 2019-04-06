import os
import sys
import json
import tqdm
import pickle
import datetime
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.nn import functional as F
from torchvision import transforms
from model import OrdinalRegressionModel, rainFallClassification, AutoencoderBN, FocalLoss, ordinalLoss
from dataset import gribDataset, ordinalGribDataset
from util.logger import Logger
from collections import Counter

torch.multiprocessing.set_sharing_strategy('file_system')


def featureMeanStdStatistics():
    datasetRoot = "/mnt/pami/xyxu/dataset"
    cropRoot = "/mnt/pami/xyxu/dataset/cropGrib"
    meanFilePath = os.path.join(datasetRoot, "mean.npy")
    stdFilePath = os.path.join(datasetRoot, "std.npy")
    if not os.path.exists(meanFilePath) or not os.path.exists(stdFilePath):
        print("Start to load feature and cache the mean and std of features")
        cropFileList = [os.path.join(cropRoot, fileName) for fileName in os.listdir(cropRoot)]

        cacheList = []

        for fileName in cropFileList:
            with open(fileName, 'rb') as f:
                oneCropDataDict = pickle.load(f)
            for cropData in oneCropDataDict.values():
                features = cropData.values.reshape(37, 17 * 17)
                cacheList.append(features)

        concatenateFeatures = np.concatenate(cacheList, axis=1)
        np.save(meanFilePath, concatenateFeatures.mean(axis=1))
        np.save(stdFilePath, concatenateFeatures.std(axis=1))

    return np.load(meanFilePath), np.load(stdFilePath)


def weightStatistics(dataset):
    # nClass = 2
    # count = [0] * nClass
    # labelsClassIndex = dataset.labelsClass
    # count[dataset.nClass] += len(dataset) - len(labelsClassIndex)
    # counter = Counter(list(labelsClassIndex.values()))
    # for i in range(dataset.nClass):
    #     count[i] += counter[i]
    #
    # weightPerClass = [0.] * nClass
    # totalNum = float(sum(count))
    # for i in range(nClass):
    #     weightPerClass[i] = totalNum / float(count[i])
    #
    weightsList = [0] * len(dataset)
    # print(count)
    # print(weightsList)
    # weightPerClass = [1,3]

    for idx, item in enumerate(dataset):
        weightsList[idx] = 3 if item[1] >= 0.1 else 1

    return weightsList


def anotherWeright(dataset):
    weightPerClass = [25, 1]
    labelsClassIndex = dataset.labelsClass
    weightsList = [0] * len(dataset)
    for idx, item in enumerate(dataset):
        weightsList[idx] = weightPerClass[labelsClassIndex[idx]]

    return weightsList


class ordinalTrainer(object):
    def __init__(self, args,
                 trainRegressionDataLoader, trainRainFallLoader, testDataLoader,
                 spaces, ranges):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainRegressionDataLoader = trainRegressionDataLoader
        self.trainRainFallLoader = trainRainFallLoader
        self.testDataLoader = testDataLoader
        self.classificationLoader = trainRainFallLoader

        self.run_datetime = datetime.datetime.now()

        self.out_path = args.out
        self.alpha = args.alpha
        self.sigma = args.sigma
        self.beta = args.beta
        self.earlyStop = args.earlyStop
        self.spaces = spaces
        self.nClass = ranges.shape[0]

        self.noiseMean = torch.zeros(args.batch_size, args.featureNums, 17, 17)
        self.noiseStd = args.noiseStd

        self.model = AutoencoderBN(self.noiseMean, self.noiseStd).to(self.device)
        self.regressionModel = OrdinalRegressionModel(self.nClass).to(self.device)
        self.rainFallClassifierModel = rainFallClassification().to(self.device)

        self.ordinalLoss = ordinalLoss(spaces).to(self.device)
        self.rainFocalLoss = FocalLoss(2, alpha=0.25, gamma=2).to(self.device)

        self.regressionOptim = torch.optim.Adam([
            {'params': self.regressionModel.parameters(), 'lr': args.lr * 10,
             'weight_decay': args.weight_decay * 10},
            {'params': self.model.parameters(), 'lr': args.lr,
             'weight_decay': args.weight_decay},
        ],
            lr=args.lr * 10, weight_decay=args.weight_decay * 10)

        self.rainFallOptim = torch.optim.Adam(self.rainFallClassifierModel.parameters(), lr=args.lr * 10)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.regressionOptim, step_size=750 * 2)

        self.criterion = nn.MSELoss()

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.logger = Logger(self.out_path)

        with open(os.path.join(self.out_path, "para.json"), "w") as f:
            json.dump(args.__dict__, f)

        self.epoch = 0
        self.iteration = 0
        self.classificationIteration = 0
        self.rainfallclassificationIteration = 0
        self.test_step = 0
        self.max_epoch = args.epochs
        self.val_interval = args.interval
        self.res = 0
        self.bestConstructLoss = 1e7
        self.bestConstructEpoch = 0
        self.best_error = 1e7;
        self.best_res_epoch = 0

    def generateOneHot(self, softmax):
        maxIdxs = torch.argmax(softmax, dim=1, keepdim=True).cpu().long()
        oneHotMask = torch.zeros(softmax.shape, dtype=torch.float32)
        oneHotMask = oneHotMask.scatter_(1, maxIdxs, 1.0)
        oneHotMask = oneHotMask.unsqueeze(-2)
        return oneHotMask

    def validate_one_epoch(self):
        self.model.eval()
        self.regressionModel.eval()
        self.rainFallClassifierModel.eval()
        self.test_step += 1

        tsthreas = [0.1, 1, 10]

        tp = [0] * len(tsthreas)  # true positive
        tn = [0] * len(tsthreas)  # true negetive
        fp = [0] * len(tsthreas)  # false positve
        fn = [0] * len(tsthreas)  # false negetive
        ts = [0] * len(tsthreas)
        totalRegressionLoss = []
        totalReconstructLoss = []
        total_error = 0
        total_count = 0
        p_error = 0
        ps_error = 0
        p_count = 0

        rainCorrect = [0] * 2
        rainCount = [0] * 2
        rainAccuracy = [0] * 3

        for batch_idx, (data, target, rainClass, _, _, _) in tqdm.tqdm(
                enumerate(self.testDataLoader), total=len(self.testDataLoader),
                desc='Test Test Data :', ncols=80,
                leave=False):

            rainNumpy = rainClass.numpy()
            gt_micaps = target.numpy()

            data = data.to(device=self.device)
            target = target.to(device=self.device)

            with torch.no_grad():
                encoder, decoder = self.model(data)
                predictValues = self.regressionModel(encoder)
                rainPreds = self.rainFallClassifierModel(data)

                rainPredsSoftMax = F.softmax(rainPreds, dim=1)

                # if predict class belong to the last class , the output will be set zero

                rainOneHotMask = self.generateOneHot(rainPredsSoftMax).to(self.device)

                # print(regressionValues[0])
                regressionValues = self.spaces * (torch.sum((predictValues > 0.5).float(), dim=1).view(-1, 1))
                zeros = torch.zeros(regressionValues.size()).to(self.device)

                regressionValues = torch.matmul(rainOneHotMask,
                                                torch.cat([zeros, regressionValues], dim=1).unsqueeze(-1)).squeeze(-1)

                # Three loss reconstruct loss , regression Loss and classification Loss
                regressionLoss = self.criterion(regressionValues, target)
                reconstructLoss = self.criterion(decoder, data)

                rainPredicted = torch.argmax(rainPredsSoftMax, dim=1).cpu().numpy()

                for i in range(2):
                    rainCorrect[i] += np.sum((rainPredicted == i) * (rainNumpy == i))
                    rainCount[i] += np.sum(rainNumpy == i)

                predictNumpy = regressionValues.cpu().numpy()

                totalRegressionLoss.append(regressionLoss.item())
                totalReconstructLoss.append(reconstructLoss.item())

                gapValues = np.abs(predictNumpy - gt_micaps)

                total_error += np.sum(gapValues)
                total_count += gapValues.shape[0]
                # print(gt_micaps[:10])
                # print(one_hot_mask[:10])
                p_ae = (gt_micaps > 0.05) * gapValues
                p_error += np.sum(p_ae)
                ps_error += np.sum(p_ae ** 2)
                p_count += np.sum(gt_micaps > 0.05)

                for i, threas in enumerate(tsthreas):
                    tp[i] += np.sum((gt_micaps >= threas) * (predictNumpy >= threas))
                    tn[i] += np.sum((gt_micaps < threas) * (predictNumpy < threas))
                    fp[i] += np.sum((gt_micaps < threas) * (predictNumpy >= threas))
                    fn[i] += np.sum((gt_micaps >= threas) * (predictNumpy < threas))

        for i, _ in enumerate(tsthreas):
            ts[i] += round(tp[i] / (tp[i] + fp[i] + fn[i]), 5)

        totalAverageError = round(total_error / total_count, 5)
        pAverageError = round(p_error / p_count, 5)
        psAverageError = round(ps_error / p_count - pAverageError ** 2, 5)

        totalLoss = np.mean(totalRegressionLoss)
        totalRLoss = np.mean(totalReconstructLoss)

        for i in range(2):
            rainAccuracy[i] += round(rainCorrect[i] / rainCount[i], 5)
        rainAccuracy[2] += round(sum(rainCorrect) / sum(rainCount), 5)

        tsDisplay = list(zip(tp, tn, fp, fn, ts))

        info = {"test_regression_loss": totalLoss,
                "test_reconstruct_loss": totalRLoss,
                "aver_gap": totalAverageError,
                "aver_p_gap": pAverageError,
                "aver_ps_gap": psAverageError,
                "p_num": p_count,
                "ts_score": tsDisplay,
                "test_rain_classification_accuracy": rainAccuracy,
                }
        print("========================== Epoch {} Test Result Show ==========================".format(self.epoch + 1))
        print(info)

        if totalAverageError < self.best_error:
            self.best_error = totalAverageError
            self.best_res_epoch = self.epoch
            info["epoch"] = self.epoch
            info["modelParam"] = self.model.state_dict()
            info["regressionParam"] = self.regressionModel.state_dict()
            info["optimParam"] = self.regressionOptim.state_dict()
            torch.save(info, os.path.join(self.out_path, str(self.epoch) + "_checkpoints.pth"))

    def train_one_epoch_for_rainFall(self):
        classCorrect = [0] * 2
        classCounnt = [0] * 2
        accuray = [0] * 3
        self.rainFallClassifierModel.train()

        for batch_idx, (data, target, rainClass, rainMask, regressionClass, regressionMask) in tqdm.tqdm(
                enumerate(self.trainRainFallLoader), total=len(self.trainRainFallLoader),
                desc='Train RainFall Classification epoch=%d' % self.epoch, ncols=100, leave=False):
            iter_idx = batch_idx + self.epoch * len(self.trainRainFallLoader)
            self.rainfallclassificationIteration = iter_idx

            assert self.rainFallClassifierModel.train
            self.rainFallOptim.zero_grad()

            logitNumpy = rainClass.numpy()

            data = data.to(device=self.device)
            logits = rainClass.to(device=self.device)

            preds = self.rainFallClassifierModel(data)
            predsSoftmax = F.softmax(preds, dim=1)

            classificationLoss = self.rainFocalLoss(preds, logits)

            classificationLoss.backward()

            self.rainFallOptim.step()

            classificationLossCpu = classificationLoss.item()

            predicted = torch.argmax(predsSoftmax, dim=1).cpu().numpy()

            for i in range(2):
                classCorrect[i] += np.sum((predicted == i) * (logitNumpy == i))
                classCounnt[i] += np.sum(logitNumpy == i)

            self.logger.scalar_summary("train_rainfall_classification_loss", classificationLossCpu,
                                       self.rainfallclassificationIteration + 1)
        for i in range(2):
            accuray[i] += round(classCorrect[i] / classCounnt[i], 5)
        accuray[2] += round(sum(classCorrect) / sum(classCounnt), 5)

        print("\nTrain Rain Fall Classification Accuracy : ", accuray)

    def train_one_epoch_for_ordinalRegression(self):
        self.model.train()
        self.regressionModel.train()

        for batch_idx, (data, target, ordinalLabels) in tqdm.tqdm(
                enumerate(self.trainRegressionDataLoader), total=len(self.trainRegressionDataLoader),
                desc='Train Regression epoch=%d' % self.epoch, ncols=100, leave=False):
            iter_idx = batch_idx + self.epoch * len(self.trainRegressionDataLoader)
            # if (self.iteration != 0) and (iter_idx - 1) != self.iteration:
            #     continue
            self.iteration = iter_idx

            assert self.regressionModel.training
            self.regressionOptim.zero_grad()

            noise = torch.normal(mean=self.noiseMean, std=self.noiseStd)
            noisedData = data + noise
            data = data.to(self.device)
            noisedData = noisedData.to(self.device)
            ordinalLabels = ordinalLabels.to(self.device)
            # target = target.to(device=self.device)

            encoder, decoder = self.model(noisedData)
            ordinalPreds = self.regressionModel(encoder)

            constructLoss = self.criterion(decoder, data)
            ce, emd = self.ordinalLoss(ordinalPreds, ordinalLabels)

            loss = constructLoss + self.alpha * ce + self.beta * emd
            loss.backward()
            self.regressionOptim.step()

            constructLossCpu = constructLoss.item()
            regressionLossCpu = ce.item()
            self.logger.scalar_summary("train_construct_loss", constructLossCpu, self.iteration + 1)
            self.logger.scalar_summary("train_regression_loss", regressionLossCpu, self.iteration + 1)

    def run(self):
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train_one_epoch_for_rainFall()
            if self.epoch % args.interval == 0:
                self.validate_one_epoch()
            self.train_one_epoch_for_ordinalRegression()


if __name__ == "__main__":
    import argparse

    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", default="/mnt/pami/DATASET/METEOROLOGY/ECforecast/GRIB", type=str)
    parser.add_argument("--micaps_path", default="/mnt/pami/DATASET/METEOROLOGY/GT_micaps", type=str)
    parser.add_argument("--save_root", default="/mnt/pami/xyxu/ecforecast/dataset")
    parser.add_argument("--out", default="/mnt/pami14/xyxu/ecforecast")
    parser.add_argument("--cropRoot", default="/mnt/pami/xyxu/dataset/cropGrib")

    parser.add_argument("--startDate", default="20160801", type=str)
    parser.add_argument("--endDate", default="20160901", type=str)
    parser.add_argument("--run_flag", default="")
    parser.add_argument("--alpha", default=4.0, type=float)
    parser.add_argument("--sigma", default=0.0, type=float)
    parser.add_argument("--beta", default=0.05, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--sampleIndex", default=2, type=int)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--gpuid", default="1", type=str)
    parser.add_argument("--interval", default=1)
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--featureNums", default=37)
    parser.add_argument("--earlyStop", default=10)
    parser.add_argument("--nClass", default=5, type=int)
    parser.add_argument("--maxRange", default=50, type=int)
    parser.add_argument("--cross", default=4, type=int)
    parser.add_argument("--noiseStd",default=1e-3,type=float)
    parser.add_argument("--spaces",default=0.5,type=float)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    run_datetime = datetime.datetime.now()
    args.out = os.path.join(args.out, run_datetime.strftime('%Y%m%d_%H%M%S.%f'), args.run_flag)

    featuresMean, featuresStd = featureMeanStdStatistics()

    transformer = transforms.Compose([
        transforms.Normalize(mean=featuresMean, std=featuresStd)
    ])

    print("========== Runing Result ========== ", args.out)
    spaces = args.spaces
    ranges = np.arange(0,35,spaces)

    rainFallDataset = gribDataset(args, transform=transformer, isTrain=True)
    regressionDataSet = ordinalGribDataset(args, transform=transformer, range=ranges)
    testDataSet = gribDataset(args, transform=transformer, isTrain=False, model=rainFallDataset.cls)

    trainRegressionDataLoader = data.DataLoader(
        regressionDataSet, batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=5)

    trainRainFallLoader = data.DataLoader(
        rainFallDataset, batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=5)
    testDataLoader = data.DataLoader(testDataSet, batch_size=64, drop_last=False)

    trainer = ordinalTrainer(args,
                             trainRegressionDataLoader=trainRegressionDataLoader,
                             testDataLoader=testDataLoader,
                             trainRainFallLoader=trainRainFallLoader,
                             spaces=spaces, ranges=ranges)

    trainer.run()
