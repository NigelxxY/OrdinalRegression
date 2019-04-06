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
from model import Regression, regressionClassification, rainFallClassification, AutoencoderBN, MeanVarLoss, \
    MeanVarianceNormalizer, NormalizerLoss, FocalLoss, NonLocalRegression,MLP
from dataset import gribDataset, regressionGribDataset
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


class NonLocalTrainer(object):
    def __init__(self, args,
                 trainLoader, testLoader):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.out_path = args.out
        self.sigma = args.sigma
        self.beta = args.beta
        self.nClass = args.nClass

        self.model = MLP().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = nn.MSELoss()

        self.trainLoader = trainLoader
        self.testLoader = testLoader

        self.run_datetime = datetime.datetime.now()

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.logger = Logger(self.out_path)

        with open(os.path.join(self.out_path, "para.json"), "w") as f:
            json.dump(args.__dict__, f)

        self.epoch = 0
        self.iteration = 0
        self.test_step = 0
        self.max_epoch = args.epochs
        self.val_interval = args.interval
        self.res = 0
        self.best_error = 1e7;
        self.best_res_epoch = 0

        self.noiseMean = torch.zeros(args.batch_size, args.featureNums, 17, 17)
        self.noiseStd = torch.div(torch.ones(args.batch_size, args.featureNums, 17, 17), 1e3)

    def validate_one_epoch(self):
        self.model.eval()
        self.test_step += 1

        tsthreas = [0.1, 1, 10]

        tp = [0] * len(tsthreas)  # true positive
        tn = [0] * len(tsthreas)  # true negetive
        fp = [0] * len(tsthreas)  # false positve
        fn = [0] * len(tsthreas)  # false negetive
        ts = [0] * len(tsthreas)

        totalRegressionLoss = []
        total_error = 0
        total_count = 0
        p_error = 0
        p_count = 0

        largeGapCount = 0
        largeGap = 0

        for batch_idx, (data, target, _, _, _, _) in tqdm.tqdm(
                enumerate(self.testLoader), total=len(self.testLoader),
                desc='Valid :', ncols=80,
                leave=False):
            gt_micaps = target.numpy()
            data, target = data.to(device=self.device), target.to(device=self.device)

            with torch.no_grad():

                predictValues = self.model(data)

                regressionLoss = self.criterion(predictValues, target)

                predictNumpy = predictValues.cpu().numpy()
                totalRegressionLoss.append(regressionLoss.item())
                # totalClassificationLoss.append(classificationLoss.item())

                # predicted = torch.argmax(preds, dim=1)
                # correct += (predicted == logits).sum().item()

                gapValues = np.abs(predictNumpy - gt_micaps)
                total_error += np.sum(gapValues)
                total_count += gt_micaps.shape[0]
                p_error += np.sum((gt_micaps > 0.01) * gapValues)
                p_count += np.sum(gt_micaps > 0.01)

                largeGap += np.sum((gapValues > 5) * gapValues)
                largeGapCount += np.sum(gapValues > 5)

                for i, threas in enumerate(tsthreas):
                    tp[i] += np.sum((gt_micaps >= threas) * (predictNumpy >= threas))
                    tn[i] += np.sum((gt_micaps < threas) * (predictNumpy < threas))
                    fp[i] += np.sum((gt_micaps < threas) * (predictNumpy >= threas))
                    fn[i] += np.sum((gt_micaps >= threas) * (predictNumpy < threas))

        for i, _ in enumerate(tsthreas):
            ts[i] += round(tp[i] / (tp[i] + fp[i] + fn[i]), 5)

        totalAverageError = round(total_error / total_count, 5)
        pAverageError = round(p_error / p_count, 5)
        totalLoss = np.sum(totalRegressionLoss)
        largeGapRatio = round(largeGapCount / total_count, 5)
        largeGapMae = round(largeGap / largeGapCount, 5)

        info = {"test_regression_loss": totalLoss,
                "ts_score": ts,
                "aver_gap": totalAverageError,
                "aver_p_gap": pAverageError,
                "large_gap_ratio": largeGapRatio,
                "large_gap_mae": largeGapMae
                }
        print("========================== Epoch {} Test Result Show ==========================".format(self.epoch + 1))

        print(info)

        # for tag, value in info.items():
        #     self.logger.scalar_summary(tag, value, self.test_step)

        # if totalAverageError < self.best_error:
        #     self.best_error = totalAverageError
        #     self.best_res_epoch = self.epoch
        #     info["epoch"] = self.epoch
        #     info["modelParam"] = self.model.state_dict()
        #     info["optimParam"] = self.optim.state_dict()
        #     torch.save(info, os.path.join(self.out_path, str(self.epoch) + "_checkpoints.pth"))

    def train_one_epoch(self):
        self.model.train()

        for batch_idx, (data, target, _, _, _, _) in tqdm.tqdm(
                enumerate(self.trainLoader), total=len(self.trainLoader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iter_idx = batch_idx + self.epoch * len(self.trainLoader)
            # if (self.iteration != 0) and (iter_idx - 1) != self.iteration:
            #     continue
            self.iteration = iter_idx

            assert self.model.training
            self.optim.zero_grad()

            data = data.to(device=self.device)
            target = target.to(device=self.device)

            predictValues = self.model(data)

            regressionLoss = self.criterion(predictValues, target)

            regressionLoss.backward()
            # for named,param in self.model.named_parameters():
            #     print("Name : " ,named)
            #     print(param.grad.data.sum())
            self.optim.step()

            regressionLossCpu = regressionLoss.item()
            self.logger.scalar_summary("train_regression_loss", regressionLossCpu, self.iteration + 1)

        for tag, value in self.model.named_parameters():
            self.logger.histo_summary(tag, value.data.cpu().numpy(), self.epoch + 1)
            self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), self.epoch + 1)

    def run(self):
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train_one_epoch()
            if (self.epoch + 1) % self.val_interval == 0:
                self.validate_one_epoch()


class AutoEncoderTrainer(object):
    def __init__(self, args,
                 trainRegressionDataLoader, trainRegressionClassificationLoader,
                 testDataLoader, trainRainFallLoader,
                 means, std):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainRegressionDataLoader = trainRegressionDataLoader
        self.trainRegressionClassificationLoader = trainRegressionClassificationLoader
        self.testDataLoader = testDataLoader
        self.classificationLoader = trainRainFallLoader

        self.run_datetime = datetime.datetime.now()

        self.out_path = args.out
        self.sigma = args.sigma
        self.beta = args.beta
        self.earlyStop = args.earlyStop
        self.nClass = args.nClass

        self.noiseMean = torch.zeros(args.batch_size, args.featureNums, 17, 17)
        self.noiseStd = 1e-3

        self.model = AutoencoderBN(self.noiseMean, self.noiseStd).to(self.device)
        self.regressionModel = Regression(self.nClass).to(self.device)
        self.classificationModel = regressionClassification(self.nClass).to(self.device)

        self.rainFallClassifierModel = rainFallClassification().to(self.device)
        self.meanStdNormalizer = MeanVarianceNormalizer(means, std).to(self.device)

        self.meanvarLoss = MeanVarLoss(self.nClass).to(self.device)
        self.normaliedLoss = NormalizerLoss(std).to(self.device)
        self.focalLoss = FocalLoss(self.nClass, alpha=0.25, gamma=2).to(self.device)
        self.rainFocalLoss = FocalLoss(2, alpha=0.25, gamma=2).to(self.device)

        self.regressionOptim = torch.optim.Adam([
            {'params': self.regressionModel.parameters(), 'lr': args.lr,
             'weight_decay': args.weight_decay},
            {'params': self.model.parameters(), 'lr': args.lr,
             'weight_decay': args.weight_decay},
        ],
            lr=args.lr * 10, weight_decay=args.weight_decay * 10)

        self.classificationOptim = torch.optim.Adam(self.classificationModel.parameters(), lr=args.lr * 100)

        self.rainFallOptim = torch.optim.Adam(self.rainFallClassifierModel.parameters(), lr=args.lr * 10)

        # self.reconstructOptim = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.regressionOptim, step_size=750 * 2)

        self.criterion = nn.MSELoss()

        self.classificationCriterion = nn.CrossEntropyLoss()

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

    def mask_norm(self, mask, threshold=0.5):

        mask_ = mask * (mask > threshold).float()
        mask_ = mask_ / mask_.sum(1).unsqueeze(-1)
        return mask_

    def generateOneHot(self, softmax):
        maxIdxs = torch.argmax(softmax, dim=1, keepdim=True).cpu().long()
        oneHotMask = torch.zeros(softmax.shape, dtype=torch.float32)
        oneHotMask = oneHotMask.scatter_(1, maxIdxs, 1.0)
        oneHotMask = oneHotMask.unsqueeze(-2)
        return oneHotMask

    def validate_one_epoch(self):
        self.model.eval()
        self.regressionModel.eval()
        self.classificationModel.eval()
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
        totalClassificationLoss = []
        totalRClassificationLoss = []
        total_error = 0
        total_count = 0
        p_error = 0
        ps_error = 0
        p_count = 0

        pxErrorList = [0] * (self.nClass)
        pxsErrorList = [0] * (self.nClass)
        pxCountList = [0] * (self.nClass)
        pxAverageError = [0] * (self.nClass)
        pxsAverageError = [0] * (self.nClass)

        classCorrect = [0] * (self.nClass)
        classCounnt = [0] * (self.nClass)
        accuray = [0] * (self.nClass + 1)

        rainCorrect = [0] * 2
        rainCount = [0] * 2
        rainAccuracy = [0] * 3

        for batch_idx, (data, target, rainClass, rainMask, regressionClass, regressionMask) in tqdm.tqdm(
                enumerate(self.testDataLoader), total=len(self.testDataLoader),
                desc='Test Test Data :', ncols=80,
                leave=False):

            rainNumpy = rainClass.numpy()
            regressionNumpy = regressionClass.numpy()
            one_hot_mask = regressionMask.numpy()
            gt_micaps = target.numpy()

            data = data.to(device=self.device)
            target = target.to(device=self.device)
            rainClass = rainClass.to(device=self.device)
            # rainMask = rainMask.to(device=self.device)
            regressionClass = regressionClass.to(device=self.device)
            regressionMask = regressionMask.to(device=self.device).unsqueeze(-2)

            with torch.no_grad():
                encoder, decoder = self.model(data)
                predictValues = self.regressionModel(encoder)
                rainPreds = self.rainFallClassifierModel(data)
                regressionPreds = self.classificationModel(data)

                rainPredsSoftMax = F.softmax(rainPreds, dim=1)
                regressionPredsSoftmax = F.softmax(regressionPreds, dim=1)

                # if predict class belong to the last class , the output will be set zero

                rainOneHotMask = self.generateOneHot(rainPredsSoftMax).to(self.device)
                regressionOneHotMask = self.generateOneHot(regressionPredsSoftmax).to(self.device)

                predictValues = self.meanStdNormalizer(predictValues).unsqueeze(-1)
                # print(predictValues[0])

                regressionValues = torch.matmul(regressionOneHotMask, predictValues).squeeze(-1)
                # print(regressionValues[0])
                zeros = torch.zeros(regressionValues.size()).to(self.device)

                regressionValues = torch.matmul(rainOneHotMask,
                                                torch.cat([zeros, regressionValues], dim=1).unsqueeze(-1)).squeeze(-1)

                # print("res: ",regressionValues[:10])
                # print("resSum: ",regressionValues.mean())
                # print("target: ",target[:10])

                # print(regressionValues[0])
                # print(target[0])

                # Three loss reconstruct loss , regression Loss and classification Loss
                regressionLoss = self.criterion(regressionValues, target)
                reconstructLoss = self.criterion(decoder, data)
                rainClassificationLoss = self.classificationCriterion(rainPreds, rainClass)
                regressionClassificationLoss = self.classificationCriterion(regressionPreds, regressionClass)

                rainPredicted = torch.argmax(rainPredsSoftMax, dim=1).cpu().numpy()
                predicted = torch.argmax(regressionPredsSoftmax, dim=1).cpu().numpy()

                for i in range(self.nClass):
                    classCorrect[i] += np.sum((predicted == i) * (regressionNumpy == i) * (rainNumpy == 1))
                    classCounnt[i] += np.sum((regressionNumpy == i) * (rainNumpy == 1))

                for i in range(2):
                    rainCorrect[i] += np.sum((rainPredicted == i) * (rainNumpy == i))
                    rainCount[i] += np.sum(rainNumpy == i)

                predictNumpy = regressionValues.cpu().numpy()
                # biasNumpy = resValues.cpu().numpy()
                # labelsIndex = predicted.cpu().numpy()
                # predictNumpy = np.array([[biasNumpy[i,0]*(idx<self.nClass) + self.center[idx]] for i,idx in enumerate(labelsIndex)])

                totalRegressionLoss.append(regressionLoss.item())
                totalReconstructLoss.append(reconstructLoss.item())
                totalClassificationLoss.append(regressionClassificationLoss.item())
                totalRClassificationLoss.append(rainClassificationLoss.item())

                gapValues = np.abs(predictNumpy - gt_micaps)

                total_error += np.sum(gapValues)
                total_count += gapValues.shape[0]
                # print(gt_micaps[:10])
                # print(one_hot_mask[:10])
                p_ae = (gt_micaps > 0.05) * gapValues
                p_error += np.sum(p_ae)
                ps_error += np.sum(p_ae ** 2)
                p_count += np.sum(gt_micaps > 0.05)

                for i in range(self.nClass):
                    ae = one_hot_mask[:, i].reshape(-1, 1) * gapValues
                    pxErrorList[i] += np.sum(ae)
                    pxsErrorList[i] += np.sum(ae ** 2)
                    pxCountList[i] += np.sum(one_hot_mask[:, i])

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

        for i in range(self.nClass):
            pxAverageError[i] += round(pxErrorList[i] / pxCountList[i], 5)
            pxsAverageError[i] += round(pxsErrorList[i] / pxCountList[i] - pxAverageError[i] ** 2, 5)

        totalLoss = np.mean(totalRegressionLoss)
        totalRLoss = np.mean(totalReconstructLoss)
        totalCLoss = np.mean(totalClassificationLoss)

        for i in range(self.nClass):
            accuray[i] += round(classCorrect[i] / classCounnt[i], 5)
        accuray[self.nClass] += round(sum(classCorrect) / sum(classCounnt), 5)

        for i in range(2):
            rainAccuracy[i] += round(rainCorrect[i] / rainCount[i], 5)
        rainAccuracy[2] += round(sum(rainCorrect) / sum(rainCount), 5)

        info = {"test_regression_loss": totalLoss,
                "test_reconstruct_loss": totalRLoss,
                "test_classification_loss": totalCLoss,
                "aver_gap": totalAverageError,
                "aver_p_gap": pAverageError,
                "aver_ps_gap": psAverageError,
                "p_num": p_count,
                }

        tsDisplay = list(zip(tp, tn, fp, fn, ts))

        classStatistics = {
            "average_p_gap": pxAverageError,
            "aver_p_s_gap": pxsAverageError,
            "p_count": pxCountList,
            "ts_score": tsDisplay,
            "test_rain_classification_accuracy": rainAccuracy,
            "test_classification_accuracy": accuray,
        }

        print(info)
        print(classStatistics)

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
                enumerate(self.classificationLoader), total=len(self.classificationLoader),
                desc='Train RainFall Classification epoch=%d' % self.epoch, ncols=100, leave=False):
            iter_idx = batch_idx + self.epoch * len(self.classificationLoader)
            self.rainfallclassificationIteration = iter_idx

            assert self.rainFallClassifierModel.train
            self.rainFallOptim.zero_grad()

            logitNumpy = rainClass.numpy()

            data = data.to(device=self.device)
            logitsFloat = rainClass.float().to(device=self.device)
            logits = rainClass.to(device=self.device)

            preds = self.rainFallClassifierModel(data)
            predsSoftmax = F.softmax(preds, dim=1)

            classificationLoss = self.rainFocalLoss(preds, logits)
            # classificationLoss = self.classificationCriterion(preds, logits)

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

        print("Train Rain Fall Classification Accuracy : ", accuray)

    def train_one_epoch_for_classification(self):
        classCorrect = [0] * self.nClass
        classCounnt = [0] * self.nClass
        accuray = [0] * (self.nClass + 1)
        self.classificationModel.train()
        for batch_idx, (data, target, rainClass, rainMask, regressionClass, regressionMask) in tqdm.tqdm(
                enumerate(self.trainRegressionClassificationLoader),
                total=len(self.trainRegressionClassificationLoader),
                desc='Train Classification epoch=%d' % self.epoch, ncols=100, leave=False):
            iter_idx = batch_idx + self.epoch * len(self.trainRegressionClassificationLoader)
            self.classificationIteration = iter_idx

            assert self.classificationModel.train
            self.classificationOptim.zero_grad()

            logitNumpy = regressionClass.numpy()

            data = data.to(device=self.device)
            logitsFloat = regressionClass.float().to(device=self.device)
            logits = regressionClass.to(device=self.device)

            preds = self.classificationModel(data)
            predsSoftmax = F.softmax(preds, dim=1)

            classificationLoss = self.focalLoss(preds, logits)
            # classificationLoss = self.classificationCriterion(preds, logits)
            meanLoss, varLoss = self.meanvarLoss(predsSoftmax, logitsFloat.unsqueeze(-1))

            loss = classificationLoss + 1 * meanLoss + 0.5 * varLoss
            loss.backward()

            self.classificationOptim.step()

            classificationLossCpu = loss.item()

            predicted = torch.argmax(predsSoftmax, dim=1).cpu().numpy()

            for i in range(self.nClass):
                classCorrect[i] += np.sum((predicted == i) * (logitNumpy == i))
                classCounnt[i] += np.sum(logitNumpy == i)

            self.logger.scalar_summary("train_classification_loss", classificationLossCpu,
                                       self.classificationIteration + 1)

        for i in range(self.nClass):
            accuray[i] += round(classCorrect[i] / classCounnt[i], 5)
        accuray[self.nClass] += round(sum(classCorrect) / sum(classCounnt), 5)

        print("Train classification Accuracy : ", accuray)

    def train_one_epoch_for_regression(self):
        self.model.train()
        self.regressionModel.train()

        for batch_idx, (data, target, rainClass, rainMask, regressionClass, regressionMask) in tqdm.tqdm(
                enumerate(self.trainRegressionDataLoader), total=len(self.trainRegressionDataLoader),
                desc='Train Regression epoch=%d' % self.epoch, ncols=100, leave=False):
            iter_idx = batch_idx + self.epoch * len(self.trainRegressionDataLoader)
            # if (self.iteration != 0) and (iter_idx - 1) != self.iteration:
            #     continue
            self.iteration = iter_idx

            assert self.regressionModel.training
            self.regressionOptim.zero_grad()

            # noise = torch.randn(data.size()).to(device=self.device)
            noise = torch.normal(mean=self.noiseMean, std=self.noiseStd).to(self.device)

            # noise = torch.normal(mean=self.noiseMean, std=self.noiseStd).to(device=self.device)
            data = data.to(device=self.device)

            rainMask = rainMask.to(device=self.device)
            regressionMask = regressionMask.to(device=self.device)

            noisedData = data + noise
            target = target.to(device=self.device)

            encoder, decoder = self.model(noisedData)
            predictValues = self.regressionModel(encoder)

            # thresholdMask = labels.narrow(1, 0, self.nClass).view(-1, 1, self.nClass)
            predictValues = self.meanStdNormalizer(predictValues)
            # resValues = torch.matmul(thresholdMask, predictValues).squeeze(-1)

            regressionLoss = self.normaliedLoss(predictValues, target, rainMask, regressionMask)
            # regressionLoss = self.criterion(resValues, target)
            constructLoss = self.criterion(decoder, data)
            # classificationLoss = self.classificationCriterion(preds, logits)
            # meanLoss, varLoss = self.meanvarLoss(predictClassesSoftmax, logitsFloat.unsqueeze(-1))

            loss = constructLoss + self.sigma * regressionLoss
            # loss = constructLoss + self.sigma* regressionLoss
            loss.backward()
            # for param in self.model.parameters():
            #     print(param.grad.data.sum())
            self.regressionOptim.step()

            constructLossCpu = constructLoss.item()
            regressionLossCpu = regressionLoss.item()
            self.logger.scalar_summary("train_construct_loss", constructLossCpu, self.iteration + 1)
            self.logger.scalar_summary("train_regression_loss", regressionLossCpu, self.iteration + 1)

    def run(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Experiments ', ncols=100):
            self.epoch = epoch
            self.train_one_epoch_for_rainFall()
            self.train_one_epoch_for_classification()
            if self.epoch % args.interval == 0:
                self.validate_one_epoch()
            self.train_one_epoch_for_regression()


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
    parser.add_argument("--sigma", default=1, type=float)
    parser.add_argument("--beta", default=2, type=float)
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
    parser.add_argument("--cross", default=0, type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    run_datetime = datetime.datetime.now()
    args.out = os.path.join(args.out, run_datetime.strftime('%Y%m%d_%H%M%S.%f'), args.run_flag)

    featuresMean, featuresStd = featureMeanStdStatistics()

    transformer = transforms.Compose([
        transforms.Normalize(mean=featuresMean, std=featuresStd)
    ])

    print("========== Runing Result ========== ", args.out)

    trainDataSet = gribDataset(args, isTrain=True, transform=transformer)
    testDataSet = gribDataset(args, transform=transformer, isTrain=False, model=trainDataSet.cls)

    trainDataLoader = data.DataLoader(
        trainDataSet,batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=5
    )

    testDataLoader = data.DataLoader(testDataSet, batch_size=64, drop_last=False)

    trainer = NonLocalTrainer(args,trainDataLoader,testDataLoader)

    trainer.run()
