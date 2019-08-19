import os
import json
import tqdm
import pickle
import datetime
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from torch.utils import data
from torch.nn import functional as F
from model import OrdinalRegressionModel, RainFallClassification, AutoencoderBN, FocalLoss, OrdinalLoss
from dataloader import Dataset
from torchvision import transforms

featuresRoot = "/mnt/pami14/xyxu/dataset/ecCropData/features"
dataRoot = "/mnt/pami14/xyxu/dataset/ecCropData"


def featureMeanStdStatistics():
    meanFilePath = os.path.join(dataRoot, "mean.npy")
    stdFilePath = os.path.join(dataRoot, "std.npy")
    if not os.path.exists(meanFilePath) or not os.path.exists(stdFilePath):
        print("Start to load feature and cache the mean and std of features")
        cropFileList = [os.path.join(featuresRoot, fileName) for fileName in os.listdir(featuresRoot)]

        cacheList = []

        for fileName in tqdm.tqdm(cropFileList,total=len(cropFileList),
                                  desc="feature statistic",ncols=100,leave=False):
            features = np.load(fileName).reshape(37, -1)
            cacheList.append(features)

        concatenateFeatures = np.concatenate(cacheList, axis=1)
        np.save(meanFilePath, concatenateFeatures.mean(axis=1))
        np.save(stdFilePath, concatenateFeatures.std(axis=1))

    return np.load(meanFilePath), np.load(stdFilePath)


class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    # def image_summary(self,tag,images,step):
    #     pass

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a historgram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # drop first bin
        bin_edges = bin_edges[1:]

        # add bin edge and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)

        for c in counts:
            hist.bucket.append(c)

        # create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


class Trainer():
    def __init__(self, args, trainRegressionDataLoader,
                 trainRainDataLoader, testDataLoader, spaces, ranges):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainRegressionDataLoader = trainRegressionDataLoader
        self.trainRainDataLoader = trainRainDataLoader
        self.testDataLoader = testDataLoader

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

        self.denoiseModel = AutoencoderBN(self.noiseMean, self.noiseStd).to(self.device)
        self.regressionModel = OrdinalRegressionModel(self.nClass).to(self.device)
        self.rainFallClassifierModel = RainFallClassification().to(self.device)

        self.ordinalLoss = OrdinalLoss(spaces).to(self.device)
        self.rainFocalLoss = FocalLoss(2, alpha=0.25, gamma=2).to(self.device)

        self.regressionOptim = torch.optim.Adam([
            {'params': self.regressionModel.parameters(), 'lr': args.lr * 10,
             'weight_decay': args.weight_decay * 10},
            {'params': self.denoiseModel.parameters(), 'lr': args.lr,
             'weight_decay': args.weight_decay},
        ], lr=args.lr * 10, weight_decay=args.weight_decay * 10)

        self.rainFallOptim = torch.optim.Adam(self.rainFallClassifierModel.parameters(), lr=args.lr * 10)

        self.criterion = nn.MSELoss()

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.logger = Logger(self.out_path)

        with open(os.path.join(self.out_path, 'para.json'), 'w') as f:
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
        self.denoiseModel.eval()
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

        for batch_idx, (data, target, rainClass, _) in tqdm.tqdm(
                enumerate(self.testDataLoader), total=len(self.testDataLoader),
                desc='Test Data :', ncols=80,
                leave=False):

            rainNumpy = rainClass.numpy()
            gt_micaps = target.numpy()

            data = data.to(device=self.device)
            target = target.to(device=self.device)

            with torch.no_grad():
                encoder, decoder = self.denoiseModel(data)
                predictValues = self.regressionModel(encoder)
                rainPreds = self.rainFallClassifierModel(data)

                regressionValues = self.spaces * (torch.sum((predictValues > 0.5).float(), dim=1).view(-1, 1))
                rainPredsSoftMax = F.softmax(rainPreds, dim=1)
                rainWeights = torch.argmax(rainPredsSoftMax,dim=1,keepdim=True).float()
                # print(rainWeights)
                # print(rainPreds)
                # print(regressionValues)

                regressionValues = rainWeights * regressionValues

                # Three loss reconstruct loss , regression Loss and classification Loss
                regressionLoss = self.criterion(regressionValues, target)
                reconstructLoss = self.criterion(decoder, data)

                rainPredicted = rainPreds.cpu().numpy()

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
            info["modelParam"] = self.denoiseModel.state_dict()
            info["regressionParam"] = self.regressionModel.state_dict()
            info["rainFallParam"] = self.rainFallClassifierModel.state_dict()
            info["optimParam"] = self.regressionOptim.state_dict()
            torch.save(info, os.path.join(self.out_path, str(self.epoch) + "_checkpoints.pth"))

    def train_one_epoch_for_regression(self):
        self.denoiseModel.train()
        self.regressionModel.train()

        for batch_idx, (data, target, _, ordinalLabels) in tqdm.tqdm(
                enumerate(self.trainRegressionDataLoader), total=len(self.trainRegressionDataLoader),
                desc='Train Regression epoch=%d' % self.epoch, ncols=100, leave=False):
            iter_idx = batch_idx + self.epoch * len(self.trainRegressionDataLoader)
            self.iteration = iter_idx

            assert self.regressionModel.training
            self.regressionOptim.zero_grad()

            noise = torch.normal(mean=self.noiseMean, std=self.noiseStd)
            noiseData = data + noise
            data = data.to(self.device)
            noiseData = noiseData.to(self.device)
            ordinalLabels = ordinalLabels.to(self.device)

            encodes, decodes = self.denoiseModel(noiseData)
            ordinalPreds = self.regressionModel(encodes)

            constructLoss = self.criterion(decodes, data)
            ce, emd = self.ordinalLoss(ordinalPreds, ordinalLabels)

            loss = constructLoss + self.alpha + self.beta * emd
            loss.backward()

            self.regressionOptim.step()

            constructLossCpu = constructLoss.item()
            regressionLossCpu = ce.item()
            self.logger.scalar_summary("train_construct_loss", constructLossCpu, self.iteration + 1)
            self.logger.scalar_summary("train_regression_loss", regressionLossCpu, self.iteration + 1)

    def train_one_epoch_for_rainfall(self):
        classifyCorrection = [0] * 2
        classCount = [0] * 2
        acc = [0] * 3
        self.rainFallClassifierModel.train()

        for batch_idx, (data, target, rainLabel, _) in tqdm.tqdm(
                enumerate(self.trainRainDataLoader), total=len(self.trainRainDataLoader),
                desc='Train RainFall Classification epoch=%d' % self.epoch, ncols=100, leave=False):
            iter_idx = batch_idx + self.epoch * len(self.trainRainDataLoader)
            self.rainIteration = iter_idx

            assert self.rainFallClassifierModel.train
            self.rainFallOptim.zero_grad()

            logitNumpy = rainLabel.numpy()

            data = data.to(device=self.device)
            logits = rainLabel.to(device=self.device)

            preds = self.rainFallClassifierModel(data)

            classifyLoss = self.rainFocalLoss(preds, logits)
            classifyLoss.backward()

            self.rainFallOptim.step()

            classifyLossCpu = classifyLoss.item()

            predsSoftmax = F.softmax(preds, dim=1)
            # print(predsSoftmax.size())
            predicted = torch.argmax(predsSoftmax, dim=1, keepdim=True).cpu().numpy()
            # print(predicted.shape)

            for i in range(2):
                classifyCorrection[i] += np.sum((predicted == i) * (logitNumpy == i))
                classCount[i] += np.sum(logitNumpy == i)

            self.logger.scalar_summary("train_rain_classification_loss", classifyLossCpu,
                                       self.rainIteration + 1)
        for i in range(2):
            acc[i] += round(classifyCorrection[i] / classCount[i], 5)
        acc[2] += round(sum(classifyCorrection) / sum(classCount), 5)

        print("\nTrain Rain Fall Classification Accuracy : ", acc)

    def run(self):
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train_one_epoch_for_rainfall()

            if self.epoch % self.val_interval == 0:
                self.validate_one_epoch()

            self.train_one_epoch_for_regression()


if __name__ == "__main__":
    import argparse

    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()

    # parser.add_argument("--root_path", default="/mnt/pami14/DATASET/METEOROLOGY/ECforecast/GRIB", type=str)
    # parser.add_argument("--micaps_path", default="/mnt/pami14/DATASET/METEOROLOGY/GT_micaps", type=str)
    parser.add_argument("--save_root", default="/mnt/pami14/xyxu/ecforecast/dataset")
    parser.add_argument("--out", default="/mnt/pami14/xyxu/ecforecast")
    parser.add_argument("--featurePath",default="/mnt/pami14/xyxu/dataset/ecCropData/features",type=str)
    parser.add_argument("--labelPath",default="/mnt/pami14/xyxu/dataset/ecCropData/labels",type=str)
    # parser.add_argument("--cropRoot", default="/mnt/pami/xyxu/dataset/cropGrib")

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
    parser.add_argument("--gpuid", default="9", type=str)
    parser.add_argument("--interval", default=1)
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--featureNums", default=37)
    parser.add_argument("--earlyStop", default=10)
    parser.add_argument("--nClass", default=5, type=int)
    parser.add_argument("--maxRange", default=50, type=int)
    parser.add_argument("--cross", default=4, type=int)
    parser.add_argument("--noiseStd", default=1e-3, type=float)
    parser.add_argument("--spaces", default=0.5, type=float)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    run_datetime = datetime.datetime.now()
    args.out = os.path.join(args.out, run_datetime.strftime('%Y%m%d_%H%M%S.%f'), args.run_flag)

    featuresMean, featuresStd = featureMeanStdStatistics()

    # print(featuresMean.shape)
    # print(featuresStd.shape)

    transformer = transforms.Compose([
        transforms.Normalize(mean=featuresMean, std=featuresStd)
    ])

    print("========== Runing Result ========== ", args.out)
    spaces = args.spaces
    ranges = np.arange(0, 35, spaces)

    rainDataset = Dataset(args, transform=transformer, range=ranges, isTrain=True, isRain=False)
    regressionDataset = Dataset(args, transform=transformer, range=ranges, isTrain=True, isRain=True)
    testDataset = Dataset(args, transform=transformer, range=ranges, isTrain=False, isRain=False)

    trainRainDataloader = data.DataLoader(rainDataset, batch_size=args.batch_size, drop_last=True, shuffle=True,
                                          num_workers=5)
    trainRegressionDataloader = data.DataLoader(regressionDataset, batch_size=args.batch_size, drop_last=True,
                                                shuffle=True, num_workers=5)
    testDataloader = data.DataLoader(testDataset, batch_size=64, drop_last=False)

    trainer = Trainer(args,
                      trainRainDataLoader=trainRainDataloader,
                      trainRegressionDataLoader=trainRegressionDataloader,
                      testDataLoader=testDataloader,
                      spaces=spaces,
                      ranges=ranges)

    trainer.run()
