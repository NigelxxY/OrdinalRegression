import os
import sys
import tqdm
import torch
import torch.nn as nn
import datetime

from model import FcnForRegression
from util.util import cal_ts, grib_bilinear_interpolation
from util.logger import Logger


# full grib trainer
class Trainer(object):
    def __init__(self, model, optimizer, device,
                 train_loader, val_loader,
                 out, max_epoch, val_interval):

        self.cuda = True

        self.device = device
        self.model = model
        self.optim = optimizer
        self.loss = nn.MSELoss

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.run_datetime = datetime.datetime.now()

        self.out_path = out
        self.log_file = os.path.join(self.out_path, "log.csv")

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.log_header = [
            'epoch',
            'iteration',
            'train/loss',
            'val/loss',
            'val/tsscore',
            'val/error',
            'elapsed_time'
        ]

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write(','.join(self.log_header) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.val_interval = val_interval
        self.res = 0
        self.best_res = 0
        self.best_res_epoch = 0

    def validate_one_epoch(self):
        training = self.model.training
        self.model.eval()

        # val_loss = 0
        tp = 0  # true positive
        tn = 0  # true negetive
        fp = 0  # false positve
        fn = 0  # false negetive
        total_loss = 0

        for batch_idx, (data, lats, lons, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid :', ncols=80,
                leave=False):

            # data, lats, lons = data.to(self.device), lats.to(self.device), lons.to(self.device)
            data = data.to(device=self.device, dtype=torch.float)

            with torch.no_grad():
                grib = self.model(data)

            interpolate_loss = torch.zeros([1], requires_grad=False).to(self.device)

            for item in target:
                p_value = grib_bilinear_interpolation(grib, torch.squeeze(lats).numpy(), torch.squeeze(lons).numpy(),
                                                      item[0].item(), item[1].item())
                p_value_cpu = p_value.item()
                gt_micaps = item[2].item()
                # print("Predict {} and true {}".format(p_value_cpu,gt_micaps))
                interpolate_loss = self.loss()(p_value, item[2].to(device=self.device, dtype=torch.float))
                if p_value_cpu > 0.1 and gt_micaps > 0.1:
                    tp += 1
                elif p_value_cpu > 0.1 and gt_micaps < 0.1:
                    tn += 1
                elif p_value_cpu < 0.1 and gt_micaps > 0.1:
                    fp += 1
                else:
                    fn += 1

            total_loss += interpolate_loss.item()

        ts = cal_ts((tp + fn), fp, tn)

        if ts > self.best_res:
            self.best_res_epoch = self.epoch
            save_filename = os.path.join(self.out_path, "epoch_" + str(self.epoch) + "_best_ts_" + str(ts) + ".pt")
            # torch.save()
            # self.model.save_state_dict(os.path.join(self.out_path,"epoch_"+str(self.epoch)+"_best_ts_"+str(ts)+".pt"))

        with open(self.log_file, 'a') as f:
            elapsed_time = (datetime.datetime.now() - self.run_datetime).total_seconds()
            log = [self.epoch] + ['-1'] * 2 + [total_loss] + [ts] + [elapsed_time]
            log = map(str, log)
            f.write('|'.join(log) + '\n')

        if self.epoch - self.best_res_epoch > 1000:
            print("Early Stop")
            sys.exit(0)

    def train_one_epoch(self):
        self.model.train()
        # print(self.model.training)

        for batch_idx, (data, lats, lons, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iter_idx = batch_idx + self.epoch * len(self.train_loader)
            if (self.iteration != 0) and (iter_idx - 1) != self.iteration:
                continue
            self.iteration = iter_idx

            # if self.iteration % self.val_interval == 0:
            #     self.validate_one_day()

            assert self.model.training

            data = data.to(device=self.device, dtype=torch.float)
            # print(data)

            self.optim.zero_grad()
            grib = self.model(data)
            interpolate_loss = torch.zeros([1], requires_grad=True).to(self.device)
            # print("Target size is: ",len(target))


            for item in target:
                gt_micaps = item[2]
                predict_value = grib_bilinear_interpolation(grib, torch.squeeze(lats).numpy(),
                                                            torch.squeeze(lons).numpy(), item[0].item(), item[1].item())
                onelabel_loss = self.loss()(predict_value, gt_micaps.to(self.device, dtype=torch.float))
                print("Predict values: {} and gt values is {}".format(predict_value.item(), gt_micaps.item()))
                interpolate_loss += onelabel_loss

            interpolate_loss.backward()
            # for param in self.model.parameters():
            #     print("Param: ",param.data)

            # for param in self.
            # .parameters():
            #     print("Grad: ",param.grad.data)
            self.optim.step()

            with open(self.log_file, 'a') as f:
                elapsed_time = (datetime.datetime.now() - self.run_datetime).total_seconds()
                log = [self.epoch, self.iteration] + [interpolate_loss.item()] + ['-1'] * 2 + [elapsed_time]
                log = map(str, log)
                f.write('|'.join(log) + '\n')

    def run(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_one_epoch()
            if self.epoch % self.val_interval == 0:
                self.validate_one_epoch()


# slice grib trainer
class SliceTrainer(object):
    def __init__(self, args, device,
                 train_loader, val_loader,
                 out, max_epoch, val_interval):

        self.cuda = True

        self.device = device
        self.model = FcnForRegression().to(device=device)
        self.optim = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.run_datetime = datetime.datetime.now()

        self.out_path = out
        # print(self.out_path)
        # self.log_file = os.path.join(self.out_path, "log.csv")

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.logger = Logger(self.out_path)

        # self.log_header = [
        #     'epoch',
        #     'iteration',
        #     'train/loss',
        #     'val/loss',
        #     'val/tsscore',
        #     'elapsed_time'
        # ]
        #
        # if not os.path.exists(self.log_file):
        #     with open(self.log_file, 'w') as f:
        #         f.write(','.join(self.log_header) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.test_step = 0
        self.max_epoch = max_epoch
        self.val_interval = val_interval
        self.res = 0
        self.best_res = 0
        self.best_res_epoch = 0

    def validate_one_epoch(self):
        self.model.eval()
        self.test_step += 1

        tp = 0  # true positive
        tn = 0  # true negetive
        fp = 0  # false positve
        fn = 0  # false negetive
        total_loss = 0
        total_error = 0
        p_error = 0
        p_count = 0

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid :', ncols=80,
                leave=False):

            data, target = data.to(device=self.device, dtype=torch.float), target.to(device=self.device,
                                                                                     dtype=torch.float)

            with torch.no_grad():
                predict_value = self.model(data)

            loss = self.criterion(predict_value, target)
            total_loss += loss.item()

            p_value_cpu = predict_value.item()
            gt_micaps = target.item()
            gap_value = abs(gt_micaps - p_value_cpu)
            total_error += gap_value
            if gt_micaps > 0:
                p_error += gap_value
                p_count += 1

            if p_value_cpu > 0.1 and gt_micaps > 0.1:
                tp += 1
            elif p_value_cpu > 0.1 and gt_micaps < 0.1:
                tn += 1
            elif p_value_cpu < 0.1 and gt_micaps > 0.1:
                fp += 1
            else:
                fn += 1

        ts = cal_ts(tp, fp, tn)
        p_error /= p_count

        info = {"test_loss": total_loss,
                "ts_score": ts,
                "aver_gap": (total_error/len(self.val_loader)),
                "aver_p_gap": p_error}

        print(info)

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, self.test_step)

        # if ts > self.best_res:
        #     self.best_res_epoch = self.epoch
        #     save_filename = os.path.join(self.out_path,"epoch_"+str(self.epoch)+"_best_ts_"+str(ts)+".pt")
        #     self.model.state_dict()
        #     # self.model.save_state_dict(os.path.join(self.out_path,"epoch_"+str(self.epoch)+"_best_ts_"+str(ts)+".pt"))
        #     torch.save(self.model.state_dict(),save_filename)
        #     self.best_res = ts

        # with open(self.log_file, 'a') as f:
        #     elapsed_time = (datetime.datetime.now() - self.run_datetime).total_seconds()
        #     log = [self.epoch] + ['-1']*2 + [total_loss] + [ts] + [p_error] + [elapsed_time]
        #     log = map(str, log)
        #     logStr = '|'.join(log) + '\n'
        #     print("\n=======" + logStr + "=======")

        #     f.write(logStr)

        if self.epoch - self.best_res_epoch > 1000:
            print("Early Stop")
            sys.exit(0)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iter_idx = batch_idx + self.epoch * len(self.train_loader)
            # if (self.iteration != 0) and (iter_idx - 1) != self.iteration:
            #     continue
            self.iteration = iter_idx

            assert self.model.training

            data, target = data.to(device=self.device, dtype=torch.float), target.to(device=self.device,
                                                                                     dtype=torch.float)

            predict_value = self.model(data)
            print(predict_value)
            print(target)
            loss = self.criterion(predict_value, target)
            # print(loss)
            total_loss += loss.item()

            self.optim.zero_grad()
            loss.backward()
            # for param in self.model.parameters():
            # print(param.grad.data.sum())
            self.optim.step()

        print('Step {}, Loss: {:.4f}'
              .format(self.iteration + 1, total_loss))

        # log trian loss values

        info = {'train_loss': total_loss}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, self.iteration + 1)

        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, value.data.cpu().numpy(), self.epoch + 1)
            self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), self.epoch + 1)

            # if (self.iteration + 1) % 100 == 0:


                    # with open(self.log_file, 'a') as f:
                    #     elapsed_time = (datetime.datetime.now() - self.run_datetime).total_seconds()
                    #     log = [self.epoch, self.iteration] + [total_loss] + ['-1'] * 3 + [elapsed_time]
                    #     log = map(str, log)
                    #     logStr = '|'.join(log) + '\n'
                    #     print("\n======="+logStr+"=======")
                    #     f.write(logStr)

    def run(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_one_epoch()
            if self.epoch % self.val_interval == 0:
                self.validate_one_epoch()
