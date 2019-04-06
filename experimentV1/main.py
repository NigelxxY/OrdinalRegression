import argparse
import datetime
import os
import pickle

import torch
import torch.utils.data as data
from trainer import SliceTrainer

from experimentV1.dataloader import gribDataSet4NormalSlice, gribDataSampler4NormalSlice, collate_fn

here = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", default="/mnt/pami/DATASET/METEOROLOGY/ECforecast/GRIB", type=str)
    parser.add_argument("--micaps_path", default="/mnt/pami/DATASET/METEOROLOGY/GT_micaps", type=str)
    parser.add_argument("--save_root", default="/mnt/pami/xyxu/dataset")
    parser.add_argument("--out", default=here)

    parser.add_argument("--start_date", default="20160801", type=str)
    parser.add_argument("--end_date", default="20160901", type=str)
    parser.add_argument("--run_flag", default="201608")

    parser.add_argument("--lr", default=1e-1, type=float)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--gpuid", default="2", type=str)
    parser.add_argument("--interval", default=5)
    parser.add_argument("--batch_size",default=8)

    args = parser.parse_args()

    train_pickle = os.path.join(args.save_root, "raw_" + args.run_flag + ".dat")
    test_pickle = os.path.join(args.save_root, "raw_test_" + args.run_flag + ".dat")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_datetime = datetime.datetime.now()
    args.out = os.path.join(args.out, run_datetime.strftime('%Y%m%d_%H%M%S.%f'), args.run_flag)

    print("========== Runing Result ========== ", args.out)

    if os.path.exists(train_pickle):
        with open(train_pickle, "rb") as f:
            trainDataSet = pickle.load(f)
    else:
        trainDataSet = gribDataSet4NormalSlice(args, train=True)
        with open(train_pickle, "wb") as f:
            pickle.dump(trainDataSet, f)

    if os.path.exists(test_pickle):
        with open(test_pickle, "rb") as f:
            testDataSet = pickle.load(f)
    else:
        testDataSet = gribDataSet4NormalSlice(args, train=False)
        with open(test_pickle, "wb") as f:
            pickle.dump(testDataSet, f)

    # print(len(trainDataSet))
    # test_list = [idx for idx in trainDataSet]
    # print(test_list)
    # raise NotImplementedError
    train_sampler = gribDataSampler4NormalSlice(trainDataSet)

    trainDataLoader = data.DataLoader(trainDataSet,
                                      batch_size=8,
                                      sampler=train_sampler,
                                      collate_fn=collate_fn)
    # print(len(trainDataLoader))
    # raise NotImplementedError
    print("========== Train Data Loading Done ==========")
    testDataLoader = data.DataLoader(testDataSet, batch_size=1, shuffle=True, num_workers=0)
    print("==========  Test Data Loading Done ==========")

    trainer = SliceTrainer(args,
                           device,
                           trainDataLoader,
                           testDataLoader,
                           args.out,
                           args.epochs,
                           args.interval)
    trainer.run()


if __name__ == "__main__":
    main()
