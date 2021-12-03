"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import time
import pdb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--results_dir', type=str, default='./results', help='store classification results')
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    # List of all files in modelnet40_test.txt
    files=[]
    with open("./data/modelnet40_normal_resampled/modelnet40_test.txt","r") as wf:
        files.append(wf.read())

    # wf.read reads all filenames as a single string seperated by \n
    files = files[0].split('\n')
    
    carValue=[]
    carPercentages=[]
    finalTarget=[]
    finalPredChoice=[]
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        # print(f'Vote pool: {vote_pool}')
        pred = vote_pool / vote_num
        copy = pred.cpu()
        # pdb.set_trace()
        # print(f'Pred tensor: {copy.numpy()}')
        # print(f'Pred[0] tensor value: {sum(copy.numpy()[0])}')
        # print(f'Pred size: {pred.size()[0]}')
        # print(f'Class 7 value(car category value): {copy.numpy()[0][7]}')
        for i in range(pred.size()[0]):
          # print(f'Percentage of car category: {copy.numpy()[i][7]/sum(copy.numpy()[i])}')
          carPercentages.append((copy.numpy()[i][7]/sum(copy.numpy()[i]))*100)
          carValue.append(copy.numpy()[i][7])

        pred_choice = pred.data.max(1)[1]
        # print(f"Outputs: {j}\n")
        # print(f'Target: {target}, {target.cpu().numpy()[-1]}')
        # print(f'Pred_choice: {pred_choice}, {pred_choice.cpu().numpy()[-1]}')
        finalTarget.append(target.cpu().numpy())
        finalPredChoice.append(pred_choice.cpu().numpy())
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            print(f'Class accuracy: {classacc}')
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        # print(f'Mean correct: {mean_correct}')
    
    #print(finalTarget.flatten())
    finalTargetFlatList = [item for sublist in finalTarget for item in sublist]
    finalPredChoiceFlatList = [item for sublist in finalPredChoice for item in sublist]
    
    #print(finalPredChoice.flatten())
    # Writing outputs to the file
    class_file = os.path.join(args.results_dir, 'classification_results.txt')
    # class_file = "/content/drive/MyDrive/pointnetResults/classification_results.txt"
    cf = open(class_file, "w")
    try:
      onlyCarFileName = os.path.join(args.results_dir, 'OnlyCarClassified.txt')
      # onlyCarFileName = "/content/drive/MyDrive/pointnetResults/OnlyCarClassified.txt"
      carFile = open(onlyCarFileName, "w")
      output_dir = os.path.join(args.results_dir, 'Output.txt')
      with open(output_dir,"w") as wf:
        print("Writing to output.txt")
        for i, vals in enumerate(zip(finalTargetFlatList, finalPredChoiceFlatList)):
            isCar = True
            # vals: (target, pred_choice)
            # file_name, target, prediction
            # if(carPercentages[i]>0.045):
            # 0.0195 threshold is chosen after the classification result analysis
            if(carPercentages[i] > 0.0195):
              isCar = False
            if(isCar):
              wf.write(files[i]+","+str(vals[0])+","+str(vals[1])+","+str(carValue[i])+","+str(carPercentages[i])+',car\n')
              carFile.write(files[i]+","+str(vals[0])+","+str(vals[1])+","+str(carValue[i])+","+str(carPercentages[i])+',car\n')
              cf.write(files[i]+','+'1\n')
            else:
              wf.write(files[i]+","+str(vals[0])+","+str(vals[1])+","+str(carValue[i])+","+str(carPercentages[i])+',not car\n')
              cf.write(files[i]+','+'0\n')
            # if(vals[1]==7):
            #   carFile.write(files[i]+","+str(vals[0])+","+str(vals[1])+","+str(carValue[i])+","+str(carPercentages[i])+'\n')
    except Exception as e:
      print(e)
    carFile.close() 
    cf.close() 
   
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    print(f'Instance accuracy: {instance_acc}')
    print(f'Class accuracy: {class_acc}')
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
