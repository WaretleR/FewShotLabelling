import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from motrackers.detectors import YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks
import sklearn
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import pickle
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from itertools import cycle
import collections
from collections import defaultdict
from facebookTimesformer.TimeSformer.models.vit import TimeSformer
from facebookTimesformer.TimeSformer.datasets.utils import tensor_normalize


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()


class Args:
    def __init__(self):
        self.dataPath = './prototypeTestInput/videos'
        self.labelsPath = './prototypeTestInput/labels'
        self.targetPath = './prototypeTestOutput'
        self.yoloPath = './mot/multi-object-tracker/examples/pretrained_models/yolo_weights'
        self.timesformerPath = './facebookTimesformer/weights/TimeSformer_divST_8x32_224_K600.pyth'
        self.selectedVideos = []
        self.classesNames = []
        self.frameFreq = 8
        self.targetX = 700
        self.targetY = 500
        self.premadeTracks = False
        self.premadeTracksPath = ''

    def processArgv(self):
        for i in range(len(sys.argv)):
            if sys.argv[i].startswith('-'):

                if sys.argv[i] == "-labels":
                    self.labelsPath = sys.argv[i + 1]

                if sys.argv[i] == "-data":
                    self.dataPath = sys.argv[i + 1]

                if sys.argv[i] == "-target":
                    self.targetPath = sys.argv[i + 1]

                if sys.argv[i] == "-premadeTracks":
                    self.premadeTracks = True
                    self.premadeTracksPath = sys.argv[i + 1]

                if sys.argv[i] == "-selected":
                    f = open(os.path.join(sys.argv[i + 1]))
                    self.selectedVideos = f.read().split('\n')
                    if self.selectedVideos[-1].isspace():
                        self.selectedVideos.pop()

                if sys.argv[i] == "-classesNames":
                    f = open(os.path.join(sys.argv[i + 1]))
                    self.classesNames = f.read().split('\n')
                    if self.classesNames[-1].isspace():
                        self.classesNames.pop()

                if sys.argv[i] == "-frameFreqMOT":
                    self.frameFreq = int(sys.argv[i + 1])

                i += 1


class EventInfo():
    def __init__(self, eventType, duration, startFrame, endFrame, framesDict):
        self.eventType = int(eventType)
        self.duration = int(duration)
        self.startFrame = int(startFrame)
        self.endFrame = int(endFrame)
        self.framesDict = framesDict
        self.leftTopX = sys.maxsize
        self.leftTopY = sys.maxsize
        self.rightBottomX = 0
        self.rightBottomY = 0


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class MOT:
    def __init__(self, args : Args):
        self.detectionModel = YOLOv3(
            weights_path=os.path.join(args.yoloPath, 'yolov3.weights'),
            configfile_path=os.path.join(args.yoloPath, 'yolov3.cfg'),
            labels_path=os.path.join(args.yoloPath, 'coco_names.json'),
            confidence_threshold=0.6,
            nms_threshold=0.2,
            draw_bboxes=True,
            use_gpu=True
        )
        self.targetX = args.targetX
        self.targetY = args.targetY

    def extractClips(self, args : Args):
        print("Extraction of clips started")
        tracksAll = {}

        for videoName in sorted(os.listdir(args.dataPath)):
            self.trackerModel = SORT(max_lost=16, tracker_output_format='mot_challenge', iou_threshold=0.1)

            if len(args.selectedVideos) == 0 or videoName in args.selectedVideos:
                tracksAll[videoName] = []
                video = cv2.VideoCapture(os.path.join(args.dataPath, videoName))
                print("Extracting frames from " + videoName)

                for i in range(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), args.frameFreq):
                    printProgressBar(i, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), prefix = 'MOT progress:')
                    video.set(cv2.CAP_PROP_POS_FRAMES, i)
                    success, image = video.read()
                    prevSize = image.shape
                    image = cv2.resize(image, (args.targetX, args.targetY))

                    bboxes, confidences, class_ids = self.detectionModel.detect(image)
                    human_ids = np.where(class_ids == 0)
                    tracks = self.trackerModel.update(bboxes[human_ids], confidences[human_ids], class_ids[human_ids])

                    for j in range(len(tracks)):
                        tracks[j] = list(tracks[j])
                        tracks[j][0] -= 1
                        tracks[j][2] = int(tracks[j][2] * prevSize[1] / self.targetX)
                        tracks[j][4] = int(tracks[j][4] * prevSize[1] / self.targetX)
                        tracks[j][3] = int(tracks[j][3] * prevSize[0] / self.targetY)
                        tracks[j][5] = int(tracks[j][5] * prevSize[0] / self.targetY)
                    tracksAll[videoName] += tracks

        eventsAll = {}

        for videoName in tracksAll:
            eventsAll[videoName] = defaultdict(list)

            for t in tracksAll[videoName]:
                eventsAll[videoName][t[1]].append([t[1], 0, 0, 0, 0, t[0] * args.frameFreq, t[2], t[3], t[4], t[5]])

        for videoName in eventsAll:

            for k in eventsAll[videoName]:
                frameStart = eventsAll[videoName][k][0][5]
                frameEnd = eventsAll[videoName][k][0][5]
                
                for i in range(len(eventsAll[videoName][k])):
                    frameStart = min(eventsAll[videoName][k][i][5], frameStart)
                    frameEnd = max(eventsAll[videoName][k][i][5], frameEnd)
                
                for i in range(len(eventsAll[videoName][k])):
                    eventsAll[videoName][k][i][2] = frameEnd - frameStart
                    eventsAll[videoName][k][i][3] = frameStart
                    eventsAll[videoName][k][i][4] = frameEnd

        os.mkdir(os.path.join(args.targetPath, 'labels'))
        for videoName in eventsAll:
            with open(os.path.join(args.targetPath, 'labels', '.'.join(videoName.split('.')[:-1]) + '.viratdata.events.txt'), 'w+') as f:
                for k in eventsAll[videoName]:
                    for e in eventsAll[videoName][k]:
                        f.write(' '.join([str(elem) for elem in e]) + '\n')

        print("Extraction of clips finished")


class Classificator:
    def __init__(self, args : Args, clf = make_pipeline(StandardScaler(), SVC(C = 0.1, kernel='linear', probability = True)), saveEmbeddings = False):
        self.model = TimeSformer(img_size=224, num_frames=8, attention_type='divided_space_time',  pretrained_model=args.timesformerPath)
        self.model.model.head = Identity()
        self.clf = clf
        self.embeddingsSize = 768
        self.framesPerVideo = 8
        self.dataMean = [0.45, 0.45, 0.45]
        self.dataStd = [0.225, 0.225, 0.225]
        self.labelsDict = {}
        self.classesNames = args.classesNames
        self.saveEmbeddings = saveEmbeddings
        self.batchSize = 1

    def findClosest(indexList, i):
        #print('Closest to ' + str(i) + ' is ' + str(min(indexList, key=lambda x:abs(x-i))))
        return min(indexList, key=lambda x:abs(x-i))

    def getLabels(self, dirPath):
        self.labelsDict = {}

        for labelName in sorted(os.listdir(dirPath)):
            print(labelName)
            videoName = labelName.split('.')[0]

            f = open(os.path.join(dirPath, labelName))
            labels = f.read().split('\n')[:-1]
            f.close()

            labels = [l.split(' ') for l in labels]
            labels = [[int(l1) for l1 in l if l1 != ''] for l in labels]

            for l in labels:
                eventName = videoName + '.' + str(l[0])
                if eventName not in self.labelsDict:
                    self.labelsDict[eventName] = EventInfo(l[1], l[2], l[3], l[4], dict())
                self.labelsDict[eventName].framesDict[l[5]] = [e for e in l[6:]]
            
    def printLabelsStats(self):
        print('Число событий: ' + str(len(self.labelsDict.keys())))

        self.classesStat = defaultdict(int)
        for k in self.labelsDict:
            self.classesStat[self.labelsDict[k].eventType] += 1
            
        if len(self.classesNames) == len(self.classesStat.keys()):
            for i in range(len(self.classesNames)):
                print(str(self.classesNames[i]) + " : " + str(self.classesStat[i+1]))
        else:
            for i in range(self.classesStat.keys()):
                print(str(i) + " : " + str(self.classesStat[i+1]))

    def getCroppedClips(self, dirPath):
        print('Reading clips from ' + dirPath + ' started')

        clips = []
        self.clipLabels = []
        self.eventsID = []

        for videoName in sorted(os.listdir(dirPath)):
            video = cv2.VideoCapture(os.path.join(dirPath, videoName))
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, image = video.read()
            
            events = {key:val for key, val in self.labelsDict.items() if key.split('.')[0] == videoName.split('.')[0]}
            self.eventsID += events.keys()
            
            for nIter, k in enumerate(events):
                printProgressBar(nIter, len(events.keys()), prefix = 'Clip reading progress:')

                clip = torch.zeros((self.framesPerVideo, 224, 224, 3), dtype = torch.uint8)
                frameNumbers = [round(events[k].startFrame + i * (events[k].duration - 1.0) / (self.framesPerVideo - 1)) for i in range(self.framesPerVideo)]
                frameNumbers = [Classificator.findClosest(events[k].framesDict.keys(), i) for i in frameNumbers]
                #print(frameNumbers)

                for i, n in enumerate(frameNumbers):
                    xmin, ymin, xlen, ylen = events[k].framesDict[n]  
                    #print(xmin, ymin, xlen, ylen)
                    
                    if xmin < events[k].leftTopX:
                        events[k].leftTopX = xmin
                    if ymin < events[k].leftTopY:
                        events[k].leftTopY = ymin
                    
                    if xmin + xlen > events[k].rightBottomX:
                        events[k].rightBottomX = xmin + xlen
                    if ymin + ylen > events[k].rightBottomY:
                        events[k].rightBottomY = ymin + ylen
                
                #print("(" + str(events[k].leftTopX) + ", " + str(events[k].leftTopY) + ") (" + str(events[k].rightBottomX) + ", " + str(events[k].rightBottomY) + ")")
                #print(str(events[k].duration) + " frames")
                
                for i, n in enumerate(frameNumbers):
                    
                    video.set(cv2.CAP_PROP_POS_FRAMES, n)
                    success, image = video.read()
                    if success:
                        bbox = image[max(events[k].leftTopY, 0) : events[k].rightBottomY, max(events[k].leftTopX, 0) : events[k].rightBottomX]
                        
                        clip[i] = torch.from_numpy(cv2.resize(bbox, (224, 224)))

                        '''if i == len(frameNumbers) - 1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            clipMedian = np.zeros((3, 224, 224), dtype = np.int32)
                            for f in clip:
                                clipMedian += f
                            clipMedian = clipMedian / len(frameNumbers)
                            clipMedian = np.swapaxes(np.swapaxes(clipMedian, 0, 2), 0, 1)
                            
                            ax.imshow(clipMedian.astype(int))
                            #rect = mpatches.Rectangle((labelInfo[k].leftTopX, labelInfo[k].leftTopY), 
                            #              labelInfo[k].rightBottomX - labelInfo[k].leftTopX, labelInfo[k].rightBottomY - labelInfo[k].leftTopY,
                            #              fill=False, edgecolor='red', linewidth=2)
                            #ax.add_patch(rect)
                            ax.set_axis_off()
                            plt.tight_layout()
                            plt.show()'''

                    else:
                        print('Error while reading frame %d in video %s' % (n, videoName))

                clips.append(clip)
                self.clipLabels.append(events[k].eventType)
        
        self.clipsTensor = torch.zeros((len(clips), 3, self.framesPerVideo, 224, 224))
        for i in range(len(clips)):
            self.clipsTensor[i] = np.transpose(tensor_normalize(torch.tensor(clips[i]), self.dataMean, self.dataStd), axes=[3, 0, 1, 2])
                
        print('Reading clips from ' + dirPath + 'finished')

    def train(self):
        print('Feature extraction started')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        print(torch.cuda.device_count())
        self.model.to(device)

        embeddings = np.zeros((len(self.clipLabels), self.embeddingsSize), dtype=np.float)

        indices = [i for i in range(len(self.clipLabels))]

        for batchNumber in range(len(self.clipLabels) // self.batchSize):
            printProgressBar(batchNumber, len(self.clipLabels) // self.batchSize, prefix='Inference progress:')
            #print(indices[batchNumber * batchSize : (batchNumber + 1) * batchSize])
            inputs = self.clipsTensor[indices[batchNumber * self.batchSize : (batchNumber + 1) * self.batchSize]]
            inputs = inputs.to(device)

            outputs = self.model(inputs)
            outputs = outputs.cpu().detach().numpy()
            embeddings[indices[batchNumber * self.batchSize : (batchNumber + 1) * self.batchSize]] = outputs

        if self.saveEmbeddings:
            embeddingsDict = {}
            for i in range(len(self.eventsID)):
                embeddingsDict[self.eventsID[i]] = embeddings[i]
            with open('embeddings_' + int(datetime.datetime.now().timestamp()) + '.pickle', 'wb') as handle:
                pickle.dump(embeddingsDict, handle)

        print('Feature extraction finished')

        print('Training on ' + str(len(self.clipLabels)) + ' events started')

        labelsByClasses = defaultdict(list)
        for i, l in enumerate(self.clipLabels):
            labelsByClasses[l].append(i)

        trainIndices = []
        for lbc in labelsByClasses:
            trainIndices = trainIndices + labelsByClasses[lbc]

        trainIndices = list(np.random.permutation(trainIndices))
            
        self.clf.fit(torch.tensor(embeddings)[trainIndices], torch.tensor(self.clipLabels)[trainIndices])
                
        print('Training on ' + str(len(self.clipLabels)) + 'events ended')

    def predict(self):
        print('Feature extraction started')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        print(torch.cuda.device_count())
        self.model.to(device)

        embeddings = np.zeros((len(self.clipLabels), self.embeddingsSize), dtype=np.float)

        indices = [i for i in range(len(self.clipLabels))]

        for batchNumber in range(len(self.clipLabels) // self.batchSize):
            printProgressBar(batchNumber, len(self.clipLabels) // self.batchSize, prefix='Inference progress:')
            #print(indices[batchNumber * batchSize : (batchNumber + 1) * batchSize])
            inputs = self.clipsTensor[indices[batchNumber * self.batchSize : (batchNumber + 1) * self.batchSize]]
            inputs = inputs.to(device)

            outputs = self.model(inputs)
            outputs = outputs.cpu().detach().numpy()
            embeddings[indices[batchNumber * self.batchSize : (batchNumber + 1) * self.batchSize]] = outputs

        if self.saveEmbeddings:
            embeddingsDict = {}
            for i in range(len(self.eventsID)):
                embeddingsDict[self.eventsID[i]] = embeddings[i]
            with open('embeddings_' + int(datetime.datetime.now().timestamp()) + '.pickle', 'wb') as handle:
                pickle.dump(embeddingsDict, handle)

        print('Feature extraction finished')

        print('Training on ' + str(len(self.clipLabels)) + ' events started')

        labelsByClasses = defaultdict(list)
        for i, l in enumerate(self.clipLabels):
            labelsByClasses[l].append(i)

        testIndices = []
        for lbc in labelsByClasses:
            testIndices = testIndices + labelsByClasses[lbc]

        self.preds = self.clf.predict(embeddings[testIndices])

    def updateLabelsWithPreds(self, targetPath):
        newLabels = []
        for labelName in sorted(os.listdir(targetPath)):
            videoName = labelName.split('.')[0]

            with open(os.path.join(targetPath, labelName), "r+") as f:
                for line in f:
                    t = [int(x) for x in line.split(' ') if x != '']
                    t[1] = self.preds[list(self.labelsDict.keys()).index(videoName + '.' + str(t[0]))]
                    newLabels.append(t)

                f.seek(0)
                for l in newLabels:
                    f.write(" ".join([str(x) for x in l]))
                    f.write("\n")
                f.close()


class Visualizer:
    def __init__(self, videosPath, labelsPath, targetPath, classesNames = []):
        self.videosPath = videosPath
        self.labelsPath = labelsPath
        self.targetPath = targetPath
        self.classesNames = classesNames

    def visualize(self):
        print("Visualization started")

        # Red color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2

        for i, labelName in enumerate(os.listdir(self.labelsPath)):
            printProgressBar(i, len(os.listdir(self.labelsPath)), prefix="Video writing progress:")

            #get labels for frames
            framesDict = defaultdict(list)

            f = open(os.path.join(self.labelsPath, labelName))
            labels = f.read().split('\n')[:-1]
            f.close()

            labels = [l.split(' ') for l in labels]
            labels = [[int(l1) for l1 in l if l1 != ''] for l in labels]

            prevID = -1
            prevFrameNumber = -1
            prevBbox = []
            for l in labels:
                if l[0] == prevID:
                    if l[5] - prevFrameNumber > 1:
                        for frameNumber in range(prevFrameNumber + 1, l[5]):
                            bbox = [x[0] + (x[1] - x[0]) * 1.0 * (frameNumber - prevFrameNumber) / (l[5] - prevFrameNumber) for x in zip(prevBbox, l[6:])]
                            framesDict[frameNumber].append(bbox + [l[1]])
                
                framesDict[l[5]].append(l[6:] + [l[1]])
                
                prevID = l[0]
                prevFrameNumber = l[5]
                prevBbox = l[6:]

            #write new video with labels
            videoName = labelName.split('.')[0]

            baseVideo = cv2.VideoCapture(os.path.join(self.videosPath, videoName + '.mp4'))
      
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            outVideo = cv2.VideoWriter(os.path.join(self.targetPath, videoName + '.avi'), fourcc, baseVideo.get(cv2.CAP_PROP_FPS), (int(baseVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(baseVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            count = 0

            while count < baseVideo.get(cv2.CAP_PROP_FRAME_COUNT):
                ret, frame = baseVideo.read()

                if ret:
                    for eventBbox in framesDict[count]:
                        xmin, ymin, xlen, ylen = eventBbox[:4]
                        label = eventBbox[4]

                        frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmin + xlen), int(ymin + ylen)), color, thickness)

                        if len(self.classesNames) > 0:
                            cv2.putText(frame, self.classesNames[label - 1], (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness) 
                        else:
                            cv2.putText(frame, str(label - 1), (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

                    outVideo.write(frame)
                count += 1

            baseVideo.release()
            outVideo.release()


def main():
    #Process command line args
    args = Args()
    args.processArgv()
    
    if not args.premadeTracks:
        #Extracting tracks
        mot = MOT(args)
        mot.extractClips(args)

    #Classification
    clf = Classificator(args)
    
    clf.getLabels(args.labelsPath)
    clf.getCroppedClips(args.dataPath)

    clf.train()
    
    if args.premadeTracks:
        clf.getLabels(args.premadeTracksPath) 
    else:
        clf.getLabels(os.path.join(args.targetPath, 'labels'))

    clf.getCroppedClips(args.dataPath)

    clf.predict()

    if args.premadeTracks:
        clf.updateLabelsWithPreds(args.premadeTracksPath)
    else:
        clf.updateLabelsWithPreds(os.path.join(args.targetPath, 'labels'))

    #Visualization
    os.mkdir(os.path.join(args.targetPath, 'videos'))

    if args.premadeTracks:
        visual = Visualizer(args.dataPath, args.premadeTracksPath, os.path.join(args.targetPath, 'videos'), args.classesNames)
    else:
        visual = Visualizer(args.dataPath, os.path.join(args.targetPath, 'labels'), os.path.join(args.targetPath, 'videos'), args.classesNames)
    visual.visualize()


if __name__ == "__main__":
    main()