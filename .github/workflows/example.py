import json
from statistics import mean
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import csv
import gspread
import math
from datetime import timedelta
from csv import reader
import sys
import re, urllib

alpha = 0.5

# Open the googleSheet that specifies difficulty of each sequence
sa = gspread.service_account('.github/workflows/service_account.json')
sh = sa.open("MilaSongsLevelScaling")
wk = sh.worksheet('TamTambouille/LianeFolie')
wk1 = wk.get_all_values()
wk = sh.worksheet('VegeBaston')
wk2 = wk.get_all_values()
durTamLiane = [wk1[d][29] for d in range(1, 190)]
durVege = [wk1[d][29] for d in range(1, 190)]


def bpmadjust(bpm):
    return 80 + int(bpm / 80)


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


# difficulty parameters =  difficulty, wordDisplayTime ,wordLevel, readingListIndex, wordsCount, intrudersCount, wordfontchange ,bpm,
# clouds, presence_barre,sequenceIndex/levelIndex, song duration


# mini-games tolerances
tolerance = {12: 0.43, 15: 0.26, 2: 0.2, 4: 0.2, 11: 0.2, 14: 0.2, 16: 0.2, 13: 0.2}

# correspondence gameId/OrderinLists
gameId_order = {12: 0, 15: 1, 2: 2, 4: 3, 11: 4, 14: 5, 16: 6, 13: 7}
order = [12, 15, 2, 4, 11, 14, 16, 13]

gameNames = ["VegeBaston", "PotesAuFeu", "Synglab", "Riversplash", "LianeFolie", "TamTambouille", "CledesChants",
             "Somnbulles"]

# load patientsdata

#f = open('Datas.json')
#data = json.load(f)

# load sequences and songs
sequences = []
for i in range(8):
    y = open('.github/workflows/'+gameNames[i] + "Sequences.json")
    y = json.load(y)
    sequences += [y['sequences']]

songs = json.load(open('.github/workflows/MilaSongs.json'))

# penalty of error
err_thresh = 0.4;

# number of levels to overlook before starting the DDA algorithm
skips = 3

# Contribution of each mini-game to the first 5 parameters of the profile
weights = [[0, 0.25, 1, 0, 1],
           [0, 0, 1, 0, 1],
           [0, 0.15, 0, 0, 0],
           [0, 0.25, 1, 1, 0],
           [0, 0.25, 0, 1, 1],
           [0, 0.25, 0, 1, 0],
           [0, 0.25, 0, 1, 1],
           [1, 0, 0, 1, 0]]


# weights of mini games are not constant; it depends on whether visual disruptors exist for example
def modifyweights(d, gameId):
    if (gameId == 16):
        if (number_of_clouds(d) > 0):
            weights[gameId_order[gameId]][1] = 1
        else:
            weights[gameId_order[gameId]][1] = 0.5
    if (gameId == 11):
        if (number_of_clouds(d) > 0):
            weights[gameId_order[gameId]][0] = 1
        else:
            weights[gameId_order[gameId]][0] = 0.5
    if (gameId == 15):
        if (pres_bar(d)):
            weights[gameId_order[gameId]][0] = 1
            weights[gameId_order[gameId]][1] = 1
        if (not pres_bar(d)):
            weights[gameId_order[gameId]][0] = 0
            weights[gameId_order[gameId]][1] = 0.5


# weight of games played at the beginning of 20mins and at the end
min_consideration = 0.60
# the first and last 20% are not considered as much the 60% of the middle
barriers = 0.2

# time played on each minigame
durations = [0] * 8
# advancement = sum( score * difficulty ) for each minigame
advancements = [0] * 8
# total advancement in the game
total = 0
# does the help bar exist?
pre_barre = 0

# columns in the produced dataset include : profile + score + gameparametres
dataframenames = ["Memoire de travail", "Synchronisation au tempo", "Coordination auditive, visuelle et praxique",
                  "Voie Lexicale", "Endurance", "inlevelDuration", "TotalDuration", "Advancement", "TotalAdvancement",
                  "Score", "PlayerId", "GameId",
                  "difficulty", "wordDisplayTime", "wordLevel", "readingListIndex", "wordsCount",
                  "intrudersCount", "wordfontchange",
                  "bpm", "clouds", "presence_barre", "sequenceIndex", "duree", "song", "levelDuration"]


# Parametres Som'n'bulles:
# wordDisplayTime
# wordLevel
# readingListIndex
# wordsCount
# intrudersCount

# find bpm from the song name
def bpm_find(s):
    return int(s[s.index('_', -5) + 1:])


def bpm_modifier(bpm):
    if (bpm < 80):
        return 0.85
    elif (bpm < 90):
        return 0.9
    elif (bpm < 98):
        return 0.95
    elif (bpm > 131):
        return 0.7
    elif (bpm > 121):
        return 0.75
    elif (bpm > 115):
        return 0.8
    elif (bpm > 108):
        return 0.95
    return 1;


# maximum possible difficutly
player_max_level_index = 199
gamelengths = [103, 71, 98, 101, 76, 92, 113, 69]


# choose the game index with regards to difficulty
def sequenceIndexer(bpm, gameId, levelIndex):
    return clamp(
        int(bpm_modifier(bpm) * float(len(sequences[gameId_order[gameId]]) - 1) * levelIndex / player_max_level_index),
        0, len(sequences[gameId_order[gameId]]) - 1)


# duration of a mini-game is a function of it's bpm and the song chosen
def dur(bpm, sequenceIndex):
    return durTamLiane[sequenceIndex] * bpm / 100.0


def modifiedtext(levelIndex):
    return int(levelIndex / 25)


def LianeFolie(difficulty, song, gameId):
    bpm = bpm_find(song)
    sequenceIndex = sequenceIndexer(bpm, gameId, difficulty)
    duration = durTamLiane[sequenceIndex]
    clouds = number_of_clouds(difficulty)
    return [-1, -1, -1, -1, -1, -1, bpm, clouds, -1, sequenceIndex, duration]


def TamTambouille(difficulty, song, gameId):
    bpm = bpm_find(song)
    sequenceIndex = sequenceIndexer(bpm, gameId, difficulty)
    duration = durTamLiane[sequenceIndex]
    return [-1, -1, -1, -1, -1, -1, bpm, -1, -1, sequenceIndex, duration]


def VegeBaston(difficulty, song, gameId):
    timeline = int(timeline_veg(difficulty))
    bpm = bpm_find(song)
    sequenceIndex = sequenceIndexer(bpm, gameId, difficulty)
    return [-1, -1, -1, -1, -1, -1, bpm, -1, timeline, sequenceIndex, -1]


def CledesChants(difficulty, song, gameId):
    bpm = bpm_find(song)
    clouds = number_of_clouds(difficulty)
    return [-1, -1, -1, -1, -1, -1, bpm, clouds, -1, -1, -1]


def pres_bar(difficulty):
    return 100.0 * (float(difficulty) / player_max_level_index) >= 40


def timeline_veg(difficulty):
    return difficulty < 30


def PotesAuFeu(difficulty, song, gameId):
    bpm = bpm_find(song)
    sequenceIndex = sequenceIndexer(bpm, gameId, difficulty)
    timeLineHidden = int(pres_bar(difficulty))
    duration = durTamLiane[sequenceIndex]
    return [-1, -1, -1, -1, -1, -1, bpm, -1, timeLineHidden, sequenceIndex, duration]


def Riversplash(difficulty, song, gameId):
    bpm = bpm_find(song)
    clouds = number_of_clouds(difficulty)
    return [-1, -1, -1, -1, -1, -1, bpm, clouds, -1, -1, -1]


def Synglab(difficulty, song, gameId):
    return [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


difficultyModifierLength = 3


def Somnbulles(difficulty, song, gameId):
    parameters = pd.read_csv("ParametersSomnBulles.csv")
    bpm = bpm_find(song)
    sequenceIndex = int(round(float(difficulty) * len(parameters['Column1']) / player_max_level_index));
    sequenceIndex = clamp(sequenceIndex, 0, len(parameters['Column1']))

    return [parameters['Column' + str(i + 1)][sequenceIndex - 1] for i in range(5)] + [modifiedtext(difficulty), bpm,
                                                                                       -1, -1, sequenceIndex, -1]


def number_of_clouds(d):
    return int((d - 21) / 20)


def transformdate(d):
    return datetime.datetime(int(d[0:4]), int(d[5:7]), int(d[8:10]), int(d[11:13]), int(d[14:16]), int(d[17:19]))


# check if two dates happen in the same day
def compareday(d1, d2):
    return d1.day == d2.day and d1.month == d2.month and d1.year == d2.year


# initial dataset ( contains patients )
df = []


def difficulty_to_parametres(difficulty, song, gameId=12):
    ret = eval(gameNames[gameId_order[gameId]] + "(difficulty,song,gameId)")
    ret = [difficulty] + [float(x) for x in ret]
    return ret + [song]


def weightofscoreinday(chunk, n):
    # if float(chunk+1) / n <= barriers or float(chunk+1) / n >= (1 - barriers):
    #    return min_consideration
    return 1


start = 0.2


# Core Function : take a player profile, the history of his games and gives back a new profile
def updateprofile(gameId, player):
    # history contains scores of previous levels, it's sliced by day
    history = player[1]
    profile = player[0]

    for i in range(5):
        profile[i] = 0
    divisor = [0] * len(weights[0])
    dayselapsed = 0

    for day in range(len(history)):
        for chunk in range(len(history[day])):
            dayselapsed += 1
            w = (start + (1 - start) * ((day + 1) / len(history)) ** 0.5) * weightofscoreinday(chunk, len(history[day]))
            val = (w * history[day][chunk][1])
            modifyweights(history[day][chunk][1], history[day][chunk][2])
            for weight in range(len(weights[0])):
                divisor[weight] += w * weights[gameId_order[history[day][chunk][2]]][weight]
                profile[weight] += val * weights[gameId_order[history[day][chunk][2]]][weight]
    for weight in range(len(weights[0])):
        if profile[weight] != 0:
            profile[weight] /= divisor[weight]
            profile[weight] *= float(dayselapsed) / (player_max_level_index*8)
    return [profile, history]


# Core Function: take a state of a player, add it to the cloud
def addtocloud(difficulty, profile, score, playerId, gameId, levelDuration, song=""):
    d = difficulty_to_parametres(difficulty, song, gameId)
    df.append(profile + [score, playerId, gameId] + d + [levelDuration])


# calculate a score for a game played, provides a much more continious calcualtion of difficulty
lowscores = [1] * 8


def score_player(playerId):
    y = []
    for roun in data:
        if (roun['inputs'] != None and len(roun['inputs']) != 0):
            if (str(type(roun['inputs'])) == "<class 'str'>"):
                roun['inputs'] = roun['inputs'].replace("true", "True")
                roun['inputs'] = roun['inputs'].replace("false", "False")
                roun['inputs'] = eval(roun['inputs'])
            if str(type(roun['inputs'])) == "<class 'dict'>":
                continue
            if (roun['playerId'] is not None and roun['playerId'] == playerId and roun['inputs'] is not None and len(
                    roun['inputs']) > 3 and roun['gameId'] in order):
                y += [roun]

    difficulty = [h['difficulty'] for h in y]
    tsCreate = [transformdate(h['tsCreate']) for h in y]
    stepIndex = [h['stepIndex'] for h in y]
    adventureName = [h['adventureName'] for h in y]
    chapterIndex = [h['chapterIndex'] for h in y]
    gameDuration = [h['gameDuration'] for h in y]
    perfomance_score = [score_chunk(h) for h in y]
    levelName = [h['levelName'] for h in y]
    gameId = [h['gameId'] for h in y]
    zipped = zip(tsCreate, stepIndex, adventureName, chapterIndex, difficulty, perfomance_score, gameDuration, gameId,
                 levelName)
    zipped = list(zipped)
    once = {}
    once_adv = []

    if (len(zipped) > 0):
        # print(playerId)
        zipped = sorted(zipped, key=lambda x: x[0])
        for x in zipped:
            if (x[2] != None and not ((x[1], x[2], x[3]) in once)):
                once[(x[1], x[2], x[3])] = 1
                once_adv += [x]
            if (x[2] == None):
                once_adv += [x]

        player = [[0] * (len(weights[0]) + 4), []]
        j =0
        for d in zipped:

            if (len(player[1]) < 1):
                player[1] = [[[d[0], d[5], d[7], d[4]]]]
            else:
                if (compareday(player[1][-1][0][0], d[0])):
                    player[1][-1] += [[d[0], d[5], d[7], d[4]]]
                else:
                    player[1].append([[d[0], d[5], d[7], d[4]]])

            player[0][5] = durations[gameId_order[d[7]]]
            player[0][6] = sum(durations)
            player[0][7] = advancements[gameId_order[d[7]]]
            player[0][8] = sum(advancements)
            if (j > skips):
                addtocloud(d[4], player[0], d[5], playerId, d[7], d[6],
                           d[8])
            j+=1
            advancements[gameId_order[d[7]]] += 1
            durations[gameId_order[d[7]]] += d[6]

            player = updateprofile(d[7], player)


# returns a score of a round
def score_chunk(round):
    y = []
    ts_ref = []
    ts = []
    score = 0
    error = 0

    if (round['gameId'] == 13):
        for h in round['inputs']:
            score += (len(h['correctWords']) / len(h['playerInputs']))
            error += (len(h['playerInputs']) - len(h['correctWords']))
        score /= len(round['inputs'])
    else:
        for t in round['inputs']:
            if ('ts_ref' in t):
                ts_ref += [t['ts_ref']]
            if ('ts' in t):
                ts += [t['ts']]

        ts_ref = list(set(ts_ref))
        if (len(ts_ref) == 0):
            return 0
        ts_ref.sort()
        ts_ref_scored = [False] * len(ts_ref)

        for x in ts:
            i = find_place(ts_ref, x)
            s = scoring(ts_ref, round['gameId'], i, x)
            if (s > 0):
                if (ts_ref_scored[i] == False):
                    score += s
                    ts_ref_scored[i] = True
            else:
                error += 1
                score += s
        score /= (error + len(ts_ref))

    score = clamp(score, 0, 1)
    return score


f.close()


# bell shaped function used for calculating scores
def scoring(ts_ref, id, k, x):
    m = abs(x - ts_ref[k])
    if (m <= tolerance[id]):
        return (1 - (m / tolerance[id]) ** 1.5)
    return -err_thresh


# find the optimal placement of a input between instructions
def find_place(ts_ref, x):
    j = 0
    mini = 10000
    for i in range(len(ts_ref)):
        if (abs(x - ts_ref[i]) < mini):
            j = i
            mini = abs(x - ts_ref[i])
    return (j)


# curve used for assigning difficulties gives a number between 0 and 1
def difficultycurve(t):
    return (-117 + 218 * math.cos(0.2 * t - math.floor(0.2 * t))) / 100.0


# give the next difficulty parametres
def nextdifficulty(profile, playedlevels, gameId, error=0):
    difficulty = difficultycurve(playedlevels)
    return (next(profile, difficulty, gameId, error))


# error between desired score and actual score
def result(difficulty, result):
    error = difficulty - result
    return (error)


# read a dataset as list of lists
def readaslist(name):
    # read csv file as a list of lists
    with open('.github/workflows/'+name + '.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        ndf = list(csv_reader)
        ndf = ndf[1:]
        # ndf = [{dataframenames[i]:a[i] for i in range(len(dataframenames))} for a in ndf]
        return ndf

avgscore=8*[0]
maxis = 8 * [0]
minis = 8 * [10000]
gamelengths=8*[0]
# nromalize a dataset using min-max normalization
def normalizing(y):
    min_max = [[min([float(y[i][u]) for i in range(len(y))]), max([float(y[i][u]) for i in range(len(y))])] for u in
               range(10)]

    global minis,maxis,avgscore,gamelengths

    for i in range(len(y)):
        maxis[gameId_order[int(y[i][11])]] = max( [maxis[gameId_order[int(y[i][11])]], float(y[i][9])])
        minis[gameId_order[int(y[i][11])]] = min([minis[gameId_order[int(y[i][11])]], float(y[i][9])])
        avgscore[gameId_order[int(y[i][11])]]+=float(y[i][9])
        gamelengths[gameId_order[int(y[i][11])]]+=1
    avgscore=[(avgscore[i]/float(gamelengths[i]) -minis[i])/(maxis[i]-minis[i])for i in range(8)]
    print("minis ",minis)
    print("maxis ",maxis)
    print("avgs ",avgscore)
    for i in range(8):
        print([float(y[k][5]) if (int(y[k][11]) == i) else 1 for k in range(len(y))])


    for i in range(len(y)):
        for a in range(9):
            if (a != 7):
                y[i][a] = clamp((float(y[i][a]) - min_max[a][0]) / (float(min_max[a][1]) - min_max[a][0]), 0, 1)
            else:
                y[i][a] = float(y[i][a]) / float(maxis[gameId_order[float(y[i][11])]])
        y[i][9] = clamp((float(y[i][9]) - minis[gameId_order[int(y[i][11])]]) / (float(maxis[gameId_order[int(y[i][11])]]) -minis[gameId_order[int(y[i][11])]]), 0, 1)
    return y


# save a dataset as a csv
def copytocsv(y, name):
    # print(y[0])

    with open(name + ".csv", "w", newline="") as g:
        writer = csv.writer(g)
        writer.writerow(dataframenames)
        writer.writerows(y)


# create the dataset from the patients inputs
def createbasecloud():
    print("reading Patients data")
    u = []
    for roun in data:
        u += [roun['playerId']]
    u = list(set(u))
    j = 0
    for h in u:
      #  if(int(h)!=55):
        j += 1
        print("Patient ", j)
        for i in range(8):
            durations[i] = 0
        for i in range(8):
            advancements[i] = 0
        score_player(h)

    copytocsv(df, "Unnormalized")
    copytocsv(normalizing(df), "Patients")


# used once
#createbasecloud()

# read 2 copies of the patients dataset ( normalized)
# read an unnormalized version of the patients dataset
unnormalized = pd.read_csv('.github/workflows/Unnormalized.csv')
patients = pd.read_csv('.github/workflows/Patients.csv')
# read the patients' data + the synthetic ones


# simulator of score
x = ["2021-10-14 13:13:44.525556+00", "2021-10-14 14:13:44.525556+00", "2021-10-14 15:13:44.525556+00",
     "2021-10-14 16:13:44.525556+00", "2021-10-14 17:13:44.525556+00"]
x = [transformdate(i) for i in x]


def tester(difficulty, n):
    return [clamp(np.random.randint(low=-10, high=10) + difficulty * 100, 0, 100) / 100.0,
            x[n % len(x)] + timedelta(days=n / len(x))]


# used to convert pandas type of series to a list
def tolisting(a):
    return [u for u in a]


# lenght of a normal level type
mak = 15

typediff=2
oldtype=0
lastdiff=0
def difficultycurve2(levels,gameId):
    global typediff
    global mak
    if (levels == mak):
        typediff = 3
        return avgscore[gameId_order[gameId]]/4.0
    if (levels == mak + 1):
        typediff = 0

        return 1
    if (mak +3 >levels > mak + 1):
        typediff = 1
        if(levels == mak+2):
            mak += 15
        return (2+avgscore[gameId_order[gameId]])/3.0
    #if (levels == mak + 3):

        #typediff = 2
        #return (1.0+3*avgscore[gameId_order[gameId]])/4.0
    typediff = 2
    return avgscore[gameId_order[gameId]]


# x1 , x2 contain the profiles as well as the wanted difficultly
# distance function between two profiles
def distance(x1, x2, w):
    return (sum([w[u] * abs(a - float(b)) ** 2 if u < len(w)-1 else w[u]*abs(a-float(b)) for u, a, b in zip(range(len(w)), x1, x2)]))**0.5
lastdiff = 0
oldtype = 0
typediff = 2
lastdiffs = [0,0,0,15]

def selectvalid(use):
    if(use):
        global lastdiffs
        if(typediff==3):
            return[lastdiffs[2]+10 ,lastdiffs[2]+20]
        if(typediff==0):
            return[lastdiffs[2]-20,lastdiffs[2]]
        if(typediff==1):
            return [lastdiffs[0]-10,lastdiffs[2]]

        return [lastdiffs[2]-5,lastdiffs[3]+10]
    return [0,player_max_level_index]
# Core Function, sklearn library doesn't allow for a customisable distance function, a basic implementation is down
# the fucntion returns the k closest neighbours in the dataset
def knn(x, gameId, k=1, wei=[],use = True):
    print("weight for score ", mean(weights[gameId_order[gameId]])/5.0)
    wei = [0, 0, 0, 0, mean(weights[gameId_order[gameId]])/5.0]

    dist = []
    vect = []
    global oldtype,typediff
    we = weights[gameId_order[gameId]] + wei
    # we = [0]*5  + oweight
    scores = []
    diffs = []
    adv = []
    pros = []
    playerids =[]
    # and ((typediff-oldtype)*(z['difficulty']-lastdiff)>=0
    for ind, z in pdf.iterrows():


        if z['GameId'] == gameId and distance(z[0:10], x[0:10], we) > 1e-9 and selectvalid(use)[0]<z['difficulty']<selectvalid(use)[1]:
            # print(typediff,oldtype,z['difficulty'],lastdiff)
            dist += [distance(z[0:10], x[0:10], we)]
            #scores += [z['Score']]
            #diffs += [z['difficulty']]
            #adv += [z['Advancement']]
            #pros+=[[tt for tt in z[0:5]]]
            #playerids += [ z['PlayerId']]
            vect += [z]
    #coupled = sorted(zip(dist, vect, diffs, scores, adv,playerids,pros), key=lambda s: s[0])
    coupled = sorted(zip(dist, vect), key=lambda s: s[0])
    #print("distances")
    #print(sorted(dist)[0:k])
    #print([[c[2],c[6], c[3], c[4],c[5]] for c in coupled[0:k]])
    return coupled[0:k]


k = 8


# Core Function: Given a certain profile, a desired score and a game type, this function will return the weighted
# average of k closest neighbours

def next(profile, difficulty, gameId, error,use=True):
    #print("k = ", k)
    u = knn(profile + [difficulty + alpha * error], gameId, k,use)

    par = [0] * len(u[0][1][12:24])
    s = sum([1 / (h[0]) for h in u])
    for lev in u:
        for v in range(len(u[0][1][12:24])):
            par[v] += 1 / (lev[0]) * lev[1][12 + v] / s
    return par + [u[0][1][-2], u[0][1][-1]]


# number of neighbors used for SMOTE
K_smote = 5
# number of Synthetic samples

pdf = None


# create additional data
def SMOTE(SyntSamples=15000):

    global pdf,avgscore,minis,maxis
    maxis= [0]*8
    minis = [1000]*8
    avgscore=[0]*8
    pdf = pd.read_csv('.github/workflows/Patients.csv')
    # rread = readaslist("Unnormalized")
    rread = readaslist("Patients")

    #print("Creating Additional Synthetic data")
   # print("Starting Smote")
    if(SyntSamples==0 and os.path.exists('NSynthetic.csv')):
        rread = readaslist('Synthetic')
        copytocsv(rread, "Synthetic")
        copytocsv(normalizing(rread), "NSynthetic")
        return 0
    for j in range(SyntSamples):
        print("Generating ", j)
        a = np.random.randint(low=0, high=len(rread))
        print(a)
        gameId = int(rread[a][11])
        u = knn([float(x) for x in rread[a][0:10]], gameId, K_smote,[],False)
        n = np.random.randint(0, K_smote)
        s = float(np.random.randint(0, 1000)) / 1000.0
        chosen = tolisting(u[n][1])
        print("the chosen ",chosen)
        new = [s * (chosen[i] - float(rread[a][i])) + float(rread[a][i]) for i in range(10)]
        new += [np.random.randint(100, 200)]

        new += [int(chosen[11])]
        new += [s * (chosen[i] - float(rread[a][i])) + float(rread[a][i]) for i in range(12, 24)]
        new += chosen[24:25]
        new += [s * (chosen[25] - float(rread[a][25])) + float(rread[a][25])]
        new[20] = round(new[20])
        new[21] = round(new[21])
        print("new guy ",new)
        rread += [new]
    copytocsv(rread, "Synthetic")
    copytocsv(normalizing(rread), "NSynthetic")


SMOTE(5000)


# print(next([0.819009, 0.822712, 0.875080,
#             0.814470,
#             0.875080,
#             0.437959,
#             0.976819],
#            0.8, 11))

# simulate a gameplay_test


def play_test():
    global pdf
    global lastdiff
    global oldtype
    global lastdiffs
    global typediff
    pdf = pd.read_csv('.github/workflows/NSynthetic.csv')
    print("Virtual Player Testing .. ")
    gameId = 15
    for gameId in order:
        lastdiff = 0
        oldtype = 0
        typediff = 2
        lastdiffs = [0, 0, 0, 15]

        for i in range(1):
            global mak
            mak = 15
            # oweights= [np.random.randint(1000)/250.0 for j in range(5)]
            test_player = [[0] * (len(weights[0]) + 4), []]
            test_player_id = 321

            levels = 0
            error = 0
            scores = []
            diffs = []
            difficulties = []

            n = 0
            for i in range(8):
                durations[i] = 0
                advancements[i] = 0
            for i in range(200):
                print("level ", i)
                oldtype = typediff
                difficulty = difficultycurve2(levels,gameId)

                print("Imposing a  difficulty of ", difficulty, "corrected ", difficulty + alpha * error)
                nexty = next(test_player[0], difficulty, gameId, error)

                s = tester(difficulty, n)
                n += 1
                print("I suggest ")
                print([str(a) + " : " + str(b) if isinstance(b, str) or b >= 0 else "" for a, b in
                       zip(dataframenames[12:], nexty)])
                score = s[0]
                difficulties.append(difficulty)
                scores.append(score)
                diffs.append(nexty[0] / 140.0)
                lastdiff = nexty[0]
                lastdiffs[typediff]=(lastdiffs[typediff]+3*nexty[0])/4.0
                print(lastdiffs)
                # print("you scored ",score)
                # addtocloud(nexty[0], test_player[0], score, test_player_id, gameId, nexty[-1], nexty[-2])
                advancements[gameId_order[gameId]] += 1
                durations[gameId_order[gameId]] += nexty[-1]
                test_player[0][5] = durations[gameId_order[gameId]]
                test_player[0][6] = sum(durations)
                test_player[0][7] = advancements[gameId_order[gameId]]
                test_player[0][8] = sum(advancements)
                levels += 1
                if (len(test_player[1]) < 1):
                    test_player[1] = [[[s[1], s[0], gameId]]]
                else:
                    if (compareday(test_player[1][-1][0][0], s[1])):
                        test_player[1][-1] += [[s[1], s[0], gameId]]
                    else:
                        test_player[1].append([[s[1], s[0], gameId]])

                error = (levels * error + difficulty - score) / (levels + 1)
                # print("err ",error)
                test_player = updateprofile(gameId, test_player)
                for a, u in zip(range(9), dataframenames[0:9]):
                    test_player[0][a] = clamp(
                        (test_player[0][a] - unnormalized[u].min()) / (unnormalized[u].max() - unnormalized[u].min()), 0, 1)
                print("new profile : ", test_player[0])
            plt.figure("Score Curve",figsize=(8,6))
            #plt.plot(scores)
            #plt.plot(difficulties)
            #plt.plot(diffs)
            plt.plot([x * player_max_level_index for x in diffs])

            plt.show()
            # plt.close()
            #plt.figure("LevelIndex")
            x=plt.figure("Game",figsize=(20,12))
            plt.plot(scores)
            plt.plot(difficulties)
            plt.plot(diffs)
            plt.savefig("Game " + gameNames[gameId_order[gameId]]+ ".png")
            plt.close()


play_test()
