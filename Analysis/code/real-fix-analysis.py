import numpy as np
from util import *


def semanticAmbiguity(csvFile):
    with open(csvFile, 'r') as csvf:
        reader = csv.DictReader(csvf, delimiter='\t')
        with open("similarHeldout2Train_similar_same.csv", 'w') as csvw:
            writer = csv.DictWriter(csvw, fieldnames=['bug', 'sim-bug', 'patch', 'sim-patch', 'b1', 'b2', 'b3', 'b4',
                                                      'b-BLEU-cum', 'b-BLEU-ave', 'b-jaccard' ,'p1', 'p2', 'p3', 'p4', 'p-BLEU-cum',
                                                      'p-jaccard', 'p-BLEU-ave'], delimiter='\t')
            writer.writeheader()
            for row in reader:
                if float(row['b-BLEU-ave']) >= 1.0:
                    writer.writerow(row)
    semanticSpace("similarHeldout2Train_similar_same.csv")


def semanticSpace(csvFile):
    pBleuCum = list()
    pBleuAve = list()
    with open(csvFile, 'r') as csvf:
        reader = csv.DictReader(csvf, delimiter='\t')
        for row in reader:
            pBleuCum.append(float(row["p-BLEU-cum"]))
            pBleuAve.append(float(row["p-BLEU-ave"]))
    #hist(pBleuCum, [0.0, 0.2, 0.4, 0.6, 0.8, 1.01], "blue", np.ones(len(pBleuCum)) / len(pBleuCum),
        #'cumulative BLEU similarity')
    hist(pBleuAve, [0.0, 0.5, 1.01], "green", np.ones(len(pBleuAve)) / len(pBleuAve),
         'average BLEU similarity')
    #hist(pBleuCum, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], "blue", np.ones(len(pBleuCum)) / len(pBleuCum),
         #'cumulative BLEU similarity')
    #hist(pBleuAve, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], "green", np.ones(len(pBleuAve)) / len(pBleuAve),
         #'average BLEU similarity')


def bugPatchSim(buggyFile, patchFile):
    bleuList = list()
    editList = list()
    jaccardList = list()
    bleuBin = np.arange(0, 1.2, 0.1)
    editBin = np.arange(0.1, 22.1, 2)
    jaccardBin = np.arange(0, 1.2, 0.1)

    with open(buggyFile, 'r') as bf:
        buggyLines = bf.readlines()
    with open(patchFile, 'r') as pf:
        patchLines = pf.readlines()

    bpSimFile = open("bugPatchSimilarity.csv", 'w')
    writer = csv.DictWriter(bpSimFile, fieldnames=['bug', 'patch', 'bleu', 'edit_distance', 'jaccard'], delimiter="\t")
    writer.writeheader()

    for idx, bl in enumerate(buggyLines):
        csvDict = dict()
        csvDict["bug"] = bl.strip()
        csvDict["patch"] = patchLines[idx].strip()

        bleu = bleuScore(bl.strip().split(), patchLines[idx].strip().split())
        # csvDict["bleu"] = str(bleu[4])
        aveBLEU = (sum(bleu[:4]) / 4)
        csvDict["bleu"] = str(aveBLEU)
        bleuList.append(aveBLEU)

        edit = calculate_edit_distance(bl, patchLines[idx])
        editList.append(edit)
        csvDict["edit_distance"] = str(edit)

        jaccard = get_jaccard_sim(bl.strip(), patchLines[idx].strip())
        jaccardList.append(jaccard)
        csvDict["jaccard"] = str(jaccard)

        # writerow() has bad behaviors regarding quote marks
        bpSimFile.write('\t'.join(
            [csvDict["bug"], csvDict["patch"], csvDict["bleu"], csvDict["edit_distance"], csvDict["jaccard"]]) + '\n')
    print("bleu: " + str(sum(bleuList) / len(bleuList)) + " edit_distance: " + str(
        sum(editList) / len(editList)) + " jaccard: " + str(sum(jaccardList) / len(jaccardList)))
    perc = sum(i >= 0.5 for i in bleuList) / len(bleuList)

    print("BLEU > 0.5: ", perc)
    hist(bleuList, bleuBin, "green", np.ones(len(bleuList)) / len(bleuList), 'BLEU similarity')
    perc = sum(i <= 1 for i in editList) / len(editList)
    print("Edit distance <= 2: ", perc)
    perc = sum(i <= 2 for i in editList) / len(editList)
    print("Edit distance <= 3: ", perc)
    hist(editList, editBin, "skyblue", np.ones(len(editList)) / len(editList), 'edit distance similarity',
         np.arange(0, 22, 2))
    perc = sum(i >= 0.6 for i in jaccardList) / len(jaccardList)
    print("Jaccard distance >= 0.6: ", perc)
    perc = sum(i >= 0.8 for i in jaccardList) / len(jaccardList)
    print("Jaccard distance >= 0.8: ", perc)
    hist(jaccardList, jaccardBin, 'gray', np.ones(len(jaccardList)) / len(jaccardList), 'jaccard similarity')


def calcNSim(buggyFile, buggySimFile=None):
    # This function is to analyze whether similar bugs will generate similar fixes.
    nMinList = list()
    if buggySimFile is not None:
        # Calculate similarity between files
        with open(buggyFile, 'r') as bf:
            buggyLines = bf.readlines()
            with open(buggySimFile, 'r') as bsf:
                buggySimLines = bsf.readlines()
                # for ix, src in enumerate(buggyLines[0:5000]):
                for ix, src in enumerate(buggyLines):
                    print("Calculating No." + str(ix) + " lines......")
                    simList = list()
                    for idx, tgt in enumerate(buggySimLines):
                        # simScore = calculate_edit_distance(src, tgt)
                        simScore = -jaccard_ngram(src, tgt)
                        simList.append((simScore, idx))
                    nMinElem = Nminelements(simList, 3, skip=False)
                    nMinList.append(nMinElem)
    else:
        # Calculate similarity only in single file
        with open(buggyFile, 'r') as bf:
            buggyLines = bf.readlines()
            for ix, src in enumerate(buggyLines[0:10000]):
                print("Calculating No." + str(ix) + " lines......")
                simList = list()
                for idx, tgt in enumerate(buggyLines):
                    # simScore = calculate_edit_distance(src, tgt)
                    simScore = -jaccard_ngram(src, tgt)
                    simList.append((simScore, idx))
                nMinElem = Nminelements(simList, 4)
                nMinList.append(nMinElem)
    return nMinList


def simBug2simPatch():
    buggyHeldOutFile = "Data/Analysis/valid-test-buggy.txt"
    patchHeldOutFile = "Data/Analysis/valid-test-fixed.txt"
    buggyTrainFile = "Data/Analysis/train-buggy.txt"
    patchTrainFile = "Data/Analysis/train-fixed-filtered.txt"

    # de-duplicate the test data
    with open("duplicate_indices.txt", 'r') as di:
        dupStr = di.readlines()

    dupIdx = [int(d.strip()) for d in dupStr]

    bleuBuggy = list()
    bleuPatch = list()
    for i in range(1, 6):
        bb = list()
        bp = list()
        bleuBuggy.append(bb)
        bleuPatch.append(bp)

    nMinList = calcNSim(buggyHeldOutFile, buggyTrainFile)
    simBPFile = open("similarHeldout2Train_full.csv", 'w')
    writer = csv.DictWriter(simBPFile,
                            fieldnames=['bug', 'sim-bug', 'patch', 'sim-patch', 'b1', 'b2', 'b3', 'b4', 'b-BLEU-cum',
                                        'b-BLEU-ave', 'b-jaccard' , 'p1', 'p2', 'p3', 'p4', 'p-BLEU-cum', 'p-BLEU-ave', 'p-jaccard'],
                            delimiter="\t")
    writer.writeheader()

    bh = open(buggyHeldOutFile, 'r')
    bLines = bh.readlines()
    bt = open(buggyTrainFile, 'r')
    btLines = bt.readlines()

    ph = open(patchHeldOutFile, 'r')
    pLines = ph.readlines()
    pt = open(patchTrainFile, 'r')
    ptLines = pt.readlines()

    for idx, l in enumerate(nMinList):
        if idx in dupIdx:
            continue

        buggyLine = bLines[idx]
        patchLine = pLines[idx]
        for pair in l:
            # pair: (score, lineNo)
            csvDict = dict()
            csvDict['bug'] = buggyLine.strip().replace('\t', ' ')
            csvDict['patch'] = patchLine.strip().replace('\t', ' ')
            csvDict['sim-bug'] = btLines[pair[1]].strip().replace('\t', ' ')
            csvDict['sim-patch'] = ptLines[pair[1]].strip().replace('\t', ' ')

            bScoreBuggy = bleuScore(btLines[pair[1]].strip().split(), buggyLine.strip().split())
            csvDict['b1'], csvDict['b2'], csvDict['b3'], csvDict['b4'], csvDict['b-BLEU-cum'] = \
                bScoreBuggy[0], bScoreBuggy[1], bScoreBuggy[2], bScoreBuggy[3], bScoreBuggy[4]
            csvDict['b-BLEU-ave'] = sum(bScoreBuggy[:4]) / 4
            csvDict['b-jaccard'] = jaccard_ngram(csvDict['bug'], csvDict['sim-bug'])

            bScorePatch = bleuScore(ptLines[pair[1]].strip().split(), patchLine.strip().split())
            csvDict['p1'], csvDict['p2'], csvDict['p3'], csvDict['p4'], csvDict['p-BLEU-cum'] = \
                bScorePatch[0], bScorePatch[1], bScorePatch[2], bScorePatch[3], bScorePatch[4]
            csvDict['p-BLEU-ave'] = sum(bScorePatch[:4]) / 4
            csvDict['p-jaccard'] = jaccard_ngram(csvDict['patch'], csvDict['sim-patch'])
            # writerow() has bad behaviors regarding quote marks
            simBPFile.write('\t'.join(
                [csvDict['bug'], csvDict['sim-bug'], csvDict['patch'], csvDict['sim-patch'], str(csvDict['b1']),
                 str(csvDict['b2']), str(csvDict['b3']), str(csvDict['b4']), str(csvDict['b-BLEU-cum']),
                 str(csvDict['b-BLEU-ave']), str(csvDict['b-jaccard']),
                 str(csvDict['p1']), str(csvDict['p2']), str(csvDict['p3']), str(csvDict['p4']),
                 str(csvDict['p-BLEU-cum']), str(csvDict['p-BLEU-ave']), str(csvDict['p-jaccard'])]) + '\n')
            for ix, b in enumerate(bScoreBuggy):
                bleuBuggy[ix].append(b)
            for ix, p in enumerate(bScorePatch):
                bleuPatch[ix].append(p)

    print(len(bleuBuggy), len(bleuPatch))

    simBPFile.close()


def newVocab(buggyFile, patchFile):
    with open(buggyFile, 'r', encoding='ISO-8859-1') as bf:
        buggyLines = bf.readlines()
    with open(patchFile, 'r', encoding='ISO-8859-1') as pf:
        patchLines = pf.readlines()

    newVocab = 0

    for idx, bl in enumerate(buggyLines):
        b = set(bl.strip().split('\t'))
        p = set(patchLines[idx].strip().split('\t'))
        
        if not p.issubset(b):
            diff = p.difference(b)
            bl = bl.strip().replace('\t', ' ')
            pl = patchLines[idx].strip().replace('\t', ' ')
            newVocab += 1         

    print (newVocab / len(buggyLines))


def processContextLine(cl, size):
    cl = cl.strip().split("###")
    cRange, fileText = cl[0], cl[1]
    cRange = [int(cRange.split(',')[0].strip('[')), int(cRange.split(',')[1].strip(']'))]
    fileText = fileText.split('\t')
    if size != 'all':
        context = fileText[(cRange[0] - size):(cRange[1] + size +1)]
    else:
        context = fileText[:]
    return context


def newVocabContext(buggyFile, patchFile, N):
    with open(buggyFile, 'r', encoding='ISO-8859-1') as bf:
        buggyLines = bf.readlines()
    with open(patchFile, 'r', encoding='ISO-8859-1') as pf:
        patchLines = pf.readlines()

    newVocab = 0
    for idx, bl in enumerate(buggyLines):
        b = set(processContextLine(bl, N))
        p = set(patchLines[idx].strip().split('\t'))
        if not p.issubset(b):
            newVocab += 1
            diff = p.difference(b)

    print (newVocab / len(buggyLines))


def syntaxSim(buggyFile, patchFile):
    with open(buggyFile, 'r', encoding='ISO-8859-1') as bf:
        buggyLines = bf.readlines()
    with open(patchFile, 'r', encoding='ISO-8859-1') as pf:
        patchLines = pf.readlines()

    sameSyntax = 0
    cnt = 0

    for idx, bl in enumerate(buggyLines):
        try:
            bs = toJavaSourceCode(bl.strip())
            buggyTokenType = lexicalAnalysis(bs)
            ps = toJavaSourceCode(patchLines[idx].strip())
            patchTokenType = lexicalAnalysis(ps)
        except:
            continue
        cnt += 1
        if buggyTokenType != '' and patchTokenType != '' and buggyTokenType == patchTokenType:
            sameSyntax += 1

    print (sameSyntax / cnt)


# section 4.1.1 experiment
# newVocab('Data/Analysis/train-buggy.txt', 'Data/Analysis/train-fixed-filtered.txt')
# newVocab('Data/Analysis/train-bpe-buggy.txt', 'Data/Analysis/train-bpe-fixed-filtered.txt')
# newVocabContext('Data/Analysis/train-context.txt', 'Data/Analysis/train-fixed-filtered.txt', 'all')
# newVocabContext('Data/Analysis/train-context-bpe.txt', 'Data/Analysis/train-bpe-fixed-filtered.txt', 'all')

# section 4.1.2 experiment: (bug, sim-bug) vs (patch, sim-patch) analysis
# simBug2simPatch()
# You need to delete the samples cannot be processed to continue, or directly draw heatmap by given results
# draw heatmap
# heatMapFromCSV("similarHeldout2Train_full.csv", "b-BLEU-ave", "p-BLEU-ave", "BLEU")
# heatMapFromCSV("similarHeldout2Train_full.csv", "b-jaccard", "p-jaccard", "Jaccard")

# section 4.1.2 experiment: Table 2; Please change the threshold for the result of each row.
# semanticAmbiguity("similarHeldout2Train_full.csv")

# section 4.2.1 experiment
# Similarity analysis between patches and bugs
# bugPatchSim("Data/Analysis/train-buggy.txt", "Data/Analysis/train-fixed-filtered.txt")

# section 4.2.2 experiment
# syntaxSim('Data/Analysis/train-buggy.txt', 'Data/Analysis/train-fixed-filtered.txt')
# syntaxSim('train-buggy-unseen.txt', 'train-fixed-filtered-unseen.txt')
# syntaxSim('train-buggy-seen.txt', 'train-fixed-filtered-seen.txt')


