import sys
import os
import csv
import glob
import shutil
import javalang
import numpy as np
import nltk
from nltk import ngrams
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({'font.size': 20})

def heatMapFromCSV(csvFile, buggyKey, patchKey, metric):
    realDataB = list()
    realDataP = list()
    with open(csvFile, 'r') as csvf:
        reader = csv.DictReader(csvf, delimiter='\t')
        for r in reader:
            try:
                realDataB.append(float(r[buggyKey]))
                realDataP.append(float(r[patchKey]))
            except:
                print(r)
                continue
    x = np.array(realDataB)
    y = np.array(realDataP)

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[10, 10])

    extent = [0, 1, 0, 1]

    plt.xlabel('Buggy ' + metric)
    plt.ylabel('Patch ' + metric)
    #plt.title('Ave-' + metric)

    #plt.clf()
    plt.imshow(heatmap.T, extent=extent, norm=mpl.colors.LogNorm(), cmap='inferno', origin='lower')

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.show()


def hist(data, bins, color, weights, title, xt=None):
    n, bins, patches = plt.hist(data, bins=bins, rwidth=0.85, color=color, weights=weights)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylabel('Percentage')
    if xt is not None:
        plt.xticks(xt)
    plt.title(title)
    print(n, bins, patches)
    plt.show()


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def jaccard_ngram(str1, str2):
    lista = str1.strip().split()
    listb = str2.strip().split()
    jaccardList = list()
    for i in range(1, 5):
        a = set(ngrams(lista, i))
        b = set(ngrams(listb, i))
        c = a.intersection(b)
        if (len(a) + len(b) - len(c)) > 0:
        	tmp = float(len(c)) / (len(a) + len(b) - len(c))
        else:
        	tmp = 0.0
        jaccardList.append(tmp)
    return sum(jaccardList) / len(jaccardList)


def calculate_edit_distance(org_code, cand_code):
    """
	The higher the score, the lower the similarity.
	Pay attention to \n symbol in the line
	"""
    org_parts = [part.strip() for part in org_code.strip().split()]
    cand_parts = [part.strip() for part in cand_code.strip().split()]

    def levenshteinDistance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    return levenshteinDistance(org_parts, cand_parts)
    pass


def bleuScore(reference, hypothesis):
    # bAve = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
    b1 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0))
    if b1 < 0.01:
        b1 = 0
    b2 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 1, 0, 0))
    if b2 < 0.01:
        b2 = 0
    b3 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 0, 1, 0))
    if b3 < 0.01:
        b3 = 0
    b4 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 0, 0, 1))
    if b4 < 0.01:
        b4 = 0
    b = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    if b < 0.01:
        b = 0
    return [b1, b2, b3, b4, b]


def Nminelements(list1, N, skip=True):
    # list1: [(score, lineNo)...]
    final_list = []

    for i in range(0, N):
        min1 = (float('Inf'), -1)

        for j in range(len(list1)):
            if list1[j][0] < min1[0]:
                min1 = list1[j]

        list1.remove(min1)
        if skip:
            # skip the first min because that is buggyline itself.
            if (i > 0):
                final_list.append(min1)
        else:
            final_list.append(min1)

    return final_list


def dataSampling():
    heldOut = list()
    test = list()
    with open("DataSplit/heldout_keys.txt", 'r') as h:
        for key in h.readlines():
            k = key.strip()
            heldOut.append(k)

    with open("DataSplit/test_keys.txt", 'r') as t:
        for key in t.readlines():
            k = key.strip()
            test.append(k)

    heldOut = set(heldOut)
    test = set(test)

    for file in glob.glob("Data/*.csv")[1001:1501]:
        key = file.strip("Data/").strip(".csv").replace('__', '/')
        if key not in heldOut and key not in test:
            # shutil.copy(file, "SampledData/")
            shutil.copy(file, "SampledHeldOut/")


def BPEinput(pattern, trainSize):
    buggyPrefix = "Files/Files-pre"
    inputFile = str(trainSize) + "_BPEinput.txt"
    with open(inputFile, 'w', encoding="ISO-8859-1") as inpf:
        for file in glob.glob(pattern)[:trainSize]:
            with open(file) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row["is_fix"] == "True" and row["lines_removed"] == '1' and row["lines_added"] == '1' and row[
                        "line_rm_start"] == row["line_add_start"]:
                        organization = row["organization"]
                        project = row["project"]
                        commit = row["commit"]
                        fileName = row["file"].replace("/", "__")
                        filePath = '#'.join([row["organization"], row["project"], row["commit"], fileName])
                        try:
                            with open(os.path.join(buggyPrefix, filePath), 'r', encoding='ISO-8859-1') as bf:
                                codeLines = bf.readlines()
                                for cl in codeLines:
                                    if cl.strip() != "":
                                        cl = cl.replace('\t', ' ')
                                        inpf.write(cl)
                        except:
                            continue


def tokenizeLine(line):
    token_string = ""
    try:
        tokens = list(javalang.tokenizer.tokenize(line))
        for token in tokens:
            v = token.value
            token_string = token_string + v + "\t"
    except Exception as e:
        print("This Line cannot be parsed: \n" + line + "\n")
    return token_string


def tokenize(file):
    f = open(file, "r", encoding='ISO-8859-1')
    file_lines = f.readlines()

    tokens_string = ""
    for line in file_lines:
        if line.strip().startswith('*') or \
                line.strip().startswith('/*') or \
                line.strip().startswith('//') or \
                line.strip().startswith('*') or \
                line.strip().startswith('/*') or \
                line.strip().startswith('//') or \
                line.strip().endswith('*/'):
            continue
        try:
            tokens = list(javalang.tokenizer.tokenize(line))
            for token in tokens:
                if isinstance(token, javalang.tokenizer.String):
                    v = '"str"'
                else:
                    v = token.value
                tokens_string = tokens_string + v + "\t"
        except Exception as e:
            # print ("A line cannot be parsed")
            # sys.stderr.write("This Line cannot be parsed: \n" + line + "\n")
            continue

    f.close()

    return tokens_string


def focusIdentifier(line):
    # focus on the change of identifiers
    tokens_string = ""
    try:
        tokens = list(javalang.tokenizer.tokenize(line))
        for token in tokens:
            if isinstance(token, javalang.tokenizer.Identifier):
                tokens_string = tokens_string + token.value + " "

    except Exception as e:
        sys.stderr.write("This Line cannot be parsed: \n" + line + "\n")

    return tokens_string


def lexicalAnalysis(line):
    tokens_string = ""
    try:
        tokens = list(javalang.tokenizer.tokenize(line))

        for token in tokens:
            if isinstance(token, javalang.tokenizer.String):
                v = "String"
            elif isinstance(token, javalang.tokenizer.EndOfInput):
                v = "EndOfInput"
            elif isinstance(token, javalang.tokenizer.Keyword):
                # v = token.value
                v = "Keyword"
            elif isinstance(token, javalang.tokenizer.Modifier):
                # v = token.value
                v = "Modifier"
            elif isinstance(token, javalang.tokenizer.BasicType):
                # v = token.value
                v = "BasicType"
            elif isinstance(token, javalang.tokenizer.Literal):
                v = "Literal"
            elif isinstance(token, javalang.tokenizer.Integer):
                v = "Integer"
            elif isinstance(token, javalang.tokenizer.DecimalInteger):
                v = "DecimalInteger"
            elif isinstance(token, javalang.tokenizer.OctalInteger):
                v = "OctalInteger"
            elif isinstance(token, javalang.tokenizer.BinaryInteger):
                v = "BinaryInteger"
            elif isinstance(token, javalang.tokenizer.HexInteger):
                v = "HexInteger"
            elif isinstance(token, javalang.tokenizer.FloatingPoint):
                v = "FloatingPoint"
            elif isinstance(token, javalang.tokenizer.DecimalFloatingPoint):
                v = "DecimalFloatingPoint"
            elif isinstance(token, javalang.tokenizer.HexFloatingPoint):
                v = "HexFloatingPoint"
            elif isinstance(token, javalang.tokenizer.Boolean):
                # v = token.value
                v = "Boolean"
            elif isinstance(token, javalang.tokenizer.Character):
                v = "Character"
            elif isinstance(token, javalang.tokenizer.Null):
                v = "Null"
            elif isinstance(token, javalang.tokenizer.Separator):
                # v = v.value
                v = "Separator"
            elif isinstance(token, javalang.tokenizer.Operator):
                v = "Operator"
            elif isinstance(token, javalang.tokenizer.Annotation):
                v = "Annotation"
            elif isinstance(token, javalang.tokenizer.Identifier):
                v = "Identifier"
            else:
                print("Error")
                print(type(token))

            tokens_string = tokens_string + v + "\t"

    except Exception as e:
        sys.stderr.write("This Line cannot be parsed: \n" + line + "\n")

    return tokens_string


def toJavaSourceCode(prediction):
    tokens = prediction.strip().split("\t")
    codeLine = ""
    delimiter = JavaDelimiter()
    for i in range(len(tokens)):
        if (i + 1 < len(tokens)):

            if (not isDelimiter(tokens[i])):
                if (not isDelimiter(tokens[i + 1])):  # STR (i) + STR (i+1)
                    codeLine = codeLine + tokens[i] + " "
                else:  # STR(i) + DEL(i+1)
                    codeLine = codeLine + tokens[i]
            else:
                if (tokens[i] == delimiter.varargs):  # ... (i) + ANY (i+1)
                    codeLine = codeLine + tokens[i] + " "
                elif (tokens[i] == delimiter.biggerThan):  # > (i) + ANY(i+1)
                    codeLine = codeLine + tokens[i] + " "
                elif (tokens[i] == delimiter.rightBrackets and i > 0):
                    if (tokens[i - 1] == delimiter.leftBrackets):  # [ (i-1) + ] (i)
                        codeLine = codeLine + tokens[i] + " "
                    else:  # DEL not([) (i-1) + ] (i)
                        codeLine = codeLine + tokens[i]
                else:  # DEL not(... or ]) (i) + ANY
                    codeLine = codeLine + tokens[i]
        else:
            codeLine = codeLine + tokens[i]
    return codeLine


def isDelimiter(token):
    return not token.upper().isupper()


class JavaDelimiter:
    @property
    def varargs(self):
        return "..."

    @property
    def rightBrackets(self):
        return "]"

    @property
    def leftBrackets(self):
        return "["

    @property
    def biggerThan(self):
        return ">"

