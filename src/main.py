# This program receives the tagger type and the path to a test file
# as command line parameters and outputs the POS tagged version of that file.
import argparse  # For positional command-line arguments

from nltk.tag.hmm import  HiddenMarkovModelTrainer
from nltk.tag import brill, brill_trainer, BrillTagger, RegexpTagger, BrillTaggerTrainer, DefaultTagger, UnigramTagger, \
    BigramTagger, TrigramTagger
from nltk.tag.brill import Pos, Word, brill24, fntbl37, nltkdemo18, nltkdemo18plus
from nltk.probability import LidstoneProbDist, MLEProbDist, SimpleGoodTuringProbDist, KneserNeyProbDist, \
    WittenBellProbDist, ELEProbDist\
    ,LaplaceProbDist,LidstoneProbDist


import matplotlib.pyplot as plt

def create_argument_parser():
    """ Create a parser and add 2 arguments to it
    --tagger to indicate the tagger type
    --train for the path to a training corpus
    --test for the path to a test corpus
    --output for the output file
    Returns:
        args: stores all args
    """
    parser = argparse.ArgumentParser(
        "Correctness of grammar of a sentence")
    parser.add_argument("--tagger", help="which tagger")
    parser.add_argument("--train", help="Path to the train file")
    parser.add_argument("--test", help="Path to the test file")
    parser.add_argument("--output", help="Path to the output file")

    args = parser.parse_args()
    # print(type(args))
    return args


def read_file(path):
    """ read file from path

    Args:
        path (string): path of file

    Returns:
        data: text of file
    """
    with open(path, 'r') as f:
        return f.read()


def split_train_set_to_train_and_dev(data):
    """split train set to train and dev

    Args:
        data (list): list of sentences with tags

    Returns:
        train: 80% of data for training
        dev:   200% of data for tunning parameters
    """
    # np.random.seed(0)
    # np.random.shuffle(data)

    n = len(data)
    train = data[:int(.8 * n)]
    dev = data[int(.8 * n):]
    return train, dev


def preprocess(data):
    """preprocess raw data into list of sentences with tags

    Args:
        data (String): text of file

    Returns:
        sentences: list of sentences with tags
    """
    list_data = []
    data = data.split('\n\n')
    n = len(data)
    for i in range(len(data)):
        if i != n - 1:
            list_data.append(data[i].split('\n'))
        else:
            list_data.append(data[i].split('\n')[:-1])
    sentences = []
    for sentence in list_data:

        temp = []
        for word_label in sentence:
            word = word_label.split()[0]
            label = word_label.split()[1]
            temp.append((word, label))
        sentences.append(temp)
    return sentences


def write_output_file(predicted_words_tags, file_path):
    """ Write output file

    Args:
        predicted_words_tags (_type_): _description_
        file_path (_type_): _description_
    """
    data = ""
    for sentence in predicted_words_tags:
        for word_tag in sentence:
            data = data + word_tag[0] + " " + word_tag[1] + '\n'
        data = data + '\n'
    with open(file_path, 'w') as f:
        f.write(data)
        f.close()


def HMM_tagger(train_data, dev_data, test_data, output_file,which_tagger,which_test):
    probability_dists = [["LaplaceProbDist",LaplaceProbDist],['LidstoneProbDist',LidstoneProbDist], ["MLEProbDist",MLEProbDist], ["SimpleGoodTuringProbDist",SimpleGoodTuringProbDist], ["WittenBellProbDist",WittenBellProbDist], ["ELEProbDist",ELEProbDist]]
    best_acc = 0
    for prob_dist in probability_dists:
        trainer = HiddenMarkovModelTrainer()
        if prob_dist[0] =="LaplaceProbDist":
             tagger = trainer.train_supervised(train_data, estimator=prob_dist[1])
        else:
            tagger = trainer.train_supervised(train_data, estimator=lambda fd, bins: prob_dist[1](fd, bins))
        acc = tagger.accuracy(dev_data)
        print(acc, prob_dist[0])
        if acc > best_acc:
            best_acc = acc
            best_prob_dist = prob_dist[0]
            best_tagger = tagger
    
    predict_pos(test_data, best_tagger, output_file, best_prob_dist,which_tagger,which_test)

    return acc


def predict_pos(test_data, tagger, output_path,exp,which_tagger,which_test):
    sentences_without_tags = []
    predicted_tags_for_all_sentences = []
    for sentence in test_data:
        sentence_without_tags = []
        for word_tag in sentence:
            sentence_without_tags.append(word_tag[0])
        sentences_without_tags.append(sentence_without_tags)
    for sentence in sentences_without_tags:
        predicted_tags_for_all_sentences.append(tagger.tag(sentence))

    write_output_file(predicted_tags_for_all_sentences, output_path)
    print("Best tagger: ",tagger.accuracy(test_data),exp)
    
    final_accuracy, labels = calculate_Accuracy (predicted_tags_for_all_sentences, test_data,tagger,which_tagger,which_test)

    
    
    return final_accuracy

def calculate_Accuracy(predictedTagswithWords, ground_truth,tagger,which_tagger,which_test):
    y_true = []
    y_pred = []
    labels = []
    
    tags={}
    for i in range(len(predictedTagswithWords)):
        truth_sentence = ground_truth[i]
        predicted_sentence = predictedTagswithWords[i]
        j = 0
        for word_tag in truth_sentence:
            true_value = word_tag[1]
            y_true.append(true_value)
            predicted_value = predicted_sentence[j][1]
            y_pred.append(predicted_value)
            j+=1
            if true_value not in tags.keys():
                tags[true_value]={predicted_value:1}
            else:
                if predicted_value not in tags[true_value].keys():
                    tags[true_value][predicted_value]=1
                else:
                     tags[true_value][predicted_value] +=1
            
            if predicted_value not in labels:
                labels.append(predicted_value)
            if true_value not in labels:
                labels.append(true_value)
    error_analysis = []
    for tag in tags.keys():
        misclassified_counter = 0
        most_mis_tag =0
        most_mis_number =0
        for i in tags[tag].keys():
            if i != tag:
                misclassified_counter+=tags[tag][i]
                if tags[tag][i]>most_mis_number:
                    most_mis_number =tags[tag][i]
                    most_mis_tag =i
        error_analysis.append([[tag,most_mis_tag],most_mis_number/sum(tags[tag].values())])
       

    error_analysis = sorted(error_analysis,key=lambda x:x[1])
    misclassified_tags = ['' for i in range(10)]
    val = [0 for i in range(10)]
    x =[]
    j= 1
    k=9
    for i in error_analysis[-10:]:
        
        misclassified_tags[k] =('('+i[0][0] + " " + str(i[0][1]+')')) 
        val[k]=i[1]  
        k-=1

        x.append(j)
        j+=1 
    

    fig = plt.figure(figsize = (10, 5))
    plt.bar(x,val,tick_label = misclassified_tags,color ='maroon',
        width = 0.4)
 
    plt.xlabel("10 pairs of most misclassified tags")
    plt.ylabel("Percentage of misclassification tag")
    plt.title("Error Analysis of "+which_tagger+' tagger for '+which_test+ " data")
    plt.savefig('error_analysis_'+which_tfig = plt.figure(figsize = (10, 5))
    plt.bar(x,val,tick_label = misclassified_tags,color ='maroon',
        width = 0.4)

    plt.xlabel("10 pairs of most misclassified tags")
    plt.ylabel("Percentage of misclassification tag")
    plt.title("Error Analysis of "+which_tagger+' tagger for '+which_test+ " data")
    plt.savefig('error_analysis_'+which_tagger+'_for_'+which_test+ '_data_.png')agger+'_for_'+which_test+ '_data_.png')
    
    acc = tagger.accuracy(test_data)
    return acc, labels



def train_brill_tagger(initial_tagger, templates, train_sents, dev_data, max_rule):
    trainer = brill_trainer.BrillTaggerTrainer(
        initial_tagger, templates, deterministic=True, trace=False)
    return trainer.train(train_sents, max_rules = max_rule)


def backoff_tagger(train_data, backoff=None):
    tagger_classes = [["UnigramTagger",UnigramTagger],['BigramTagger', BigramTagger], ["TrigramTagger",TrigramTagger]]
    initial_tags = []
    for tagger in tagger_classes:
        initial_tags.append([tagger[0], tagger[1](train_data, backoff=backoff)])
    regex_tagger = RegexpTagger([
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'(The|the|A|a|An|an)$', 'AT'),  # articles
        (r'.*able$', 'JJ'),  # adjectives
        (r'.*ness$', 'NN'),  # nouns formed from adjectives
        (r'.*ly$', 'RB'),  # adverbs
        (r'.*s$', 'NNS'),  # plural nouns
        (r'.*ing$', 'VBG'),  # gerunds
        (r'.*ed$', 'VBD'),  # past tense verbs
        (r'.*', 'NN')  # nouns (default)
    ])
    
    initial_tags.append(["RegexpTagger",regex_tagger])
    
    return initial_tags


def brill_tagger(train_data, dev_data, test_data, output_file,which_tagger,which_test):
    backoff = RegexpTagger([
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'(The|the|A|a|An|an)$', 'AT'),  # articles
        (r'.*able$', 'JJ'),  # adjectives
        (r'.*ness$', 'NN'),  # nouns formed from adjectives
        (r'.*ly$', 'RB'),  # adverbs
        (r'.*s$', 'NNS'),  # plural nouns
        (r'.*ing$', 'VBG'),  # gerunds
        (r'.*ed$', 'VBD'),  # past tense verbs
        (r'.*', 'NN')  # nouns (default)
    ])
    # we should tune initial tagger:
    initial_taggers = backoff_tagger(train_data, backoff)
    
    templates = [["nltkdemo18", nltkdemo18()], ["nltkdemo18plus", nltkdemo18plus()], ["fntbl37", fntbl37()],
                 ["brill24", brill24()]]
    best_acc = 0
    best_template = 0
    best_max_rule = 0
    best_initial_tagger = 0
    print("best_initial_tagger "+"best_max_rules "+"best_template")
    max_rules =[10,50,100]
    for template in templates:
        for initial_tagger in initial_taggers:
            for max_rule in max_rules:
                tagger = train_brill_tagger(initial_tagger[1], template[1], train_data, dev_data,max_rule)
                accuracy = tagger.accuracy(dev_data)

            # find max accuracy.....
            ######################################################
                acc = tagger.accuracy(dev_data)
                print(acc, template[0],initial_tagger[0],max_rule)
                if acc > best_acc:
                    best_acc = acc
                    best_template = template[0]
                    best_tagger = tagger
                    best_max_rule = max_rule
                    best_initial_tagger = initial_tagger[0]


    # testing
    exp = best_initial_tagger+' '+str(best_max_rule)+' '+best_template
    predict_pos(test_data, best_tagger, output_file,exp,which_tagger,which_test)
    return tagger



if __name__ == '__main__':
    args = create_argument_parser()
    train_data = read_file(args.train)
    test_data = read_file(args.test)
    train_data = pre_processed_text = preprocess(train_data)
    train_data, dev_data = split_train_set_to_train_and_dev(train_data)
    test_data = pre_processed_text = preprocess(test_data)
    which_test = args.test
    which_test = which_test.split('/')[1].split('.')[0]
    
    
    if (args.tagger == 'hmm'):
        HMM_tagger(train_data, dev_data, test_data, args.output,args.tagger,which_test)
    else:
        brill_tagger(train_data, dev_data, test_data, args.output,args.tagger,which_test)