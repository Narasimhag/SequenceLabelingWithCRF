from hw2_corpus_tool import get_data
import time
import sys
import pycrfsuite

def utterance_to_features(dialogue, i):
    features = []
    # t_len = 0
    text = []
    
    if(i>0):
        # 1. feature to identify speaker change --> present as CHANGE if speaker changes
        # absent if speaker does not change, if it is first utterance it is always CHANGE
        if(dialogue[i-1].speaker != dialogue[i].speaker):
            features.append('CHANGE')
        
        # 2. feature for every word in utterance --> TOKEN_word
        # 3. feature for every pos tag in utterance --> POS_postag
        if dialogue[i].pos is None:
            features.append('NO_WORD')
        
        else:
            features.append('START_TOKEN_'+ dialogue[i].pos[0].token)
            features.append('START_POS_'+ dialogue[i].pos[0].pos)
            features.append('END_TOKEN_' + dialogue[i].pos[-1].token)
            features.append('END_POS_' + dialogue[i].pos[-1].pos)

            for k in dialogue[i].pos:
                features.append("TOKEN_" + k.token)
                features.append("POS_" + k.pos)
                # t_len += len(k.token)
                text.append(k.token)

            # features.append("TEXT_" + dialogue[i].text)
            features.append("TEXT_" + ''.join(text))
            # if ( "?" in dialogue[i].text):
            #     features.append("QUES")
            features.append("NUMT_" + str(len(dialogue[i].pos)))

            if(len(dialogue[i].pos) >= 2):
                for k in range(len(dialogue[i].pos) - 1):
                    features.append("BI_TOKEN" + dialogue[i].pos[k].token + "," + dialogue[i].pos[k+1].token)
                    features.append("BI_POS" + dialogue[i].pos[k].pos + "," + dialogue[i].pos[k+1].pos)
            
            # if(len(dialogue[i].pos) >= 3):
            #     for k in range(len(dialogue[i].pos) - 2):
            #         features.append("TRI_TOKEN" + dialogue[i].pos[k].token + "," + dialogue[i].pos[k+1].token + "," + dialogue[i].pos[k+2].token)
            #         features.append("TRI_POS" + dialogue[i].pos[k].pos + "," + dialogue[i].pos[k+1].pos + "," + dialogue[i].pos[k+2].pos)

            # features.append("LEN_" + str(t_len))
            

    # 4. feature to mark first utterance of dialogue --> labelled FIRST
    # absence of this feature means it is not first
    else:
        features.append("FIRST")
        features.append("CHANGE")
        if dialogue[i].pos is None:
            features.append('NO_WORD')
        
        else:
            features.append('START_TOKEN_'+ dialogue[i].pos[0].token)
            features.append('START_POS_'+ dialogue[i].pos[0].pos)
            features.append('END_TOKEN_' + dialogue[i].pos[-1].token)
            features.append('END_POS_' + dialogue[i].pos[-1].pos)

            for k in dialogue[i].pos:
                features.append("TOKEN_" + k.token)
                features.append("POS_" + k.pos)
                # t_len += len(k.token)
                text.append(k.token)
            features.append("TEXT_" + ''.join(text))
            # if ( "?" in dialogue[i].text):
            #     features.append("QUES")
            features.append("NUMT_" + str(len(dialogue[i].pos)))

    return features

def dialogue_to_features(dialogue):
    return [utterance_to_features(dialogue, i) for i in range(len(dialogue))]
# print(list(get_utterances_from_filename('/Volumes/GoogleDrive/My Drive/Academics/Spring 2020/NLP/Project 2/train/train/0005.csv'))[57])
def dialogue_to_labels(dialogue):
    return [utterance.act_tag for utterance in dialogue]

def data_to_features(data_dir):

    data = list(get_data(data_dir))
    feature_set = [dialogue_to_features(dialogue) for dialogue in data]
    label_set = [dialogue_to_labels(dialogue) for dialogue in data]
    # print(feature_set)
    # print(label_set)
    return feature_set, label_set

if __name__ == "__main__":
    start_time = time.time()
    # data_to_features('/Volumes/GoogleDrive/My Drive/Academics/Spring 2020/NLP/Project 2/train/train/')
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_file_path = sys.argv[3]

    train_x, train_y = data_to_features(train_data_path)
    test_x, test_y = data_to_features(test_data_path)

    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(train_x, train_y):
        trainer.append(xseq, yseq)
    
    trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
    })

    #train the model using the trainer's train method
    trainer.train('something_adv.crfsuite')

    #declare and use the tagger to use the model created
    tagger = pycrfsuite.Tagger()
    tagger.open('something_adv.crfsuite')

    pred_y = []
    for x in test_x:
        pred_y.append(tagger.tag(x))
    # pred_y = tagger.tag(test_x)
    # print(pred_y)
    with open(output_file_path, 'w') as fp:
        for x in pred_y:
            for y in x:
                fp.write(y + '\n')
            fp.write('\n')

    end_time = time.time()


    correct = 0
    count = 0
    for c, v in zip(pred_y, test_y):
        for k, z in zip(c,v):
            if k == z:
                correct += 1
                count += 1
            else:
                count += 1
    
    accuracy = (correct/count) * 100
    print(correct, count, accuracy, (end_time-start_time))
    