import sys
import torch
import json
from torch.utils.data import DataLoader
import os
import model
import Train
import bleu_eval


test_data = "MLDS_hw2_1_data/testing_data" 
test_json = "MLDS_hw2_1_data/testing_label.json"  
model_path = "saved/modelarch.h5" 
outputfile_path = "output_file.txt"  

def main():
   
    try:
        os.chmod(test_data, 0o755) 
        
    except PermissionError:
        print("Permission denied!")

    # Load the model
    modelIP = torch.load(model_path, weights_only=False)

  
    files_dir = 'MLDS_hw2_1_data/testing_data/feat'
    i2w, w2i, dictionary = Train.dictonaryFunc(4)
    test_dataset = Train.test_dataloader(files_dir)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=8)

  
    model = modelIP
    ss = Train.testfun(test_dataloader, model, i2w)


   
    try:
        with open(outputfile_path, 'w') as f:
            for id, s in ss:
                f.write('{},{}\n'.format(id, s))
            print('File updated successfully!')
    except FileNotFoundError:
        with open(outputfile_path, 'x') as f:
            for id, s in ss:
                f.write('{},{}\n'.format(id, s))
            print('File created and updated successfully!')

    # BLEU Evaluation
    test = json.load(open(test_json, 'r'))
    result = {}

    with open(outputfile_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma + 1:]
            result[test_id] = caption

    bleu = []
    for item in test:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(bleu_eval.BLEU(result[item['id']], captions, True))
        bleu.append(score_per_video[0])

    average = sum(bleu) / len(bleu)
    print("Average BLEU score is " + str(average))

if __name__ == '__main__':
    main()
