import re
import time
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer, RobertaTokenizer, RobertaForMaskedLM
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import logging
import argparse
import codecs
import json
import os
import nltk
import spacy
from CandidatesGenerator import CandidatesGenerator



def write_results(lista, ruta_archivo):
    """
    Escribe los elementos de una lista en un archivo, uno por línea.

    :param lista: Lista de elementos a escribir en el archivo.
    :param ruta_archivo: Ruta completa del archivo donde se guardarán los elementos.
    """
    try:
        with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
            for elemento in lista:
                archivo.write(f"{elemento}\n")
        print(f"Archivo guardado correctamente en: {ruta_archivo}")
    except Exception as e:
        print(f"Error al escribir el archivo: {e}")



class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info'):

        self.logger = logging.getLogger(filename)
        # # format_str = logging.Formatter(fmt)  # 设置日志格式
        # if args.local_rank == 0 :
        #     level = level
        # else:
        #     level = 'warning'
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        # sh.setFormatter(format_str)  # 设置屏幕上显示的格式

        th = logging.FileHandler(filename,'w')
        # formatter = logging.Formatter('%(asctime)s => %(name)s * %(levelname)s : %(message)s')
        # th.setFormatter(formatter)

        self.logger.addHandler(sh)  # 代表在屏幕上输出，如果注释掉，屏幕将不输出
        self.logger.addHandler(th)  # 代表在log文件中输出，如果注释掉，将不再向文件中写入数据


class KPE_Dataset(Dataset):
    def __init__(self, docs_pairs):

        self.docs_pairs = docs_pairs
        self.total_examples = len(self.docs_pairs)

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):

        doc_pair = self.docs_pairs[idx]
        ori_example = doc_pair[0]
        masked_example = doc_pair[1]
        doc_id = doc_pair[2]

        return [ori_example, masked_example, doc_id]
'''


"""pre process code from SIFRank"""
class Result:

    def __init__(self,N=15):
        self.database=""
        self.predict_keyphrases = []
        self.true_keyphrases = []
        self.file_names = []
        self.lamda=0.0
        self.beta=0.0

    def update_result(self, file_name, pre_kp, true_kp):
        self.file_names.append(file_name)
        self.predict_keyphrases.append(pre_kp)
        self.true_keyphrases.append(true_kp)

    def get_parameters(self,database="",lamda=0.6,beta=0.0):
        self.database = database
        self.lamda = lamda
        self.beta = beta

    def write_results(self):
        return 0

'''



def write_string(s, output_path):
    with open(output_path, 'w') as output_file:
        output_file.write(s)


def read_file(input_path):
    with open(input_path, 'r', errors='replace_with_space') as input_file:
        return input_file.read()

def clean_text(text="",database="Inspec"):

    #Specially for Duc2001 Database
    if(database=="Duc2001" or database=="Semeval2017"):
        pattern2 = re.compile(r'[\s,]' + '[\n]{1}')
        while (True):
            if (pattern2.search(text) is not None):
                position = pattern2.search(text)
                start = position.start()
                end = position.end()
                # start = int(position[0])
                text_new = text[:start] + "\n" + text[start + 2:]
                text = text_new
            else:
                break

    pattern2 = re.compile(r'[a-zA-Z0-9,\s]' + '[\n]{1}')
    while (True):
        if (pattern2.search(text) is not None):
            position = pattern2.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + " " + text[start + 2:]
            text = text_new
        else:
            break

    pattern3 = re.compile(r'\s{2,}')
    while (True):
        if (pattern3.search(text) is not None):
            position = pattern3.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + "" + text[start + 2:]
            text = text_new
        else:
            break

    pattern1 = re.compile(r'[<>[\]{}]')
    text = pattern1.sub(' ', text)
    text = text.replace("\t", " ")
    text = text.replace(' p ','\n')
    text = text.replace(' /p \n','\n')
    lines = text.splitlines()
    # delete blank line
    text_new=""
    for line in lines:
        if(line!='\n'):
            text_new+=line+'\n'

    return text_new

def get_long_data(file_path="data/nus/nus_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                fulltxt = jsonl['fulltext']
                doc = ' '.join([abstract, fulltxt])
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="nus")
                doc = doc.replace('\n', ' ')
                data[jsonl['name']] = doc
                labels[jsonl['name']] = keywords
            except:
                raise ValueError
    return data,labels

def get_short_data(file_path="data/kp20k/kp20k_valid2k_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                doc =abstract
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="kp20k")
                doc = doc.replace('\n', ' ')
                data[i] = doc
                labels[i] = keywords
            except:
                raise ValueError
    return data,labels


def get_duc2001_data(file_path="data/DUC2001"):
    pattern = re.compile(r'<TEXT>(.*?)</TEXT>', re.S)
    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            if (fname == "annotations.txt"):
                # left, right = fname.split('.')
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                lines = text.splitlines()
                for line in lines:
                    left, right = line.split("@")
                    d = right.split(";")[:-1]
                    l = left
                    labels[l] = d
                f.close()
            else:
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                text = re.findall(pattern, text)[0]

                text = text.lower()
                text = clean_text(text,database="Duc2001")
                data[fname]=text.strip("\n")
                # data[fname] = text
    return data,labels

def get_inspec_data(file_path="data/Inspec"):

    data={}
    labels={}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            left, right = fname.split('.')
            if (right == "abstr"):
                infile = os.path.join(dirname, fname)
                f=open(infile)
                text=f.read()
                text = text.replace("%", '')
                text=clean_text(text)
                data[left]=text
            if (right == "uncontr"):
                infile = os.path.join(dirname, fname)
                f=open(infile)
                text=f.read()
                text=text.replace("\n",' ')
                text=clean_text(text,database="Inspec")
                text=text.lower()
                label=text.split("; ")
                labels[left]=label
    return data,labels

def get_semeval2017_data(data_path="data/SemEval2017/docsutf8",labels_path="data/SemEval2017/keys"):

    data={}
    labels={}
    for dirname, dirnames, filenames in os.walk(data_path):
        for fname in filenames:
            if not fname.endswith('.txt'):
                continue
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            # f = open(infile, 'rb')
            # text = f.read().decode('utf8')
            with codecs.open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", '')
            text = clean_text(text,database="Semeval2017")
            data[left] = text.lower()
            # f.close()
    for dirname, dirnames, filenames in os.walk(labels_path):
        for fname in filenames:
            if not fname.endswith('.txt'):
                continue
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            f = open(infile, 'rb')
            text = f.read().decode('utf8')
            text = text.strip()
            ls=text.splitlines()
            labels[left] = ls
            f.close()
    return data,labels


def get_exec_dataset(data_path="data/exec_example"):

    data={}
    for dirname, dirnames, filenames in os.walk(data_path):
        for fname in filenames:
            if not fname.endswith('.txt'):
                continue
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            # f = open(infile, 'rb')
            # text = f.read().decode('utf8')
            with codecs.open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", '')
            text = clean_text(text,database="Semeval2017")
            data[left] = text.lower()
            # f.close()

    return data


def remove (text):
    #print(text)
    text_len = len(text.split())
    remove_chars = '[’!"#$%&\'()*+,./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, '', text)
    re_text_len = len(text.split())
    if text_len != re_text_len:
        #print(text)
        return True
    else:
        return False

'''
def dedup_stem(candidates):
    new_can = {}
    can_dedup_stemmed = {}
    for can in candidates:
        can_dedup_stemmed[' '.join(porter.stem(t) for t in can.split())] = can

    for stemmed_can, can in can_dedup_stemmed.items():
        re_flag = remove(can)
        if re_flag:
            candidate_tokens = tokenizer.tokenize(can)
            candidate_len = len(candidate_tokens)
            new_can[can.lower()] = candidate_len
    return  new_can
'''



def find_candidate_mention(tok_candidate,ori_tokens):

    candidate_len = len(tok_candidate)
    ori_doc = ' '.join(ori_tokens)
    can_token = ' '.join(tok_candidate)

    conteo = can_token.count('Ġ')
    #print(tok_candidate,conteo)
    if model_type == 'bert':
        mask = ' '.join(['[MASK]'] * candidate_len)
    else:
        mask = ' '.join(['<mask>'] * candidate_len)


    candidate_re = re.compile(r"\b" + can_token + r"\b")

    masked_doc = re.sub(candidate_re, mask, ori_doc)
    match = candidate_re.findall(ori_doc)
    #print('>', ori_doc)
    #print('>', masked_doc)
    #print('len match >',len(match), match)

    return match,masked_doc


def generate_absent_doc(ori_encode_dict, candidates, idx):

    print(candidates)
    count = 0
    doc_pairs = []
    ori_input_ids = ori_encode_dict["input_ids"].squeeze()
    ori_tokens = tokenizer.convert_ids_to_tokens(ori_input_ids)

    # There are multi candidates for a document
    for id, candidate in enumerate(candidates):

        # Remove stopwords in a candidate
        if remove(candidate):
            count +=1
            continue

        ####

        match=[]
        try:
            if model_type == 'bert':
                tok_candidate = tokenizer.tokenize(candidate)
                match, masked_doc = find_candidate_mention(tok_candidate, ori_tokens)
            else:
                ## roberta
                tok_candidate = tokenizer.tokenize(' ' + candidate)
                match, masked_doc= find_candidate_mention(tok_candidate,ori_tokens)
                if len(match) == 0:
                    print("try again")
                    tok_candidate = tokenizer.tokenize(candidate)
                    match, masked_doc = find_candidate_mention(tok_candidate, ori_tokens)



        except:
            print("cannot replace: ", candidate)
            count += 1
            continue

        if len(match) == 0:
            count +=1
            ori_doc = ' '.join(ori_tokens)
            can_token = ' '.join(tok_candidate)
            #print("do not find: ", candidate)
            #print("candidate:", can_token)

            #print("ori_docs: ", ori_doc)

            continue

        masked_tokens = masked_doc.split()
        masked_input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
        if model_type=='bert':
            len_masked_tokens = len(masked_tokens) - masked_tokens.count('[PAD]')
        else:
            len_masked_tokens = len(masked_tokens) - masked_tokens.count('<pad>')

        #print('len',len_masked_tokens)
        try:
            assert len(masked_input_ids) == MAX_LEN
        except:
            count +=1
            print("unmcatched: ", candidate)
            continue

        masked_attention_mask = np.zeros(MAX_LEN)
        masked_attention_mask[:len_masked_tokens] = 1
        masked_token_type_ids = np.zeros(MAX_LEN)
        masked_encode_dict = {
            "input_ids": torch.Tensor(masked_input_ids).to(torch.long),
            "token_type_ids": torch.Tensor(masked_token_type_ids).to(torch.long),
            "attention_mask": torch.Tensor(masked_attention_mask).to(torch.long),
            "candidate": candidate,
            "freq": len(match)
        }
        doc_pairs.append([ori_encode_dict, masked_encode_dict, idx])

    return doc_pairs, count


def get_PRF(num_c, num_e, num_s):
    F1 = 0.0
    P = float(num_c) / float(num_e) if num_e!=0 else 0.0
    R = float(num_c) / float(num_s) if num_s!=0 else 0.0
    if (P + R == 0.0):
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1


def print_PRF(P, R, F1, N):

    log.logger.info("\nN=" + str(N))
    log.logger.info("P=" + str(P))
    log.logger.info("R=" + str(R))
    log.logger.info("F1=" + str(F1))
    return 0

def mean_pooling(model_output, attention_mask):
    hidden_states = model_output.hidden_states
    token_embeddings = hidden_states[args.layer_num] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def max_pooling(model_output, attention_mask):
    hidden_states = model_output.hidden_states
    token_embeddings = hidden_states[args.layer_num]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]

def cls_emebddings(model_output):
    hidden_states = model_output.hidden_states
    token_embeddings = hidden_states[args.layer_num] #First element of model_output contains all token embeddings
    doc_embeddings = token_embeddings[:,0,:]
    return doc_embeddings


def keyphrases_selection_exec(path, list_of_names,   model, dataloader, k_val , log):

    model.eval()

    cos_similarity_list = {}
    candidate_list = []
    cos_score_list = []
    doc_id_list = []



    for id, [ori_doc, masked_doc, doc_id] in enumerate(tqdm(dataloader,desc="Executing:")):
        #print(ori_doc)
        ori_input_ids = torch.squeeze(ori_doc["input_ids"].to('cpu'),1)
        #ori_token_type_ids = ori_input_ids.clone()
        #ori_token_type_ids.zero_()
        #
        ori_attention_mask = torch.squeeze(ori_doc["attention_mask"].to('cpu'), 1)

        masked_input_ids = torch.squeeze(masked_doc["input_ids"].to('cpu'), 1)
        #masked_token_type_ids = ori_input_ids.clone()
        #masked_token_type_ids.zero_()
        #print(masked_token_type_ids)
        #masked_token_type_ids = torch.squeeze(masked_doc["token_type_ids"].to('cpu'), 1)
        masked_attention_mask = torch.squeeze(masked_doc["attention_mask"].to('cpu'), 1)
        candidate = masked_doc["candidate"]

        if model_type=='bert':
            ori_token_type_ids = torch.squeeze(ori_doc["token_type_ids"].to('cpu'), 1)
            masked_token_type_ids = torch.squeeze(masked_doc["token_type_ids"].to('cpu'), 1)


        # Predict hidden states features for each layer
        with torch.no_grad():
            # See the models docstrings for the detail of the inputs

            if model_type=='bert':
                ori_outputs = model(input_ids=ori_input_ids, attention_mask=ori_attention_mask, token_type_ids=ori_token_type_ids, output_hidden_states=True)
                masked_outputs = model(input_ids=masked_input_ids, attention_mask=masked_attention_mask, token_type_ids=masked_token_type_ids, output_hidden_states=True)
            else:
                ori_outputs = model(input_ids=ori_input_ids, attention_mask=ori_attention_mask,output_hidden_states=True)
                masked_outputs = model(input_ids=masked_input_ids, attention_mask=masked_attention_mask,output_hidden_states=True)
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            if args.doc_embed_mode == "mean":
                ori_doc_embed =mean_pooling(ori_outputs, ori_attention_mask)
                masked_doc_embed = mean_pooling(masked_outputs, masked_attention_mask)
            elif  args.doc_embed_mode == "cls":
                ori_doc_embed = cls_emebddings(ori_outputs)
                masked_doc_embed = cls_emebddings(masked_outputs)
            elif args.doc_embed_mode == "max":
                ori_doc_embed = max_pooling(ori_outputs, ori_attention_mask)
                masked_doc_embed = max_pooling(masked_outputs, masked_attention_mask)

            cosine_similarity = torch.cosine_similarity(ori_doc_embed, masked_doc_embed, dim=1).cpu()
            score = cosine_similarity
            doc_id_list.extend(doc_id.numpy().tolist())
            candidate_list.extend(candidate)
            cos_score_list.extend(score.numpy())

    cos_similarity_list["doc_id"] = doc_id_list
    cos_similarity_list["candidate"] = candidate_list
    cos_similarity_list["score"] = cos_score_list
    cosine_similarity_rank = pd.DataFrame(cos_similarity_list)

    for i in range(len(doc_list)):
        doc_results = cosine_similarity_rank.loc[cosine_similarity_rank['doc_id']==i]
        ranked_keyphrases = doc_results.sort_values(by='score')
        top_k = ranked_keyphrases.reset_index(drop = True)
        top_k_can = top_k.loc[:, ['candidate']].values.tolist()
        #print(top_k)

        candidates_set = set()
        candidates_dedup = []
        for temp in top_k_can:
            temp = temp[0].lower()
            if temp in candidates_set:
                continue
            else:
                candidates_set.add(temp)
                candidates_dedup.append(temp)


        #log.logger.info("Sorted_Candidate: {} \n".format(top_k_can))
        #log.logger.info("Candidates_Dedup: {} \n".format(candidates_dedup))

        j = 0
        Matched = candidates_dedup[:k_val]
        Name= list_of_names[i]
        write_results( Matched,os.path.join(path,str(Name)+'.key'))
        log.logger.info("TOP-K {}: {} \n".format(i, Matched))





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input dataset.")
    parser.add_argument("--dataset_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The input dataset name.")
    parser.add_argument("--doc_embed_mode",
                        default="mean",
                        type=str,
                        required=True,
                        help="The method for doc embedding.")
    parser.add_argument("--layer_num",
                        default=-1,
                        type=int,
                        help="The hidden state layer of BERT.")
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True,
                        help="Batch size for testing.")
    parser.add_argument("--checkpoints",
                        default=None,
                        type=str,
                        required=False,
                        help="Checkpoint for pre-trained Bert model")
    parser.add_argument("--log_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Path for Logging file")
    parser.add_argument("--model_type",
                        default='bert',
                        type=str,
                        help="Model type")

    parser.add_argument("--lang",
                        default='en',
                        type=str,
                        required=True,
                        help="language")

    parser.add_argument("--type_execution",
                        default='eval',
                        type=str,
                        required=True,
                        help="Type of execution: eval or exec")
    parser.add_argument("--k_value",
                        default='15',
                        type=str,
                        required=True,
                        help="K-elements to return")

    parser.add_argument("--model_name_or_path",
                        default='bert-uncased',
                        type=str,
                        help="model used")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus")

    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Whether not to use CUDA when available")
    args = parser.parse_args()

    start = time.time()
    log = Logger(args.log_dir + args.dataset_name + '.kpe.' + args.doc_embed_mode + '.log')


    k_val = int(args.k_value)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()




    porter=nltk.PorterStemmer()


    data = get_exec_dataset(args.dataset_dir)
    log.logger.info("Dataset")
    log.logger.info(data)






    global model_type
    model_type = args.model_type
    global model_name
    model_name = args.model_name_or_path
    global lang
    lang = args.lang
    global en_model




    generator = CandidatesGenerator(lang)



    if model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForMaskedLM.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)

    model.to(device)

    global MAX_LEN
    MAX_LEN=  tokenizer.model_max_length

    docs_pairs = []
    doc_list = []
    labels = []

    t_n = 0
    candidate_num = 0


    log.logger.info("Start Testing ...")

    log.logger.info("Max length del modelo: "+ str(MAX_LEN))




    list_of_names=[]

    ## EVALUATION??
    for idx, (key, doc) in enumerate(data.items()):

        doc = ' '.join(doc.split()[:512])
        list_of_names.append(key)
        doc_list.append(doc)

        # Statistic on empty docs
        empty_doc = 0
        try:
             generator.generate_candidates(doc)
        except:
            empty_doc += 1
            print("doc: ", doc)

        # Generate candidates (lower)
        cans = generator.keyphrase_candidate
        candidates = []
        for can, pos in cans:
            candidates.append(can.lower())
        candidate_num += len(candidates)

        if model_type=='roberta':
            doc=' '+doc.lower()
        ori_encode_dict = tokenizer.encode_plus(
            doc,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_LEN,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )

        doc_pairs, count = generate_absent_doc(ori_encode_dict, candidates, idx)
        docs_pairs.extend(doc_pairs)
        t_n +=count




    #print("candidate_num: ", candidate_num)
    #print("unmatched: ", t_n)



    dataset = KPE_Dataset(docs_pairs)
    #print("examples: ", dataset.total_examples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    keyphrases_selection_exec(args.dataset_dir, list_of_names,  model, dataloader,k_val, log)
    end = time.time()


    log.logger.info("Processing time: {}".format(end-start))
