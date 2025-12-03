import re
import time
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer, RobertaTokenizer, RobertaForMaskedLM
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import logging
import codecs
import os
import nltk
import torch
from .CandidatesGenerator import CandidatesGenerator
from typing import List, Union, Dict
from pathlib import Path


class MDERankModel:
    """
    Carga y mantiene en memoria el modelo HuggingFace
    para evitar recargarlo al procesar múltiples datasets.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.candidatesGenerator= CandidatesGenerator(self.cfg.lang)


        if self.cfg.model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(self.cfg.model_name_or_path)
            self.model = RobertaForMaskedLM.from_pretrained(self.cfg.model_name_or_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model_name_or_path)
            self.model = BertForMaskedLM.from_pretrained(self.cfg.model_name_or_path)

        device = "cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu"

        self.model.to(device)
        self.device = device

        self.MAX_LEN = self.tokenizer.model_max_length

        self.porter = nltk.PorterStemmer()
        self.eval_mode = False








    def extract_terms(self, dataset, eval_dataset=None, k_values=15):


        # start = time.time()
        data, references = prepare_data(dataset,eval_dataset)

        self.cfg.logger.info(data)
        self.cfg.logger.info(references)
        self.results=[]

        if references is not None:
            self.eval_mode=True
        else:
            self.eval_mode=False


        docs_pairs = []
        doc_list = []
        labels = []
        labels_stemed = []
        t_n = 0
        candidate_num = 0

        # log.logger.info("Start Testing ...")

        # log.logger.info("Max length del modelo: " + str(self.MAX_LEN))

        for idx, (key, doc) in enumerate(data.items()):




            doc = ' '.join(doc.split()[:512])
            #print(doc)
            doc_list.append(doc)

            if self.eval_mode is True:
                # Get stemmed labels and document segments
                labels.append([ref.replace(" \n", "") for ref in references[key]])

                labels_s = []
                set_total_cans = set()
                for l in references[key]:
                    tokens = l.split()
                    labels_s.append(' '.join(self.porter.stem(t) for t in tokens))

                labels_stemed.append(labels_s)



            cans = self.candidatesGenerator.generate_candidates(doc)
            candidates = []
            for can, pos in cans:
                candidates.append(can.lower())
            candidate_num += len(candidates)

            self.cfg.logger.info("Candidates")
            self.cfg.logger.info(candidates)
            if self.cfg.model_type == 'roberta':
                doc = ' ' + doc.lower()
                ori_encode_dict = self.tokenizer.encode_plus(
                doc,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.MAX_LEN,  # Pad & truncate all sentences.
                padding='max_length',
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
                )

            doc_pairs, count = self.generate_absent_doc(ori_encode_dict, candidates, idx)
            docs_pairs.extend(doc_pairs)
            t_n += count

        # print("candidate_num: ", candidate_num)
        # print("unmatched: ", t_n)

        dataset = KPE_Dataset(docs_pairs)
        # print("examples: ", dataset.total_examples)
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size)

        self.keyphrases_selection(doc_list, labels_stemed, labels, dataloader)
        end = time.time()

        # log.logger.info("Processing time: {}".format(end - start))

    def keyphrases_selection(self, doc_list, labels_stemed, labels, dataloader):
        #print(labels)
        #print(labels_stemed)
        self.model.eval()

        cos_similarity_list = {}
        candidate_list = []
        cos_score_list = []
        doc_id_list = []

        P = R = F1 = 0.0
        num_c_5 = num_c_10 = num_c_15 = 0
        num_e_5 = num_e_10 = num_e_15 = 0
        num_s = 0
        lamda = 0.0

        for id, [ori_doc, masked_doc, doc_id] in enumerate(tqdm(dataloader, desc="Evaluating:")):
            # print(ori_doc)
            ori_input_ids = torch.squeeze(ori_doc["input_ids"].to('cpu'), 1)
            # ori_token_type_ids = ori_input_ids.clone()
            # ori_token_type_ids.zero_()
            #
            ori_attention_mask = torch.squeeze(ori_doc["attention_mask"].to('cpu'), 1)

            masked_input_ids = torch.squeeze(masked_doc["input_ids"].to('cpu'), 1)
            # masked_token_type_ids = ori_input_ids.clone()
            # masked_token_type_ids.zero_()
            # print(masked_token_type_ids)
            # masked_token_type_ids = torch.squeeze(masked_doc["token_type_ids"].to('cpu'), 1)
            masked_attention_mask = torch.squeeze(masked_doc["attention_mask"].to('cpu'), 1)
            candidate = masked_doc["candidate"]

            if self.cfg.model_type == 'bert':
                ori_token_type_ids = torch.squeeze(ori_doc["token_type_ids"].to('cpu'), 1)
                masked_token_type_ids = torch.squeeze(masked_doc["token_type_ids"].to('cpu'), 1)

            # Predict hidden states features for each layer
            with torch.no_grad():
                # See the models docstrings for the detail of the inputs

                if self.cfg.model_type == 'bert':
                    ori_outputs = self.model(input_ids=ori_input_ids, attention_mask=ori_attention_mask,
                                             token_type_ids=ori_token_type_ids, output_hidden_states=True)
                    masked_outputs = self.model(input_ids=masked_input_ids, attention_mask=masked_attention_mask,
                                                token_type_ids=masked_token_type_ids, output_hidden_states=True)
                else:
                    ori_outputs = self.model(input_ids=ori_input_ids, attention_mask=ori_attention_mask,
                                             output_hidden_states=True)
                    masked_outputs = self.model(input_ids=masked_input_ids, attention_mask=masked_attention_mask,
                                                output_hidden_states=True)
                # Transformers models always output tuples.
                # See the models docstrings for the detail of all the outputs
                # In our case, the first element is the hidden state of the last layer of the Bert model
                if self.cfg.doc_embed_mode == "mean":
                    ori_doc_embed = mean_pooling(ori_outputs, ori_attention_mask, self.cfg.layer_num)
                    masked_doc_embed = mean_pooling(masked_outputs, masked_attention_mask, self.cfg.layer_num)
                elif self.cfg.doc_embed_mode == "cls":
                    ori_doc_embed = cls_embeddings(ori_outputs, self.cfg.layer_num)
                    masked_doc_embed = cls_embeddings(masked_outputs, self.cfg.layer_num)
                elif self.cfg.doc_embed_mode == "max":
                    ori_doc_embed = max_pooling(ori_outputs, ori_attention_mask, self.cfg.layer_num)
                    masked_doc_embed = max_pooling(masked_outputs, masked_attention_mask, self.cfg.layer_num)

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
            doc_results = cosine_similarity_rank.loc[cosine_similarity_rank['doc_id'] == i]
            ranked_keyphrases = doc_results.sort_values(by='score')
            top_k = ranked_keyphrases.reset_index(drop=True)
            top_k_can = top_k.loc[:, ['candidate']].values.tolist()
            # print(top_k)

            candidates_set = set()
            candidates_dedup = []
            for temp in top_k_can:
                temp = temp[0].lower()
                if temp in candidates_set:
                    continue
                else:
                    candidates_set.add(temp)
                    candidates_dedup.append(temp)

            # log.logger.info("Sorted_Candidate: {} \n".format(top_k_can))
            # log.logger.info("Candidates_Dedup: {} \n".format(candidates_dedup))
            self.cfg.logger.info("Sorted_Candidate: {} \n".format(top_k_can))
            self.cfg.logger.info("Candidates_Dedup: {} \n".format(candidates_dedup))


            self.results.append(candidates_dedup)

            if self.eval_mode is False:
                continue
            j = 0
            Matched = candidates_dedup[:15]


            for id, temp in enumerate(candidates_dedup[0:15]):
                tokens = temp.split()
                tt = ' '.join(self.porter.stem(t) for t in tokens)
                if (tt in labels_stemed[i] or temp in labels[i]):
                    Matched[id] = [temp]
                    if (j < 5):
                        num_c_5 += 1
                        num_c_10 += 1
                        num_c_15 += 1

                    elif (j < 10 and j >= 5):
                        num_c_10 += 1
                        num_c_15 += 1

                    elif (j < 15 and j >= 10):
                        num_c_15 += 1
                j += 1



            if (len(top_k[0:5]) == 5):
                num_e_5 += 5
            else:
                num_e_5 += len(top_k[0:5])

            if (len(top_k[0:10]) == 10):
                num_e_10 += 10
            else:
                num_e_10 += len(top_k[0:10])

            if (len(top_k[0:15]) == 15):
                num_e_15 += 15
            else:
                num_e_15 += len(top_k[0:15])

            num_s += len(labels[i])

        if self.eval_mode is False:
            return

        # en_model
        p, r, f = get_PRF(num_c_5, num_e_5, num_s)
        print_PRF(p, r, f, 5, self.cfg.logger)
        p, r, f = get_PRF(num_c_10, num_e_10, num_s)
        print_PRF(p, r, f, 10,self.cfg.logger)
        p, r, f = get_PRF(num_c_15, num_e_15, num_s)
        print_PRF(p, r, f, 15,self.cfg.logger)

    def generate_absent_doc(self, ori_encode_dict, candidates, idx):
        count = 0
        doc_pairs = []
        ori_input_ids = ori_encode_dict["input_ids"].squeeze()
        ori_tokens = self.tokenizer.convert_ids_to_tokens(ori_input_ids)

        # There are multi candidates for a document
        for id, candidate in enumerate(candidates):

            # Remove stopwords in a candidate
            if remove(candidate):
                count += 1
                continue

            ####

            match = []
            try:
                if self.cfg.model_type == 'bert':
                    tok_candidate = self.tokenizer.tokenize(candidate)
                    match, masked_doc = find_candidate_mention(tok_candidate, ori_tokens, self.cfg.model_type)
                else:
                    ## roberta
                    tok_candidate = self.tokenizer.tokenize(' ' + candidate)
                    match, masked_doc = find_candidate_mention(tok_candidate, ori_tokens, self.cfg.model_type)
                    if len(match) == 0:
                        print("try again")
                        tok_candidate = self.tokenizer.tokenize(candidate)
                        match, masked_doc = find_candidate_mention(tok_candidate, ori_tokens, self.cfg.model_type)



            except:
                print("cannot replace: ", candidate)
                count += 1
                continue

            if len(match) == 0:
                count += 1
                ori_doc = ' '.join(ori_tokens)
                can_token = ' '.join(tok_candidate)
                # print("do not find: ", candidate)
                # print("candidate:", can_token)

                # print("ori_docs: ", ori_doc)

                continue

            masked_tokens = masked_doc.split()
            masked_input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            if self.cfg.model_type == 'bert':
                len_masked_tokens = len(masked_tokens) - masked_tokens.count('[PAD]')
            else:
                len_masked_tokens = len(masked_tokens) - masked_tokens.count('<pad>')

            # print('len',len_masked_tokens)
            try:
                assert len(masked_input_ids) == self.MAX_LEN
            except:
                count += 1
                print("unmcatched: ", candidate)
                continue

            masked_attention_mask = np.zeros(self.MAX_LEN)
            masked_attention_mask[:len_masked_tokens] = 1
            masked_token_type_ids = np.zeros(self.MAX_LEN)
            masked_encode_dict = {
                "input_ids": torch.Tensor(masked_input_ids).to(torch.long),
                "token_type_ids": torch.Tensor(masked_token_type_ids).to(torch.long),
                "attention_mask": torch.Tensor(masked_attention_mask).to(torch.long),
                "candidate": candidate,
                "freq": len(match)
            }
            doc_pairs.append([ori_encode_dict, masked_encode_dict, idx])

        return doc_pairs, count






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


def write_string(s, output_path):
    with open(output_path, 'w') as output_file:
        output_file.write(s)


def read_file(input_path):
    with open(input_path, 'r', errors='replace_with_space') as input_file:
        return input_file.read()


def clean_text_separating_punctuation(text: str) -> str:
    # Separar cualquier signo de puntuación con espacios
    # Grupo 1: signos antes de palabra
    # Grupo 2: signos después de palabra
    text = re.sub(r'([^\w\s])', r' \1 ', text)

    # Reemplazar múltiples espacios por uno
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
def clean_text(text: str = "") -> str:
    """
    Limpia texto proveniente de datasets como Inspec, Duc2001 o SemEval2017.
    Normaliza saltos de línea, espacios múltiples, elimina ciertos caracteres
    y corrige artefactos de formato específicos de algunos datasets.
    """

        # Sustituye (espacio|coma)+\n por un salto limpio
    pattern = re.compile(r'[\s,]\n')
    text = pattern.sub("\n", text)

    # 2. Sustituye [letra|número|coma|espacio]+\n por " "
    pattern = re.compile(r'([a-zA-Z0-9,\s])\n')
    text = pattern.sub(r'\1 ', text)

    # 3. Colapsa múltiples espacios a uno solo
    text = re.sub(r'\s{2,}', ' ', text)

    # 4. Elimina caracteres extraños
    text = re.sub(r'[<>[\]{}]', ' ', text)

    # 5. Reemplaza tabulaciones
    text = text.replace("\t", " ")

    # 6. Limpieza de tokens específicos
    text = text.replace(" p ", "\n")
    text = text.replace(" /p \n", "\n")

    # 7. Quita líneas completamente vacías
    lines = text.splitlines()
    cleaned = "\n".join(line for line in lines if line.strip())

    cleaned= clean_text_separating_punctuation(cleaned)
    return cleaned


def load_dataset(data_path: str) -> Dict[str, str]:
    """
    Carga los textos de SemEval2017 y los limpia usando clean_text.
    Devuelve un diccionario: {doc_id → texto_limpiado}.
    """
    texts = {}

    for dirname, _, filenames in os.walk(data_path):
        for fname in filenames:
            doc_id, _ = fname.split('.', maxsplit=1)
            infile = os.path.join(dirname, fname)

            with codecs.open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", "")

            cleaned = clean_text(text)
            texts[doc_id] = cleaned.lower()

    return texts


def load_key_dataset(labels_path: str) -> Dict[str, List[str]]:
    """
    Carga las keywords de SemEval2017 y elimina saltos de línea y espacios extra.
    Devuelve {doc_id → [keyword1, keyword2, ...]}.
    """
    labels = {}

    for dirname, _, filenames in os.walk(labels_path):
        for fname in filenames:
            doc_id, _ = fname.split('.', maxsplit=1)
            infile = os.path.join(dirname, fname)

            with open(infile, "rb") as f:
                content = f.read().decode("utf-8")

            # Limpieza: quitar espacios, líneas vacías, \n innecesarios
            keywords = [
                kw.strip()
                for kw in content.splitlines()
                if kw.strip()
            ]

            labels[doc_id] = keywords

    return labels


def remove(text):
    # print(text)
    text_len = len(text.split())
    remove_chars = '[’!"#$%&\'()*+,./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, '', text)
    re_text_len = len(text.split())
    if text_len != re_text_len:
        # print(text)
        return True
    else:
        return False


def find_candidate_mention(tok_candidate, ori_tokens, model_type):
    candidate_len = len(tok_candidate)
    ori_doc = ' '.join(ori_tokens)
    can_token = ' '.join(tok_candidate)

    conteo = can_token.count('Ġ')
    # print(tok_candidate,conteo)
    if model_type == 'bert':
        mask = ' '.join(['[MASK]'] * candidate_len)
    else:
        mask = ' '.join(['<mask>'] * candidate_len)

    candidate_re = re.compile(r"\b" + can_token + r"\b")

    masked_doc = re.sub(candidate_re, mask, ori_doc)
    match = candidate_re.findall(ori_doc)
    # print('>', ori_doc)
    # print('>', masked_doc)
    # print('len match >',len(match), match)

    return match, masked_doc


def get_PRF(num_c, num_e, num_s):
    F1 = 0.0
    P = float(num_c) / float(num_e) if num_e != 0 else 0.0
    R = float(num_c) / float(num_s) if num_s != 0 else 0.0
    if (P + R == 0.0):
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1


def print_PRF(P, R, F1, N,logger):
    logger.info("TOP-K {}: P {} R {} F1  {} ".format(N, P,R,F1))
    print(P,R,F1,N)
    """
    log.logger.info("\nN=" + str(N))
    log.logger.info("P=" + str(P))
    log.logger.info("R=" + str(R))
    log.logger.info("F1=" + str(F1))
    """
    return 0


def mean_pooling(model_output, attention_mask, layer_num):
    hidden_states = model_output.hidden_states
    token_embeddings = hidden_states[layer_num]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def max_pooling(model_output, attention_mask, layer_num):
    hidden_states = model_output.hidden_states
    token_embeddings = hidden_states[layer_num]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]


def cls_embeddings(model_output, layer_num):
    hidden_states = model_output.hidden_states
    token_embeddings = hidden_states[layer_num]  # First element of model_output contains all token embeddings
    doc_embeddings = token_embeddings[:, 0, :]
    return doc_embeddings


def prepare_data(dataset,eval_dataset):

    data= {}
    references= {}
    if isinstance(dataset, str):
        data = load_dataset(dataset)

        # Si es Path → leer archivo
        # Caso 1: dataset es lista → generar documentos
    if isinstance(dataset, list):
        for i, text in enumerate(dataset):
            data[f"doc_{i}"] = clean_text(text).lower()

    if eval_dataset is None:
        return data, None

    if isinstance(eval_dataset, str):
        references = load_key_dataset(eval_dataset)

        # Si es Path → leer archivo
        # Caso 1: dataset es lista → generar documentos
    if isinstance(eval_dataset, list):
        for i, text in enumerate(eval_dataset):
            eval_dataset[f"doc_{i}"] = text

    return data, references