
import spacy
class CandidatesGenerator:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self,lang):

        if lang == 'en':
            self.model = spacy.load("en_core_web_sm")
        if lang == 'es':
            self.model = spacy.load("es_core_news_sm")
        self.keyphrase_candidate=[]
        self.lang=lang


    def generate_candidates(self,text):


        candidates = []
        doc = self.model(text)
        for sent in doc.sents:

            doc2= self.model(sent.text)
            print(sent.text)
            sintagmas = [chunk.text for chunk in doc2.noun_chunks]
            #resultado.append((sent.text.strip(), sintagmas))
            # print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
            for chunk in sintagmas:
                print(chunk)
                chunk_processed = remove_starting_articles(chunk,self.lang)
                # chunk_processed = chunk_processed.lower()
                if len(chunk_processed) < 2:
                    continue
                candidates.append([chunk_processed, 0])
        '''
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}

        self.tokens = []
        self.tokens_tagged = []
        self.tokens = en_model.word_tokenize(text)
        self.tokens_tagged = en_model.pos_tag(text)
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "IN")

        '''
        print(candidates)

        self.keyphrase_candidate = candidates  # extract_candidates(self.tokens_tagged, en_model)

def remove_starting_articles(text,lang):
    # Lista de artículos a eliminar
    articles=[]
    if lang== 'es':
        articles = ['la ', 'el ', 'un ','una ','unos ','unas ', 'los ', 'las ','esta ','este ','estos ','estas ','cada ']
    else:
        articles = ['a ', 'the ', 'an ', 'this ', 'those ', 'that ', 'which ','every ']
    text_low= text

    # Iterar sobre cada artículo
    for article in articles:
        # Si el texto comienza con el artículo, quitarlo

        if text_low.lower().startswith(article):
            text = text[len(article):]  # Quitar el artículo

    return text


'''
stopword_dict = set(stopwords.words('english'))

GRAMMAR1 = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR2 = """  NP:
        {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR3 = """  NP:
        {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""


def extract_candidates(tokens_tagged, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """
    np_parser = nltk.RegexpParser(GRAMMAR1)  # Noun phrase parser
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1

    return keyphrase_candidate


'''
