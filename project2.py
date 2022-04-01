import sys

import requests
import spacy
from bs4 import BeautifulSoup
from googleapiclient.discovery import build


from spacy_help_functions import create_entity_pairs
from spanbert import SpanBERT

nlp = spacy.load("en_core_web_lg")

# Load pre-trained SpanBERT model
spanbert = SpanBERT("SpanBERT/pretrained_spanbert")

relations = {
    1: 'Schools_Attended',
    2: 'Work_For',
    3: 'Live_In',
    4: 'Top_Member_Employees'
}
relations_internal = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}
relations_list = {
    1: ['PERSON', 'ORGANIZATION'],
    2: ['PERSON', 'ORGANIZATION'],
    3: ['PERSON', 'LOCATION', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY'],
    4: ['PERSON', 'ORGANIZATION']
}
named_entity_types = {
    'Schools_Attended': {'Subject': ['PERSON'], 'Object': ['ORGANIZATION']},
    'Work_For': {'Subject': ['PERSON'], 'Object': ['ORGANIZATION']},
    'Live_In': {'Subject': ['PERSON'], 'Object': ['LOCATION', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY']},
    'Top_Member_Employees': {'Subject': ['ORGANIZATION'], 'Object': ['PERSON']}
}

class ISE:

    def __init__(self):
        self.apikey = ''
        self.engineId = ''
        self.relation = 0
        self.threshold = 0
        self.query = ''

        self.tuples = []
        self.visited_urls = set()
        self.urls = set()
        self.tuples_dict = {}

    #this is the same google search function used in project 1
    def googleSearch(self):
        service = build("customsearch", "v1",
                        developerKey=self.apikey)

        res = service.cse().list(
            q=self.query,
            cx=self.engineId,
        ).execute()
        self.urls = set()
        for item in res['items']:
            if item['link'] in self.visited_urls:
                continue
            self.urls.add(item['link'])

    #this function extracts text from url using beautifulsoup
    def extractText(self):
        i = 1
        for url in self.urls:
            self.visited_urls.add(url)
            print('URL ( {} / {}): {}'.format(
                i,
                len(self.urls),
                url
            ))

            i += 1
            try:
                html = requests.get(url, timeout=30).content
                htmlParse = BeautifulSoup(html, 'html.parser')

                text = htmlParse.find_all(text=True)

                output = ''
                blacklist = [
                    '[document]',
                    'noscript',
                    'header',
                    'html',
                    'meta',
                    'head',
                    'input',
                    'script',
                    'link',
                    'style'
                ]

                for t in text:
                    if t.parent.name not in blacklist:
                        output += '{} '.format(t)

                print('Fetching text from url ...')
                if len(output) > 20000:
                    print('Trimming webpage content from {} to 20000 characters'.format(len(output)))
                    output = output[0:20000]

                print('Webpage length (num characters): {}'.format(len(output)))
                print('Annotating the webpage using spacy...')

                output.strip()
                self.extract_tuples(output)
            except:
                print('Timeout reached, trying next URL')

    #this function uses spacy and spanbert to extract final relations
    def extract_tuples(self, raw_text):

        # get entities of interest based on the relation given as input
        entities_of_interest = relations_list[self.relation]
        relation_name = relations[self.relation]
        internal_name = relations_internal[self.relation]

        # Apply spacy model to raw text (to split to sentences, tokenize, extract entities etc.)
        doc = nlp(raw_text)
        sentences_len = len([s for s in doc.sents])
        print(
            'Extracted {} sentences. Processing each sentence one by one to check for presence of right pair of named '
            'entity types; if so, will run the second pipeline ...'.format(
                sentences_len))
        ctr = 0
        for sentence in doc.sents:
            ctr += 1
            if ctr % 5 == 0:
                print("Processed {} / {} sentences".format(ctr, sentences_len))
            candidate_pairs = []
            sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)

            # based on the input relation, get the subject and the object values list that are required
            req_subject_list = named_entity_types[relation_name]['Subject']
            req_object_list = named_entity_types[relation_name]['Object']

            for ep in sentence_entity_pairs:
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
                candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject
            # Classify Relations for all Candidate Entity Pairs using SpanBERT
            candidate_pairs = [p for p in candidate_pairs if
                               (p["subj"][1] in req_subject_list and
                                p["obj"][1] in req_object_list)]

            # In case there are no candidate pairs, then stop the process
            if len(candidate_pairs) == 0:
                continue

            relation_preds = spanbert.predict(candidate_pairs)  # get predictions: list of (relation, confidence) pairs

            for ex, pred in list(zip(candidate_pairs, relation_preds)):

                if pred[0] == internal_name:
                    print('=== Extracted Relation ===')
                    print('Input tokens : {}'.format([token.text for token in sentence]))

                    print("Output Confidence: {} ; Subject: {} ; Object: {} ;".format(pred[1], ex["subj"][0],
                                                                                      ex["obj"][0]))

                    # check if the confidence of the extracted tuple is greater than the threshold
                    if pred[1] > self.threshold:
                        value = ex["subj"][0] + ' ' + ex["obj"][0]
                        # check whether the tuple is already present, if not then directly add the new entry
                        if not self.tuples_dict.__contains__(value):
                            self.tuples.append({'Subject': ex["subj"][0]
                                                   , 'Object': ex["obj"][0]
                                                   , 'Relation': pred[0]
                                                   , 'Confidence': pred[1]
                                                   , 'Value': value
                                                   , 'Used': False})
                            self.tuples_dict[value] = pred[1]
                        #     If the tuple is already present then check whether it has a better confidence.
                        #     If yes, then replace the old tuple with the new one
                        elif self.tuples_dict.__contains__(value) and pred[1] > self.tuples_dict[value]:
                            for tuple in self.tuples:
                                if tuple['Value'] == value:
                                    tuple['Confidence'] = pred[1]
                                    break

                            self.tuples_dict[value] = pred[1]
                            print('Adding to set of extracted relations')
                        else:
                            print('Duplicate with lower confidence than existing record. Ignoring this')
                    else:
                        print('Confidence is lower than threshold confidence. Ignoring this.')

                    print('==================')

    # This function sorts the list of tuples in decreasing order of the confidence.
    def sortTuples(self):
        def comp(elem):
            return -elem['Confidence']

        self.tuples = sorted(self.tuples, key=comp)

    # this function finds the next tuple to be used as a query for the next iteration
    def findNext(self):
        op = None
        for row in self.tuples:
            if row['Used']:
                continue
            op = row['Value']
            row['Used'] = True
            break

        return op

    #this function is used to print tuples
    def printOp(self):
        print('================== ALL RELATIONS for {} ( {} ) ================='.format(
            relations_internal[self.relation],
            len(self.tuples)
        ))
        for row in self.tuples:
            print("Confidence: {}\t| Subject: {}| \tObject: {}".format(row['Confidence'],
                                                                       row['Subject'],
                                                                       row['Object']))


if __name__ == '__main__':
    ise = ISE()
    ise.apikey = sys.argv[1]
    ise.engineId = sys.argv[2]
    ise.relation = int(sys.argv[3])
    ise.threshold = float(sys.argv[4])
    ise.query = sys.argv[5]
    ise.query = ise.query.replace('"', '')  # remove quotes if present
    k = int(sys.argv[6])
    ise.tuples_dict[ise.query] = 1.00
    itr = 0
    print('Parameters:')
    print('Client key\t= ' + ise.apikey)
    print('Engine key\t= ' + ise.engineId)
    print('Relation\t= ' + relations_internal[ise.relation])
    print('Threshold\t= ' + str(ise.threshold))
    print('Query\t= ' + ise.query)
    print('# of Tuples\t= ' + str(k))
    print('=========== Iteration: {} - Query: {} ==========='.format(itr, ise.query))
    itr += 1
    while len(ise.tuples) < k:
        ise.googleSearch()
        ise.extractText()
        ise.sortTuples()
        ise.printOp()
        local_query = ise.findNext()
        if local_query is None:
            print('ISE has stalled before retrieving k high-confidence tuples')
            break
        if len(ise.tuples) >= k:
            print('Total # of iterations = ' + str(itr))
            break
        else:
            print('=========== Iteration: {} - Query: {} ==========='.format(itr, local_query))

        ise.query = local_query
        itr += 1
