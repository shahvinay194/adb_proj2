import spacy
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs

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


# entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]

# Load spacy model
# nlp = spacy.load("en_core_web_lg")
#
# # Load pre-trained SpanBERT model
# spanbert = SpanBERT("./pretrained_spanbert")


def example_helper(raw_text, r_number, threshold, nlp, spanbert):
    # raw_text = "Zuckerberg attended Harvard University, where he launched the Facebook social networking service from his dormitory room on February 4, 2004, with college roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz, and Chris Hughes. Bill Gates stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella. "

    # # TODO: filter entities of interest based on target relation
    # entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    entities_of_interest = relations_list[r_number]
    relation_name = relations[r_number]
    internal_name = relations_internal[r_number]
    extracted_relations = []
    #
    # # Load spacy model
    # nlp = spacy.load("en_core_web_lg")
    #
    # # Load pre-trained SpanBERT model
    # spanbert = SpanBERT("./pretrained_spanbert")

    # Apply spacy model to raw text (to split to sentences, tokenize, extract entities etc.)
    doc = nlp(raw_text)
    print(
        'Extracted {} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...'.format(
            len(doc.sents)))
    for sentence in doc.sents:
        # print("\n\nProcessing sentence: {}".format(sentence))
        # print("Tokenized sentence: {}".format([token.text for token in sentence]))
        ents = get_entities(sentence, entities_of_interest)
        # print("spaCy extracted entities: {}".format(ents))

        # create entity pairs
        candidate_pairs = []
        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)

        req_subject_list = named_entity_types[relation_name]['Subject']
        req_object_list = named_entity_types[relation_name]['Object']

        # for ep in sentence_entity_pairs:
        #     # TODO: keep subject-object pairs of the right type for the target relation (e.g., Person:Organization for the "Work_For" relation)
        #     if ep[1][1] in req_subject_list and ep[2][1] in req_object_list:
        #         candidate_pairs.append({"tokens": ep[0][0], "subj": ep[1], "obj": ep[2][1]})  # e1=Subject, e2=Object
        #     if ep[2][1] in req_subject_list and ep[1][1] in req_object_list:
        #         candidate_pairs.append({"tokens": ep[0][0], "subj": ep[2][1], "obj": ep[1][1]})  # e1=Object, e2=Subject
        # candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject
        for ep in sentence_entity_pairs:
            # TODO: keep subject-object pairs of the right type for the target relation (e.g., Person:Organization for the "Work_For" relation)
            candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
            candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject
        # Classify Relations for all Candidate Entity Pairs using SpanBERT
        candidate_pairs = [p for p in candidate_pairs if
                           (p["subj"][1] in req_subject_list and p["obj"][
                               1] in req_object_list)]  # ignore subject entities with date/location type
        print("Candidate entity pairs:")
        for p in candidate_pairs:
            print("Subject: {}\tObject: {}".format(p["subj"][0:2], p["obj"][0:2]))
        print("Applying SpanBERT for each of the {} candidate pairs. This should take some time...".format(
            len(candidate_pairs)))

        if len(candidate_pairs) == 0:
            continue

        relation_preds = spanbert.predict(candidate_pairs)  # get predictions: list of (relation, confidence) pairs

        # Print Extracted Relations

        # if len(relation_preds) > 0:
        #     print("\nExtracted relations:")
        for ex, pred in list(zip(candidate_pairs, relation_preds)):

            if pred[0] == internal_name:
                print('=== Extracted Relation ===')
                print('Input tokens : {}'.format([token.text for token in sentence]))

                print("Output Confidence: {} ; Subject: {} ; Object: {} ;".format(pred[1], ex["subj"][0], ex["obj"][0]))

                if pred[1] > threshold:

                    extracted_relations.append({'Subject': ex["subj"][0]
                                               , 'Object': ex["obj"][0]
                                               , 'Relation': pred[0]
                                               , 'Confidence': pred[1]})
                    print('Adding to set of extracted relations')
                else:
                    print('Duplicate with lower confidence than existing record. Ignoring this.')


        return extracted_relations

        # TODO: focus on target relations
        # '1':"per:schools_attended"
        # '2':"per:employee_of"
        # '3':"per:cities_of_residence"
        # '4':"org:top_members/employees"
