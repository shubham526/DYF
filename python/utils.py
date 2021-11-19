import spacy
import re
from typing import List, Tuple
from spacy.tokens import Doc
import string
import gzip
import tqdm
from pykson import Pykson, JsonObject, StringField, IntegerField, ListField, ObjectListField, ObjectField, Pykson, \
    BooleanField
from object_models import Location, Entity, AnnotatedText, AspectLinkExample, Aspect, Context

class TextProcessor:
    def __init__(self, model='en_core_web_sm'):
        try:
            self._model = spacy.load(model, disable=["ner", "parser"])
        except OSError:
            print(f"Downloading spaCy model {model}")
            spacy.cli.download(model)
            print(f"Finished downloading model")
            self._model = spacy.load(model, disable=["ner", "parser"])

    @staticmethod
    def download_spacy_model(model="en_core_web_sm"):
        print(f"Downloading spaCy model {model}")
        spacy.cli.download(model)
        print(f"Finished downloading model")

    @staticmethod
    def load_model(model="en_core_web_sm"):
        return spacy.load(model, disable=["ner", "parser"])

    def preprocess(self, text: str) -> str:

        # Remove Unicode characters otherwise spacy complains
        text = text.encode("ascii", "ignore")
        text = text.decode()
        doc = self._model(text)

        # 1. Tokenize
        tokens = [token for token in doc]

        # 2. Remove numbers
        tokens = [token for token in tokens if not (token.like_num or token.is_currency)]

        # 3. Remove stopwords
        tokens = [token for token in tokens if not token.is_stop]

        # 4. Remove special tokens
        tokens = [token for token in tokens if
                  not (token.is_punct or token.is_space or token.is_quote or token.is_bracket)]
        tokens = [token for token in tokens if token.text.strip() != ""]

        # 5. Lemmatization
        text = " ".join([token.lemma_ for token in tokens])

        # 6. Remove non-alphabetic characters
        text = re.sub(r"[^a-zA-Z\']", " ", text)

        # 7. Remove non-Unicode characters
        text = re.sub(r"[^\x00-\x7F]+", "", text)

        # . 8. Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # 9. Lowercase
        text = text.lower()

        return text


def aspect_link_examples(json_file: str) -> AspectLinkExample:
    """
    Reads the JSON-L file in gzip format.
    Generates an AspectLinkExample in a lazy way (using yield).
    :param json_file: JSON-L file in gzip format.
    """
    with gzip.open(json_file, 'rt', encoding='UTF-8') as zipfile:
        for line in zipfile:
            example = Pykson().from_json(line, AspectLinkExample)
            yield example


def read_data(json_file: str, total: int) -> List[AspectLinkExample]:
    examples: List[str] = []
    with gzip.open(json_file, 'rt', encoding='UTF-8') as zipfile:
        for line in tqdm.tqdm(zipfile, total=total):
            example = Pykson().from_json(line, AspectLinkExample)
            examples.append(example)

    return examples


def get_negative_doc_list(candidate_aspects: List[Aspect], true_aspect: str) -> List[Tuple[str, str]]:
    processor = TextProcessor()
    return [
        (aspect.aspect_id, processor.preprocess(aspect.aspect_content.content))
        for aspect in candidate_aspects if aspect.aspect_id != true_aspect
    ]


def get_positive_doc(candidate_aspects: List[Aspect], true_aspect: str) -> str:
    processor = TextProcessor()
    doc_pos: str = ''
    for aspect in candidate_aspects:
        if aspect.aspect_id == true_aspect:
            doc_pos = processor.preprocess(aspect.aspect_content.content)
            break

    return doc_pos


