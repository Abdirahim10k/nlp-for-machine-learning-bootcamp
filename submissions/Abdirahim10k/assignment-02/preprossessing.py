import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk

T1 = "URGENT!!! You have WON a £1,000 gift card. Call 0712345678 NOW!"
T2 = "I don't think this jacket is worth the price. Not happy at all 😞"
T3 = "Mr. Smith arrived in London on Friday. He met the Apple team at 9 a.m."
T4 = "   The   delivery   was   fast,   and the  packaging   was   great.   "

WORDS = [
    "connection", "connected", "connecting", "organization", "organized",
    "university", "universal", "flies", "wolves", "better", "ran"
]

NEGATION_KEEP = {"not", "n't", "no", "never"}


def main():
    # Task 1: Lowercasing
    print("=== Task 1: Lowercasing ===")
    t1_lower = T1.lower()
    print("Before:", T1)
    print("After: ", t1_lower)
    print()

    # Task 2: Whitespace
    print("=== Task 2: Whitespace Normalization ===")
    t4_clean_ws = re.sub(r"\s+", " ", T4).strip()
    print("Before:", repr(T4))
    print("After: ", repr(t4_clean_ws))
    print()

    # Task 3: Regex cleaning
    print("=== Task 3: Regex Cleaning ===")
    t1_deleted = re.sub(r"[^a-z\s]", "", t1_lower)
    deleted_chars = re.findall(r"[^a-z\s]", t1_lower)
    print("Before (lowercased T1):", t1_lower)
    print("After [^a-z\s]:        ", t1_deleted)
    print("Deleted characters:     ", deleted_chars)
    print()

    t1_replaced = re.sub(r"\b0\d{9,10}\b", "PHONE_NUMBER", T1)
    t1_replaced = re.sub(r"[£$€]\s?\d[\d,]*(?:\.\d+)?", "MONEY", t1_replaced)
    t1_replaced = re.sub(r"!{3,}", "!", t1_replaced)
    print("Before (original T1):   ", T1)
    print("After custom regex rules:", t1_replaced)
    print()

    # Task 4: Tokenization
    print("=== Task 4: Tokenization ===")
    t2_split = T2.split()
    t2_wtok = word_tokenize(T2)
    print("T2 split():         ", t2_split, "| Count:", len(t2_split))
    print("T2 word_tokenize(): ", t2_wtok, "| Count:", len(t2_wtok))

    t4_tokens = word_tokenize(t4_clean_ws)
    print("T4 word_tokenize(): ", t4_tokens)
    print("T4 token count:     ", len(t4_tokens))
    print("T4 vocabulary size: ", len(set(t4_tokens)), "| Unique tokens:", sorted(set(t4_tokens)))
    print()

    # Task 5: Stopwords and negation
    print("=== Task 5: Stopwords and Negation ===")
    default_stops = set(stopwords.words("english"))
    custom_stops = default_stops - NEGATION_KEEP
    t2_lower_tokens = [tok.lower() for tok in t2_wtok]
    t2_rm_default = [tok for tok in t2_lower_tokens if tok not in default_stops]
    t2_rm_custom = [tok for tok in t2_lower_tokens if tok not in custom_stops]

    print("Before (lowercased T2 tokens):", t2_lower_tokens)
    print("After default stopwords:      ", t2_rm_default)
    print("After negation-keeping stops: ", t2_rm_custom)
    print()

    # Task 6: Stemming
    print("=== Task 6: Stemming ===")
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer("english")
    print(f"{\'Word\':<15} {\'Porter\':<15} {\'Lancaster\':<15} {\'Snowball\':<15}")
    for w in WORDS:
        print(f"{w:<15} {porter.stem(w):<15} {lancaster.stem(w):<15} {snowball.stem(w):<15}")
    print()

    # Task 7: Lemmatization
    print("=== Task 7: Lemmatization ===")
    lemmatizer = WordNetLemmatizer()
    pos_map = {
        "connection": "n", "connected": "v", "connecting": "v",
        "organization": "n", "organized": "v", "university": "n",
        "universal": "a", "flies": "n", "wolves": "n",
        "better": "a", "ran": "v"
    }
    print(f"{\'Word\':<15} {\'Default\':<15} {\'POS\':<6} {\'With POS\':<15}")
    for w in WORDS:
        p = pos_map[w]
        print(f"{w:<15} {lemmatizer.lemmatize(w):<15} {p:<6} {lemmatizer.lemmatize(w, pos=p):<15}")
    print()

    # Task 8: Sentence splitting
    print("=== Task 8: Sentence Splitting ===")
    sents = sent_tokenize(T3)
    print("Before:", T3)
    print("Sentence count:", len(sents))
    for idx, s in enumerate(sents, 1):
        print(f"  [{idx}] {s}")
    print()

    # Task 9: POS tagging and NER
    print("=== Task 9: POS Tagging and NER ===")
    t3_tokens = word_tokenize(T3)
    tags = pos_tag(t3_tokens)
    tree = ne_chunk(tags)
    print("POS tags:", tags)
    print("NER tree:")
    print(tree.pformat())
    for subtree in tree:
        if hasattr(subtree, "label"):
            print("Entity:", " ".join(c[0] for c in subtree.leaves()), "| Label:", subtree.label())
    print()

    # Task 10: Full pipeline
    print("=== Task 10: Full Pipeline ===")
    def pipeline(text, stop_set):
        step1 = text.lower()
        print("  [lowercase]:", step1)
        step2 = re.sub(r"[^a-z0-9\'\s]", " ", step1)
        step2 = re.sub(r"\s+", " ", step2).strip()
        print("  [regex]:    ", step2)
        step3 = word_tokenize(step2)
        print("  [tokenize]: ", step3)
        step4 = [t for t in step3 if t not in stop_set]
        print("  [stopwords]:", step4)
        step5 = [lemmatizer.lemmatize(t) for t in step4]
        print("  [lemmatize]:", step5)
        return step5

    print("Run 1 (Default NLTK stopwords):")
    pipeline(T2, default_stops)
    print("Run 2 (Negation-keeping stopwords):")
    pipeline(T2, custom_stops)


if __name__ == "__main__":
    main()
