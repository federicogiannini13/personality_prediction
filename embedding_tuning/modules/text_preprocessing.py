import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys

sys.path.insert(0, "../../")


class TextPreProcessing:
    """
    Class that, giving a text, does the preprocessing operations.
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words("english")
        self.modify_stop_words()

    def lemmatize(self, sentence):
        """
        Process the gived sentence. It follows these steps:
        1) String tokenization
        2) String lemmatization
        3) Strings filtering: remove stopwords, punctuation, digits

        Parameters
        ----------
        sentence: str
            sentence to be lemmatized.

        Returns
        -------
        lemmatized_words: list
            a list of string containing the lemmatized tokens.

        """
        sentence_words = nltk.word_tokenize(sentence)
        pos = nltk.pos_tag(sentence_words)
        pos_conv = self.convert_pos_tags(pos)
        sentence_words_corr = self.correct_words(sentence_words, pos_conv)
        pos_conv = self.correct_pos(pos, pos_conv, sentence_words_corr)
        lemmatized_words = []
        i = 0
        for w in sentence_words:
            if w not in self.stop_words:
                if pos_conv[i] == "no_lem" or pos_conv[i] == "ignore":
                    w_l = w
                else:
                    w_l = self.lemmatizer.lemmatize(
                        sentence_words_corr[i], pos=pos_conv[i]
                    )
                w_l = self.clean_token(w_l)
                if w_l not in self.stop_words and self.check_token(w_l):
                    lemmatized_words.append(w_l)
            i += 1
        return lemmatized_words

    def modify_stop_words(self):
        self.stop_words.pop(self.stop_words.index("not"))
        to_add = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ".",
            ",",
            ":",
            ";",
            '"',
            "/",
            "(",
            ")",
            "?",
            "!",
            "...",
            "@",
            "-",
            "",
            None,
        ]
        for w in to_add:
            self.stop_words.append(w)

    def correct_pos(self, pos, pos_conv, words):
        i = 0
        while i < len(words) - 1:
            if (
                pos_conv[i] != "ignore"
                and pos_conv[i] != "no_lem"
                and self.lemmatizer.lemmatize(words[i], pos_conv[i]) == "be"
                and pos[i + 1][1] == "VBG"
            ):
                pos_conv[i + 1] = "a"
                i += 1
            i += 1
        return pos_conv

    def check_token(self, t):
        digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        for d in digits:
            if d in t:
                return False
        if len(t) < 3:
            return False
        return True

    def clean_token(self, t):
        if t[-1:] == "-" or t[-1:] == "/":
            t = t[:-1]
        t = self.normalize(t)
        return (
            t.replace("'", "")
            .replace("$", "")
            .replace("&", "")
            .replace("`", "")
            .replace(".", "")
        )

    def normalize(self, t):
        i = 0
        remove = []
        while i < len(t):
            x = t[i : i + 1]
            j = 0
            while t[i + j + 1 : i + j + 2] == x:
                j += 1
            if j > 1:
                remove.append((i, i + j + 1))
            i += j + 1
        t_ = ""
        i = 0
        for r in remove:
            x = t[r[0] : r[0] + 1]
            t_ += t[i : r[0]] + x
            if (
                r[1] < len(t)
                and x != "a"
                and x != "e"
                and x != "i"
                and x != "o"
                and x != "u"
            ):
                t_ += x
            i = r[1]
        if i < len(t):
            t_ = t_ + t[i:]
        return t_

    def correct_words(self, words, pos):
        words_c = []
        i = 0
        for w in words:
            words_c.append(self.correct_word(w, pos[i]))
            i += 1
        return words_c

    def correct_word(self, w, pos):
        w = w.replace("n't", "not")
        if pos == "v":
            if w == "'s" or w == "'m":
                w = "be"
            elif w == "'ve":
                w = "have"
        return w

    def pos(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        pos = self.convert_pos_tags(nltk.pos_tag(sentence_words))
        return pos

    def pos_original(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        return nltk.pos_tag(sentence_words)

    def convert_pos_tags(self, pos):
        pos_ = []
        for p in pos:
            pos_.append(self.convert_pos_tag(p))
        return pos_

    def convert_pos_tag(self, p):
        p = p[1]
        if p == "FW":
            p = "no_lem"
        elif p == "JJR" or p == "JJS":
            p = "a"
        elif p[0:2] == "JJ":
            p = "s"
        elif p[0:2] == "NN":
            p = "n"
        elif p[0:2] == "RB":
            p = "r"
        elif p == "VBN":
            p = "ignore"
        elif p[0:2] == "VB":
            p = "v"
        else:
            p = "ignore"
        return p
