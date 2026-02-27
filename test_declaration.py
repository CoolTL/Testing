"""
test_declaration.py
===================
Unit tests for declaration.py
"""

import math
import unittest

from declaration import (
    FULL_TEXT,
    FrequencyAnalyzer,
    MarkovChain,
    NGramAnalyzer,
    ConceptGraph,
    SentimentAnalyzer,
    TextTokenizer,
    Token,
    NGram,
    SentimentResult,
    _PARAGRAPHS,
)


class TestTextTokenizer(unittest.TestCase):

    def setUp(self):
        self.tok = TextTokenizer(remove_stopwords=True)
        self.tok_raw = TextTokenizer(remove_stopwords=False)

    def test_words_returns_lowercase(self):
        words = self.tok.words("Freedom and Liberty")
        for w in words:
            self.assertEqual(w, w.lower())

    def test_stopwords_removed(self):
        words = self.tok.words("We hold these truths to be self-evident")
        for w in words:
            self.assertNotIn(w, {"we", "these", "to", "be"})

    def test_words_no_stopwords_removal(self):
        words = self.tok_raw.words("We hold these truths")
        self.assertIn("we", words)

    def test_sentences_splits_on_period(self):
        text = "Sentence one. Sentence two. Sentence three."
        sents = self.tok.sentences(text)
        self.assertGreaterEqual(len(sents), 2)

    def test_flat_tokens_returns_token_objects(self):
        tokens = self.tok.flat_tokens(_PARAGRAPHS)
        self.assertTrue(all(isinstance(t, Token) for t in tokens))

    def test_flat_tokens_para_ids_in_range(self):
        tokens = self.tok.flat_tokens(_PARAGRAPHS)
        ids = {t.para_id for t in tokens}
        self.assertTrue(ids.issubset(set(range(len(_PARAGRAPHS)))))

    def test_tokenize_paragraphs_length(self):
        result = self.tok.tokenize_paragraphs(_PARAGRAPHS)
        self.assertEqual(len(result), len(_PARAGRAPHS))

    def test_words_strips_punctuation(self):
        words = self.tok_raw.words("tyranny, oppression; injustice.")
        for w in words:
            self.assertNotIn(",", w)
            self.assertNotIn(";", w)
            self.assertNotIn(".", w)


class TestFrequencyAnalyzer(unittest.TestCase):

    def setUp(self):
        tok = TextTokenizer()
        self.fa = FrequencyAnalyzer(tok).fit(_PARAGRAPHS)

    def test_vocabulary_size_positive(self):
        self.assertGreater(self.fa.vocabulary_size(), 0)

    def test_top_terms_length(self):
        top = self.fa.top_terms(10)
        self.assertLessEqual(len(top), 10)

    def test_top_terms_sorted_descending(self):
        top = self.fa.top_terms(20)
        scores = [s for _, s in top]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_global_frequency_known_word(self):
        count = self.fa.global_frequency("laws")
        self.assertGreater(count, 0)

    def test_global_frequency_unknown_word(self):
        count = self.fa.global_frequency("xyzxyzxyz")
        self.assertEqual(count, 0)

    def test_top_terms_for_section_returns_list(self):
        terms = self.fa.top_terms_for_section(0, n=5)
        self.assertIsInstance(terms, list)
        self.assertLessEqual(len(terms), 5)

    def test_top_terms_for_section_out_of_range(self):
        with self.assertRaises(IndexError):
            self.fa.top_terms_for_section(999)


class TestNGramAnalyzer(unittest.TestCase):

    def setUp(self):
        tok = TextTokenizer()
        self.ng = NGramAnalyzer(tok).fit(_PARAGRAPHS)

    def test_top_bigrams_returns_ngrams(self):
        bigrams = self.ng.top_bigrams(10, min_count=2)
        for bg in bigrams:
            self.assertIsInstance(bg, NGram)
            self.assertEqual(len(bg.tokens), 2)

    def test_top_trigrams_returns_ngrams(self):
        trigrams = self.ng.top_trigrams(5, min_count=2)
        for tg in trigrams:
            self.assertIsInstance(tg, NGram)
            self.assertEqual(len(tg.tokens), 3)

    def test_bigram_pmi_is_finite_or_neginf(self):
        bigrams = self.ng.top_bigrams(10, min_count=2)
        for bg in bigrams:
            self.assertTrue(math.isfinite(bg.pmi) or bg.pmi == float("-inf"))

    def test_continuations_returns_sorted_list(self):
        # Find a word that has continuations
        bigrams = self.ng.top_bigrams(5, min_count=2)
        if bigrams:
            w = bigrams[0].tokens[0]
            conts = self.ng.continuations((w,))
            if conts:
                counts = [c for _, c in conts]
                self.assertEqual(counts, sorted(counts, reverse=True))

    def test_continuations_bad_prefix_raises(self):
        with self.assertRaises(ValueError):
            self.ng.continuations(("a", "b", "c"))


class TestConceptGraph(unittest.TestCase):

    def setUp(self):
        tok = TextTokenizer()
        words = [t.word for t in tok.flat_tokens(_PARAGRAPHS)]
        self.g = ConceptGraph(window=5).fit(words)

    def test_node_count_positive(self):
        self.assertGreater(self.g.node_count(), 0)

    def test_edge_count_positive(self):
        self.assertGreater(self.g.edge_count(), 0)

    def test_bfs_starts_with_given_node(self):
        order = self.g.bfs("laws")
        if order:
            self.assertEqual(order[0], "laws")

    def test_bfs_unknown_node_returns_empty(self):
        self.assertEqual(self.g.bfs("xyzxyz"), [])

    def test_dfs_starts_with_given_node(self):
        order = self.g.dfs("laws")
        if order:
            self.assertEqual(order[0], "laws")

    def test_dfs_no_duplicates(self):
        order = self.g.dfs("laws")
        self.assertEqual(len(order), len(set(order)))

    def test_shortest_path_same_node(self):
        path, cost = self.g.shortest_path("laws", "laws")
        self.assertEqual(path, ["laws"])
        self.assertEqual(cost, 0.0)

    def test_shortest_path_unknown_returns_empty(self):
        path, cost = self.g.shortest_path("laws", "xyzxyz")
        self.assertEqual(path, [])
        self.assertEqual(cost, float("inf"))

    def test_pagerank_sums_to_approx_one(self):
        pr = self.g.pagerank()
        total = sum(pr.values())
        self.assertAlmostEqual(total, 1.0, delta=0.05)

    def test_top_nodes_by_degree_length(self):
        top = self.g.top_nodes_by_degree(5)
        self.assertLessEqual(len(top), 5)


class TestMarkovChain(unittest.TestCase):

    def setUp(self):
        self.mc = MarkovChain(order=2, seed=42).fit(_PARAGRAPHS)

    def test_generate_returns_string(self):
        result = self.mc.generate(max_words=30)
        self.assertIsInstance(result, str)

    def test_generate_not_empty(self):
        result = self.mc.generate(max_words=30)
        self.assertGreater(len(result), 0)

    def test_generate_deterministic_with_seed(self):
        mc1 = MarkovChain(order=2, seed=7).fit(_PARAGRAPHS)
        mc2 = MarkovChain(order=2, seed=7).fit(_PARAGRAPHS)
        self.assertEqual(mc1.generate(50), mc2.generate(50))

    def test_generate_different_seeds_differ(self):
        mc1 = MarkovChain(order=2, seed=1).fit(_PARAGRAPHS)
        mc2 = MarkovChain(order=2, seed=2).fit(_PARAGRAPHS)
        # Very unlikely to be identical
        self.assertNotEqual(mc1.generate(50), mc2.generate(50))

    def test_bad_order_raises(self):
        with self.assertRaises(ValueError):
            MarkovChain(order=0)

    def test_perplexity_positive(self):
        ppl = self.mc.perplexity(FULL_TEXT[:200])
        self.assertGreater(ppl, 0)


class TestSentimentAnalyzer(unittest.TestCase):

    def setUp(self):
        self.sa = SentimentAnalyzer()

    def test_positive_text(self):
        result = self.sa.analyze("liberty freedom justice equal rights")
        self.assertGreater(result.polarity, 0)

    def test_negative_text(self):
        result = self.sa.analyze("tyranny despotism oppression murder")
        self.assertLess(result.polarity, 0)

    def test_neutral_text(self):
        result = self.sa.analyze("the cat sat on the mat")
        self.assertAlmostEqual(result.polarity, 0.0, delta=0.1)

    def test_analyze_paragraphs_length(self):
        results = self.sa.analyze_paragraphs(_PARAGRAPHS)
        self.assertEqual(len(results), len(_PARAGRAPHS))

    def test_polarity_in_range(self):
        for p in _PARAGRAPHS:
            r = self.sa.analyze(str(p["text"]))
            self.assertGreaterEqual(r.polarity, -1.0)
            self.assertLessEqual(r.polarity, 1.0)

    def test_subjectivity_in_range(self):
        for p in _PARAGRAPHS:
            r = self.sa.analyze(str(p["text"]))
            self.assertGreaterEqual(r.subjectivity, 0.0)
            self.assertLessEqual(r.subjectivity, 1.0)

    def test_str_representation(self):
        result = self.sa.analyze("liberty justice freedom")
        s = str(result)
        self.assertIn("POSITIVE", s)

    def test_counts_are_non_negative(self):
        result = self.sa.analyze("test sentence here")
        self.assertGreaterEqual(result.positive_count, 0)
        self.assertGreaterEqual(result.negative_count, 0)


class TestFullText(unittest.TestCase):

    def test_full_text_not_empty(self):
        self.assertGreater(len(FULL_TEXT), 0)

    def test_full_text_contains_key_phrase(self):
        self.assertIn("unalienable Rights", FULL_TEXT)

    def test_full_text_contains_declaration(self):
        self.assertIn("Free and Independent States", FULL_TEXT)

    def test_paragraphs_have_required_keys(self):
        for p in _PARAGRAPHS:
            self.assertIn("id", p)
            self.assertIn("section", p)
            self.assertIn("text", p)


if __name__ == "__main__":
    unittest.main()
