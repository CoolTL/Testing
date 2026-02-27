"""
declaration.py
==============
An extremely complex analysis engine for the United States Declaration of Independence.

Features
--------
* Full text stored as structured paragraphs with metadata
* TextTokenizer  – sentence & word tokenization with stop-word removal
* FrequencyAnalyzer – term frequency, inverse-document frequency (TF-IDF)
* NGramAnalyzer – bigram / trigram extraction and scoring (PMI)
* ConceptGraph – weighted undirected co-occurrence graph (BFS / DFS / shortest path)
* MarkovChain – variable-order Markov text generator
* SentimentAnalyzer – lexicon-based polarity & subjectivity scorer
* CLIParser – command-line dispatch table
* ReportGenerator – renders a formatted analysis report
"""

from __future__ import annotations

import collections
import heapq
import itertools
import math
import random
import re
import string
import sys
import textwrap
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

# ---------------------------------------------------------------------------
# 1. RAW TEXT
# ---------------------------------------------------------------------------

_PARAGRAPHS: List[Dict[str, object]] = [
    {
        "id": 0,
        "section": "preamble",
        "text": (
            "The unanimous Declaration of the thirteen united States of America, "
            "When in the Course of human events, it becomes necessary for one people "
            "to dissolve the political bands which have connected them with another, "
            "and to assume among the powers of the earth, the separate and equal "
            "station to which the Laws of Nature and of Nature's God entitle them, "
            "a decent respect to the opinions of mankind requires that they should "
            "declare the causes which impel them to the separation."
        ),
    },
    {
        "id": 1,
        "section": "philosophy",
        "text": (
            "We hold these truths to be self-evident, that all men are created equal, "
            "that they are endowed by their Creator with certain unalienable Rights, "
            "that among these are Life, Liberty and the pursuit of Happiness.--That "
            "to secure these rights, Governments are instituted among Men, deriving "
            "their just powers from the consent of the governed, --That whenever any "
            "Form of Government becomes destructive of these ends, it is the Right of "
            "the People to alter or to abolish it, and to institute new Government, "
            "laying its foundation on such principles and organizing its powers in "
            "such form, as to them shall seem most likely to effect their Safety and "
            "Happiness. Prudence, indeed, will dictate that Governments long "
            "established should not be changed for light and transient causes; and "
            "accordingly all experience hath shewn, that mankind are more disposed "
            "to suffer, while evils are sufferable, than to right themselves by "
            "abolishing the forms to which they are accustomed. But when a long train "
            "of abuses and usurpations, pursuing invariably the same Object evinces "
            "a design to reduce them under absolute Despotism, it is their right, it "
            "is their duty, to throw off such Government, and to provide new Guards "
            "for their future security.--Such has been the patient sufferance of "
            "these Colonies; and such is now the necessity which constrains them to "
            "alter their former Systems of Government. The history of the present "
            "King of Great Britain is a history of repeated injuries and usurpations, "
            "all having in direct object the establishment of an absolute Tyranny "
            "over these States. To prove this, let Facts be submitted to a candid world."
        ),
    },
    {
        "id": 2,
        "section": "grievances",
        "text": (
            "He has refused his Assent to Laws, the most wholesome and necessary for "
            "the public good. "
            "He has forbidden his Governors to pass Laws of immediate and pressing "
            "importance, unless suspended in their operation till his Assent should "
            "be obtained; and when so suspended, he has utterly neglected to attend "
            "to them. "
            "He has refused to pass other Laws for the accommodation of large "
            "districts of people, unless those people would relinquish the right of "
            "Representation in the Legislature, a right inestimable to them and "
            "formidable to tyrants only. "
            "He has called together legislative bodies at places unusual, "
            "uncomfortable, and distant from the depository of their public Records, "
            "for the sole purpose of fatiguing them into compliance with his measures. "
            "He has dissolved Representative Houses repeatedly, for opposing with "
            "manly firmness his invasions on the rights of the people. "
            "He has refused for a long time, after such dissolutions, to cause others "
            "to be elected; whereby the Legislative powers, incapable of Annihilation, "
            "have returned to the People at large for their exercise; the State "
            "remaining in the mean time exposed to all the dangers of invasion from "
            "without, and convulsions within. "
            "He has endeavoured to prevent the population of these States; for that "
            "purpose obstructing the Laws for Naturalization of Foreigners; refusing "
            "to pass others to encourage their migrations hither, and raising the "
            "conditions of new Appropriations of Lands. "
            "He has obstructed the Administration of Justice, by refusing his Assent "
            "to Laws for establishing Judiciary powers. "
            "He has made Judges dependent on his Will alone, for the tenure of their "
            "offices, and the amount and payment of their salaries. "
            "He has erected a multitude of New Offices, and sent hither swarms of "
            "Officers to harrass our people, and eat out their substance. "
            "He has kept among us, in times of peace, Standing Armies without the "
            "Consent of our legislatures. "
            "He has affected to render the Military independent of and superior to "
            "the Civil power. "
            "He has combined with others to subject us to a jurisdiction foreign to "
            "our constitution, and unacknowledged by our laws; giving his Assent to "
            "their Acts of pretended Legislation: "
            "For Quartering large bodies of armed troops among us: "
            "For protecting them, by a mock Trial, from punishment for any Murders "
            "which they should commit on the Inhabitants of these States: "
            "For cutting off our Trade with all parts of the world: "
            "For imposing Taxes on us without our Consent: "
            "For depriving us in many cases, of the benefits of Trial by Jury: "
            "For transporting us beyond Seas to be tried for pretended offences: "
            "For abolishing the free System of English Laws in a neighbouring "
            "Province, establishing therein an Arbitrary government, and enlarging "
            "its Boundaries so as to render it at once an example and fit instrument "
            "for introducing the same absolute rule into these Colonies: "
            "For taking away our Charters, abolishing our most valuable Laws, and "
            "altering fundamentally the Forms of our Governments: "
            "For suspending our own Legislatures, and declaring themselves invested "
            "with power to legislate for us in all cases whatsoever. "
            "He has abdicated Government here, by declaring us out of his Protection "
            "and waging War against us. "
            "He has plundered our seas, ravaged our Coasts, burnt our towns, and "
            "destroyed the lives of our people. "
            "He is at this time transporting large Armies of foreign Mercenaries to "
            "compleat the works of death, desolation and tyranny, already begun with "
            "circumstances of Cruelty & perfidy scarcely paralleled in the most "
            "barbarous ages, and totally unworthy the Head of a civilized nation. "
            "He has constrained our fellow Citizens taken Captive on the high Seas "
            "to bear Arms against their Country, to become the executioners of their "
            "friends and Brethren, or to fall themselves by their Hands. "
            "He has excited domestic insurrections amongst us, and has endeavoured "
            "to bring on the inhabitants of our frontiers, the merciless Indian "
            "Savages, whose known rule of warfare, is an undistinguished destruction "
            "of all ages, sexes and conditions."
        ),
    },
    {
        "id": 3,
        "section": "appeal",
        "text": (
            "In every stage of these Oppressions We have Petitioned for Redress in "
            "the most humble terms: Our repeated Petitions have been answered only "
            "by repeated injury. A Prince, whose character is thus marked by every "
            "act which may define a Tyrant, is unfit to be the ruler of a free people. "
            "Nor have We been wanting in attentions to our Brittish brethren. We have "
            "warned them from time to time of attempts by their legislature to extend "
            "an unwarrantable jurisdiction over us. We have reminded them of the "
            "circumstances of our emigration and settlement here. We have appealed "
            "to their native justice and magnanimity, and we have conjured them by "
            "the ties of our common kindred to disavow these usurpations, which, "
            "would inevitably interrupt our connections and correspondence. They too "
            "have been deaf to the voice of justice and of consanguinity. We must, "
            "therefore, acquiesce in the necessity, which denounces our Separation, "
            "and hold them, as we hold the rest of mankind, Enemies in War, in Peace Friends."
        ),
    },
    {
        "id": 4,
        "section": "declaration",
        "text": (
            "We, therefore, the Representatives of the united States of America, in "
            "General Congress, Assembled, appealing to the Supreme Judge of the world "
            "for the rectitude of our intentions, do, in the Name, and by Authority "
            "of the good People of these Colonies, solemnly publish and declare, That "
            "these United Colonies are, and of Right ought to be Free and Independent "
            "States; that they are Absolved from all Allegiance to the British Crown, "
            "and that all political connection between them and the State of Great "
            "Britain, is and ought to be totally dissolved; and that as Free and "
            "Independent States, they have full Power to levy War, conclude Peace, "
            "contract Alliances, establish Commerce, and to do all other Acts and "
            "Things which Independent States may of right do. And for the support of "
            "this Declaration, with a firm reliance on the protection of divine "
            "Providence, we mutually pledge to each other our Lives, our Fortunes "
            "and our sacred Honor."
        ),
    },
]

FULL_TEXT: str = "\n\n".join(p["text"] for p in _PARAGRAPHS)  # type: ignore[index]

# ---------------------------------------------------------------------------
# 2. STOP WORDS
# ---------------------------------------------------------------------------

_STOP_WORDS: FrozenSet[str] = frozenset(
    """
    a an the and or but nor so yet for of in on at to by as is are was were
    be been being have has had do does did will would shall should may might
    must can could it its he his him she her they them their we us our who
    which that this these those what when where how all any both each few
    more most other some such no not only same than too very s t just from
    with among into upon out about up over after before between through
    """.split()
)

# ---------------------------------------------------------------------------
# 3. SENTIMENT LEXICON
# ---------------------------------------------------------------------------

_POSITIVE_WORDS: FrozenSet[str] = frozenset(
    """
    equal free liberty happiness justice peace right rights good wholesome
    sacred honor security safety prudence magnanimity rectitude divine
    independence consent protection support strength firm power reliance
    just inalienable unalienable worthy honest noble ordain establish
    """.split()
)

_NEGATIVE_WORDS: FrozenSet[str] = frozenset(
    """
    tyranny tyrant despotism oppression usurpation injury abuse murder
    plunder destruction desolation cruelty perfidy barbarous war invaded
    refused dissolved neglected harass dependent swarms taxed imprisoned
    destroyed devastated obstruct suspend abolish relinquish fatigue
    """.split()
)

# ---------------------------------------------------------------------------
# 4. DATA CLASSES
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """A single normalised word token with its source location."""

    word: str
    para_id: int
    position: int  # 0-based index within its paragraph


@dataclass
class NGram:
    """Represents an n-gram with its pointwise mutual information score."""

    tokens: Tuple[str, ...]
    raw_count: int
    pmi: float = 0.0

    def __str__(self) -> str:
        return " ".join(self.tokens)

    def __repr__(self) -> str:
        return f"NGram({self!s}, count={self.raw_count}, pmi={self.pmi:.4f})"


@dataclass
class SentimentResult:
    """Polarity and subjectivity scores for a piece of text."""

    positive_count: int
    negative_count: int
    total_tokens: int
    polarity: float          # [-1, +1]
    subjectivity: float      # [0, 1]

    def __str__(self) -> str:
        label = "POSITIVE" if self.polarity > 0.05 else ("NEGATIVE" if self.polarity < -0.05 else "NEUTRAL")
        return (
            f"Sentiment({label}, polarity={self.polarity:+.3f}, "
            f"subjectivity={self.subjectivity:.3f})"
        )


@dataclass
class GraphNode:
    """Node in the ConceptGraph."""

    word: str
    neighbours: Dict[str, float] = field(default_factory=dict)  # word -> weight

    def degree(self) -> float:
        return sum(self.neighbours.values())


# ---------------------------------------------------------------------------
# 5. TOKENIZER
# ---------------------------------------------------------------------------


class TextTokenizer:
    """
    Multi-stage tokenizer:
      1. Split into sentences (heuristic regex)
      2. Tokenize each sentence into words
      3. Normalize: lower-case, strip punctuation
      4. Optionally remove stop words
    """

    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+")
    _WORD_SPLIT = re.compile(r"[a-zA-Z']+")

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords

    # ------------------------------------------------------------------
    def sentences(self, text: str) -> List[str]:
        return [s.strip() for s in self._SENTENCE_SPLIT.split(text) if s.strip()]

    # ------------------------------------------------------------------
    def words(self, text: str) -> List[str]:
        raw = self._WORD_SPLIT.findall(text.lower())
        raw = [w.strip("'") for w in raw]
        if self.remove_stopwords:
            raw = [w for w in raw if w not in _STOP_WORDS and len(w) > 1]
        return raw

    # ------------------------------------------------------------------
    def tokenize_paragraphs(
        self, paragraphs: List[Dict[str, object]]
    ) -> List[List[Token]]:
        result: List[List[Token]] = []
        for para in paragraphs:
            pid = int(para["id"])  # type: ignore[arg-type]
            words = self.words(str(para["text"]))
            result.append(
                [Token(word=w, para_id=pid, position=i) for i, w in enumerate(words)]
            )
        return result

    # ------------------------------------------------------------------
    def flat_tokens(
        self, paragraphs: List[Dict[str, object]]
    ) -> List[Token]:
        return list(itertools.chain.from_iterable(self.tokenize_paragraphs(paragraphs)))


# ---------------------------------------------------------------------------
# 6. FREQUENCY ANALYZER (TF-IDF)
# ---------------------------------------------------------------------------


class FrequencyAnalyzer:
    """
    Computes term frequency (TF), document frequency (DF), and TF-IDF
    scores across a collection of documents (paragraphs).
    """

    def __init__(self, tokenizer: TextTokenizer):
        self._tok = tokenizer
        self._para_tokens: List[List[str]] = []
        self._all_words: List[str] = []
        self._global_freq: collections.Counter = collections.Counter()
        self._doc_freq: collections.Counter = collections.Counter()
        self._tf: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}
        self._tfidf: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def fit(self, paragraphs: List[Dict[str, object]]) -> "FrequencyAnalyzer":
        n_docs = len(paragraphs)
        self._para_tokens = [self._tok.words(str(p["text"])) for p in paragraphs]
        self._all_words = list(itertools.chain.from_iterable(self._para_tokens))
        self._global_freq = collections.Counter(self._all_words)

        # TF per document
        self._tf = []
        for words in self._para_tokens:
            freq: collections.Counter = collections.Counter(words)
            total = len(words) or 1
            self._tf.append({w: c / total for w, c in freq.items()})

        # Document frequency
        self._doc_freq = collections.Counter()
        for words in self._para_tokens:
            for w in set(words):
                self._doc_freq[w] += 1

        # IDF (smoothed)
        self._idf = {
            w: math.log((1 + n_docs) / (1 + df)) + 1
            for w, df in self._doc_freq.items()
        }

        # TF-IDF
        self._tfidf = [
            {w: tf_val * self._idf.get(w, 1.0) for w, tf_val in doc_tf.items()}
            for doc_tf in self._tf
        ]
        return self

    # ------------------------------------------------------------------
    def top_terms(self, n: int = 20) -> List[Tuple[str, float]]:
        """Global TF-IDF: average score across all paragraphs."""
        aggregate: Dict[str, float] = collections.defaultdict(float)
        for doc in self._tfidf:
            for w, score in doc.items():
                aggregate[w] += score
        return sorted(aggregate.items(), key=lambda x: x[1], reverse=True)[:n]

    # ------------------------------------------------------------------
    def top_terms_for_section(self, section_id: int, n: int = 10) -> List[Tuple[str, float]]:
        if section_id >= len(self._tfidf):
            raise IndexError(f"section_id {section_id} out of range")
        return sorted(self._tfidf[section_id].items(), key=lambda x: x[1], reverse=True)[:n]

    # ------------------------------------------------------------------
    def global_frequency(self, word: str) -> int:
        return self._global_freq.get(word.lower(), 0)

    # ------------------------------------------------------------------
    def vocabulary_size(self) -> int:
        return len(self._global_freq)


# ---------------------------------------------------------------------------
# 7. N-GRAM ANALYZER
# ---------------------------------------------------------------------------


class NGramAnalyzer:
    """
    Extracts bigrams and trigrams, computes pointwise mutual information (PMI).

    PMI(x, y) = log2( P(x,y) / (P(x) * P(y)) )
    """

    def __init__(self, tokenizer: TextTokenizer):
        self._tok = tokenizer
        self._unigram_counts: collections.Counter = collections.Counter()
        self._bigram_counts: collections.Counter = collections.Counter()
        self._trigram_counts: collections.Counter = collections.Counter()
        self._total_unigrams: int = 0

    # ------------------------------------------------------------------
    def fit(self, paragraphs: List[Dict[str, object]]) -> "NGramAnalyzer":
        all_words: List[str] = []
        for p in paragraphs:
            all_words.extend(self._tok.words(str(p["text"])))

        self._total_unigrams = len(all_words)
        self._unigram_counts = collections.Counter(all_words)

        self._bigram_counts = collections.Counter(
            zip(all_words, all_words[1:])
        )
        self._trigram_counts = collections.Counter(
            zip(all_words, all_words[1:], all_words[2:])
        )
        return self

    # ------------------------------------------------------------------
    def _pmi(self, *words: str, joint_count: int) -> float:
        n = self._total_unigrams
        if n == 0 or joint_count == 0:
            return float("-inf")
        p_joint = joint_count / n
        p_product = math.prod(self._unigram_counts[w] / n for w in words)
        if p_product == 0:
            return float("-inf")
        return math.log2(p_joint / p_product)

    # ------------------------------------------------------------------
    def top_bigrams(self, n: int = 15, min_count: int = 2) -> List[NGram]:
        results: List[NGram] = []
        for (w1, w2), cnt in self._bigram_counts.items():
            if cnt < min_count:
                continue
            pmi = self._pmi(w1, w2, joint_count=cnt)
            results.append(NGram(tokens=(w1, w2), raw_count=cnt, pmi=pmi))
        results.sort(key=lambda g: g.pmi, reverse=True)
        return results[:n]

    # ------------------------------------------------------------------
    def top_trigrams(self, n: int = 10, min_count: int = 2) -> List[NGram]:
        results: List[NGram] = []
        for (w1, w2, w3), cnt in self._trigram_counts.items():
            if cnt < min_count:
                continue
            pmi = self._pmi(w1, w2, w3, joint_count=cnt)
            results.append(NGram(tokens=(w1, w2, w3), raw_count=cnt, pmi=pmi))
        results.sort(key=lambda g: g.pmi, reverse=True)
        return results[:n]

    # ------------------------------------------------------------------
    def continuations(self, prefix: Tuple[str, ...]) -> List[Tuple[str, int]]:
        """All words that follow a given prefix bigram, sorted by frequency."""
        if len(prefix) == 1:
            w = prefix[0]
            results = [
                (w2, cnt)
                for (w1, w2), cnt in self._bigram_counts.items()
                if w1 == w
            ]
        elif len(prefix) == 2:
            w1, w2 = prefix
            results = [
                (w3, cnt)
                for (a, b, w3), cnt in self._trigram_counts.items()
                if a == w1 and b == w2
            ]
        else:
            raise ValueError("prefix must be length 1 or 2")
        return sorted(results, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# 8. CONCEPT GRAPH
# ---------------------------------------------------------------------------


class ConceptGraph:
    """
    Builds a weighted undirected co-occurrence graph.
    Words within a sliding window of size `window` are considered co-occurring.
    Edge weight = number of co-occurrences.

    Supports:
    * BFS / DFS traversal
    * Dijkstra shortest path (inverse weight = distance)
    * PageRank-style centrality
    * Community detection (greedy modularity approximation)
    """

    def __init__(self, window: int = 5):
        self.window = window
        self._nodes: Dict[str, GraphNode] = {}

    # ------------------------------------------------------------------
    def fit(self, tokens: List[str]) -> "ConceptGraph":
        self._nodes = {}
        for i, word in enumerate(tokens):
            if word not in self._nodes:
                self._nodes[word] = GraphNode(word=word)
            context = tokens[i + 1: i + 1 + self.window]
            for other in context:
                if other == word:
                    continue
                if other not in self._nodes:
                    self._nodes[other] = GraphNode(word=other)
                self._nodes[word].neighbours[other] = (
                    self._nodes[word].neighbours.get(other, 0) + 1
                )
                self._nodes[other].neighbours[word] = (
                    self._nodes[other].neighbours.get(word, 0) + 1
                )
        return self

    # ------------------------------------------------------------------
    def bfs(self, start: str) -> List[str]:
        if start not in self._nodes:
            return []
        visited: Set[str] = set()
        queue: collections.deque = collections.deque([start])
        order: List[str] = []
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            neighbours = sorted(
                self._nodes[node].neighbours.keys(),
                key=lambda w: -self._nodes[node].neighbours[w],
            )
            queue.extend(n for n in neighbours if n not in visited)
        return order

    # ------------------------------------------------------------------
    def dfs(self, start: str) -> List[str]:
        if start not in self._nodes:
            return []
        visited: Set[str] = set()
        order: List[str] = []

        def _visit(w: str) -> None:
            if w in visited:
                return
            visited.add(w)
            order.append(w)
            for nb in sorted(
                self._nodes[w].neighbours.keys(),
                key=lambda x: -self._nodes[w].neighbours[x],
            ):
                _visit(nb)

        _visit(start)
        return order

    # ------------------------------------------------------------------
    def shortest_path(self, source: str, target: str) -> Tuple[List[str], float]:
        """Dijkstra with distance = 1 / co-occurrence weight."""
        if source not in self._nodes or target not in self._nodes:
            return [], float("inf")

        dist: Dict[str, float] = {source: 0.0}
        prev: Dict[str, Optional[str]] = {source: None}
        heap: List[Tuple[float, str]] = [(0.0, source)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            if u == target:
                break
            for v, weight in self._nodes[u].neighbours.items():
                edge_cost = 1.0 / (weight + 1e-9)
                new_dist = dist[u] + edge_cost
                if new_dist < dist.get(v, float("inf")):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))

        # Reconstruct
        if target not in dist:
            return [], float("inf")
        path: List[str] = []
        cur: Optional[str] = target
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path, dist[target]

    # ------------------------------------------------------------------
    def pagerank(self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
        """Compute PageRank scores."""
        n = len(self._nodes)
        if n == 0:
            return {}
        rank: Dict[str, float] = {w: 1.0 / n for w in self._nodes}

        for _ in range(max_iter):
            new_rank: Dict[str, float] = {}
            for word, node in self._nodes.items():
                incoming = 0.0
                for other_word, other_node in self._nodes.items():
                    if word in other_node.neighbours:
                        w = other_node.neighbours[word]
                        out_total = other_node.degree() or 1
                        incoming += rank[other_word] * (w / out_total)
                new_rank[word] = (1 - damping) / n + damping * incoming

            # Check convergence
            delta = sum(abs(new_rank[w] - rank[w]) for w in rank)
            rank = new_rank
            if delta < tol:
                break

        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    def top_nodes_by_degree(self, n: int = 10) -> List[Tuple[str, float]]:
        return sorted(
            [(w, node.degree()) for w, node in self._nodes.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:n]

    # ------------------------------------------------------------------
    def node_count(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    def edge_count(self) -> int:
        return sum(len(node.neighbours) for node in self._nodes.values()) // 2


# ---------------------------------------------------------------------------
# 9. MARKOV CHAIN TEXT GENERATOR
# ---------------------------------------------------------------------------


class MarkovChain:
    """
    Variable-order Markov chain text generator.
    Trained on the Declaration of Independence.
    """

    def __init__(self, order: int = 2, seed: Optional[int] = None):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self._rng = random.Random(seed)
        self._chain: Dict[Tuple[str, ...], List[str]] = collections.defaultdict(list)
        self._starts: List[Tuple[str, ...]] = []

    # ------------------------------------------------------------------
    def fit(self, paragraphs: List[Dict[str, object]]) -> "MarkovChain":
        tokenizer = TextTokenizer(remove_stopwords=False)
        for p in paragraphs:
            words = tokenizer.words(str(p["text"]))
            if len(words) <= self.order:
                continue
            self._starts.append(tuple(words[: self.order]))
            for i in range(len(words) - self.order):
                key = tuple(words[i: i + self.order])
                self._chain[key].append(words[i + self.order])
        return self

    # ------------------------------------------------------------------
    def generate(self, max_words: int = 80) -> str:
        if not self._starts:
            return ""
        state = self._rng.choice(self._starts)
        output: List[str] = list(state)

        for _ in range(max_words - self.order):
            nexts = self._chain.get(state)
            if not nexts:
                break
            next_word = self._rng.choice(nexts)
            output.append(next_word)
            state = tuple(output[-self.order:])

        return " ".join(output)

    # ------------------------------------------------------------------
    def perplexity(self, text: str) -> float:
        """
        Approximate perplexity of a given text under this Markov model.
        Lower is better (model is more confident about this text).
        """
        tokenizer = TextTokenizer(remove_stopwords=False)
        words = tokenizer.words(text)
        if len(words) <= self.order:
            return float("inf")

        log_prob = 0.0
        count = 0
        for i in range(len(words) - self.order):
            key = tuple(words[i: i + self.order])
            next_word = words[i + self.order]
            continuations = self._chain.get(key, [])
            total = len(continuations)
            freq = continuations.count(next_word)
            # Laplace smoothing
            prob = (freq + 1) / (total + len(self._chain) + 1)
            log_prob += math.log(prob)
            count += 1

        if count == 0:
            return float("inf")
        return math.exp(-log_prob / count)


# ---------------------------------------------------------------------------
# 10. SENTIMENT ANALYZER
# ---------------------------------------------------------------------------


class SentimentAnalyzer:
    """
    Lexicon-based sentiment analysis.
    Polarity  = (pos - neg) / (pos + neg + 1e-9)  ∈ [-1, +1]
    Subjectivity = (pos + neg) / total_tokens      ∈ [0, 1]
    """

    def __init__(self):
        self._tokenizer = TextTokenizer(remove_stopwords=False)

    # ------------------------------------------------------------------
    def analyze(self, text: str) -> SentimentResult:
        words = self._tokenizer.words(text)
        pos = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
        total = len(words) or 1
        polarity = (pos - neg) / (pos + neg + 1e-9)
        subjectivity = (pos + neg) / total
        return SentimentResult(
            positive_count=pos,
            negative_count=neg,
            total_tokens=total,
            polarity=polarity,
            subjectivity=subjectivity,
        )

    # ------------------------------------------------------------------
    def analyze_paragraphs(
        self, paragraphs: List[Dict[str, object]]
    ) -> List[Tuple[str, SentimentResult]]:
        return [
            (str(p["section"]), self.analyze(str(p["text"])))
            for p in paragraphs
        ]


# ---------------------------------------------------------------------------
# 11. REPORT GENERATOR
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Produces a formatted textual analysis report."""

    _WIDTH = 72

    def __init__(self, paragraphs: List[Dict[str, object]]):
        self._paragraphs = paragraphs
        self._tokenizer = TextTokenizer()
        self._freq = FrequencyAnalyzer(self._tokenizer).fit(paragraphs)
        self._ngram = NGramAnalyzer(self._tokenizer).fit(paragraphs)
        self._sentiment = SentimentAnalyzer()
        all_words = [t.word for t in self._tokenizer.flat_tokens(paragraphs)]
        self._graph = ConceptGraph(window=5).fit(all_words)
        self._markov = MarkovChain(order=2, seed=42).fit(paragraphs)

    # ------------------------------------------------------------------
    def _hr(self, char: str = "=") -> str:
        return char * self._WIDTH

    # ------------------------------------------------------------------
    def _header(self, title: str) -> str:
        pad = (self._WIDTH - len(title) - 2) // 2
        return f"\n{self._hr()}\n{' ' * pad} {title} {' ' * pad}\n{self._hr()}"

    # ------------------------------------------------------------------
    def generate(self) -> str:
        lines: List[str] = []

        lines.append(self._header("DECLARATION OF INDEPENDENCE — ANALYSIS REPORT"))

        # ── Overview ──────────────────────────────────────────────────
        lines.append(self._header("OVERVIEW"))
        total_words_raw = len(TextTokenizer(remove_stopwords=False).flat_tokens(self._paragraphs))
        lines.append(f"  Paragraphs    : {len(self._paragraphs)}")
        lines.append(f"  Total words   : {total_words_raw}")
        lines.append(f"  Vocabulary    : {self._freq.vocabulary_size()} unique terms")
        lines.append(f"  Graph nodes   : {self._graph.node_count()}")
        lines.append(f"  Graph edges   : {self._graph.edge_count()}")

        # ── Top TF-IDF Terms ──────────────────────────────────────────
        lines.append(self._header("TOP 20 TERMS BY TF-IDF"))
        for rank, (word, score) in enumerate(self._freq.top_terms(20), 1):
            lines.append(f"  {rank:2d}. {word:<22s} {score:.4f}")

        # ── Top Bigrams ───────────────────────────────────────────────
        lines.append(self._header("TOP BIGRAMS BY PMI"))
        for bg in self._ngram.top_bigrams(10, min_count=2):
            lines.append(f"  {str(bg):<35s}  count={bg.raw_count}  PMI={bg.pmi:.3f}")

        # ── Sentiment ─────────────────────────────────────────────────
        lines.append(self._header("SENTIMENT BY SECTION"))
        for section, result in self._sentiment.analyze_paragraphs(self._paragraphs):
            lines.append(f"  {section:<14s}  {result}")

        # ── PageRank Centrality ───────────────────────────────────────
        lines.append(self._header("TOP 10 CONCEPT NODES (PageRank)"))
        pr = self._graph.pagerank()
        for i, (word, score) in enumerate(list(pr.items())[:10], 1):
            lines.append(f"  {i:2d}. {word:<22s} {score:.6f}")

        # ── Shortest Path ──────────────────────────────────────────────
        lines.append(self._header("SHORTEST PATH: 'liberty' → 'tyranny'"))
        path, cost = self._graph.shortest_path("liberty", "tyranny")
        if path:
            lines.append("  " + " → ".join(path))
            lines.append(f"  (total inverse-weight cost: {cost:.4f})")
        else:
            lines.append("  No path found between these concepts.")

        # ── Markov Generation ──────────────────────────────────────────
        lines.append(self._header("GENERATED TEXT (Markov Chain, order=2)"))
        generated = self._markov.generate(max_words=60)
        lines.append(textwrap.fill("  " + generated, width=self._WIDTH))

        lines.append(f"\n{self._hr()}\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 12. CLI
# ---------------------------------------------------------------------------


class CLIParser:
    """
    Dispatch-table based CLI.
    Usage: python declaration.py <command> [args...]

    Commands:
      report              – full analysis report
      freq  [n]           – top n TF-IDF terms (default 20)
      sentiment           – sentiment per section
      path  <w1> <w2>     – shortest concept-graph path between two words
      generate [words]    – generate text with Markov chain
      search  <term>      – search occurrences of a term
      bigrams [n]         – top n bigrams
      pagerank [n]        – top n PageRank nodes
    """

    def __init__(self, paragraphs: List[Dict[str, object]]):
        self._paragraphs = paragraphs
        self._tokenizer = TextTokenizer()
        self._freq = FrequencyAnalyzer(self._tokenizer).fit(paragraphs)
        self._ngram = NGramAnalyzer(self._tokenizer).fit(paragraphs)
        self._sentiment = SentimentAnalyzer()
        all_words = [t.word for t in self._tokenizer.flat_tokens(paragraphs)]
        self._graph = ConceptGraph(window=5).fit(all_words)
        self._markov = MarkovChain(order=2, seed=0).fit(paragraphs)
        self._report_gen = ReportGenerator(paragraphs)

        self._commands: Dict[str, Callable[[List[str]], None]] = {
            "report":    self._cmd_report,
            "freq":      self._cmd_freq,
            "sentiment": self._cmd_sentiment,
            "path":      self._cmd_path,
            "generate":  self._cmd_generate,
            "search":    self._cmd_search,
            "bigrams":   self._cmd_bigrams,
            "pagerank":  self._cmd_pagerank,
            "help":      self._cmd_help,
        }

    # ------------------------------------------------------------------
    def run(self, argv: List[str]) -> None:
        if not argv:
            self._cmd_help([])
            return
        cmd, *args = argv
        handler = self._commands.get(cmd)
        if handler is None:
            print(f"Unknown command: {cmd!r}. Try 'help'.", file=sys.stderr)
            sys.exit(1)
        handler(args)

    # ------------------------------------------------------------------
    def _cmd_report(self, args: List[str]) -> None:
        print(self._report_gen.generate())

    def _cmd_freq(self, args: List[str]) -> None:
        n = int(args[0]) if args else 20
        for word, score in self._freq.top_terms(n):
            print(f"  {word:<25s} {score:.4f}")

    def _cmd_sentiment(self, args: List[str]) -> None:
        for section, result in self._sentiment.analyze_paragraphs(self._paragraphs):
            print(f"  {section:<14s}  {result}")

    def _cmd_path(self, args: List[str]) -> None:
        if len(args) < 2:
            print("Usage: path <word1> <word2>", file=sys.stderr)
            return
        path, cost = self._graph.shortest_path(args[0].lower(), args[1].lower())
        if path:
            print(" → ".join(path))
            print(f"Cost: {cost:.4f}")
        else:
            print("No path found.")

    def _cmd_generate(self, args: List[str]) -> None:
        n = int(args[0]) if args else 60
        print(textwrap.fill(self._markov.generate(max_words=n), width=72))

    def _cmd_search(self, args: List[str]) -> None:
        if not args:
            print("Usage: search <term>", file=sys.stderr)
            return
        term = args[0].lower()
        total = self._freq.global_frequency(term)
        print(f"'{term}' appears {total} time(s) across all paragraphs.")
        for p in self._paragraphs:
            text = str(p["text"])
            occurrences = len(re.findall(re.escape(term), text, re.IGNORECASE))
            if occurrences:
                print(f"  Section [{p['section']}]: {occurrences} occurrence(s)")

    def _cmd_bigrams(self, args: List[str]) -> None:
        n = int(args[0]) if args else 15
        for bg in self._ngram.top_bigrams(n, min_count=2):
            print(f"  {str(bg):<30s}  count={bg.raw_count}  PMI={bg.pmi:.3f}")

    def _cmd_pagerank(self, args: List[str]) -> None:
        n = int(args[0]) if args else 10
        pr = self._graph.pagerank()
        for i, (word, score) in enumerate(list(pr.items())[:n], 1):
            print(f"  {i:2d}. {word:<22s} {score:.6f}")

    def _cmd_help(self, args: List[str]) -> None:
        print(textwrap.dedent(CLIParser.__doc__ or ""))


# ---------------------------------------------------------------------------
# 13. ENTRY POINT
# ---------------------------------------------------------------------------


def main() -> None:
    cli = CLIParser(_PARAGRAPHS)
    cli.run(sys.argv[1:])


if __name__ == "__main__":
    main()
