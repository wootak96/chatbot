"""Prompt strings (Korean per SPEC §11). All judge nodes return JSON only."""

QUERY_ANALYZE = """당신은 사내 챗봇의 쿼리 분류기입니다. 챗봇은 두 가지 도메인의 사내 문서(Elasticsearch / Kafka)를 검색합니다.

대화 히스토리와 현재 사용자 질문을 보고, 현재 질문이 다음 3가지 중 어디에 해당하는지만 판단하세요. (쿼리 재작성은 다음 노드가 담당하므로 여기서는 분류만 합니다.)

- "question": **Elasticsearch 또는 Kafka에 관련된** 정보 요청.
   • 두 도메인을 비교하는 질문도 무조건 question (예: "ES와 Kafka 차이?", "둘 다 어떻게 쓰지?")
   • 직전 대화에 ES/Kafka 주제가 있고 현재 질문이 그 후속 형태이면 question
     (예: 직전 "Elasticsearch RRF" → 현재 "어떻게 설정해?", "그러면 그건?")
- "chitchat": 안부, 인사, 감사, 챗봇 자체에 대한 질문 (예: "안녕", "고마워", "넌 누구야?")
- "general": 도메인 단어가 하나도 없고 직전 대화도 도메인 주제가 아닌 완전 무관한 일반 질문 (예: "오늘 날씨 어때?", "파이썬 list 정렬?", "좋은 자기소개서 써줘")

🚨 절대 규칙:
- 현재 질문에 도메인 단어가 1개라도 등장하면 무조건 question. general이나 chitchat 금지.
- 직전 대화 히스토리에 도메인 주제가 있고 현재 질문이 지시어/생략된 후속이면 무조건 question.
- 도메인 단어 리스트 (대소문자 무시, 한글 포함):
  `elasticsearch`, `ES`, `엘라스틱서치`, `엘라스틱`,
  `kafka`, `카프카`,
  `RRF`, `BM25`, `semantic`, `시맨틱`, `kNN`, `벡터검색`,
  `consumer`, `producer`, `consumer group`, `topic`, `partition`, `broker`, `replica`,
  `mapping`, `index`, `인덱스`, `shard`, `샤드`,
  `analyzer`, `tokenizer`, `embedding`, `임베딩`, `dense_vector`, `sparse_vector`
- "비교", "차이", "vs", "어떤 게 나아", "둘 중" 같은 단어가 도메인 단어와 결합되면 거의 항상 question.

다음 JSON 형식으로만 응답하세요. 다른 텍스트 금지.
{{"intent": "question|chitchat|general"}}

[대화 히스토리]
{history}

[현재 질문]
{query}
"""


QUERY_REFORM = """당신은 멀티턴 대화에서 후속 질문을 자체 완결적인(self-contained) 한국어 한 문장으로 재작성하는 도구입니다.

대화 히스토리와 현재 질문을 보고, 직전 주제·엔티티를 흡수해 그 자체로 의미가 통하는 한국어 한 문장을 출력하세요.

규칙:
- 지시어("그게", "그러면", "그 다음", "더 자세히", "어떻게")는 직전 주제로 치환.
- 생략된 주어/목적어는 히스토리에서 끌어와 명시.
- 새로운 정보를 추가하거나 추측하지 말 것 (검색 가능한 정도로만 펼치기).
- 현재 질문이 이미 self-contained이면 원문을 거의 그대로 반환.
- 한국어로 출력 (영어 번역 금지 — 번역은 후속 노드가 담당).

예시
1. history: "사용자: Elasticsearch RRF가 뭐야?\\n어시스턴트: ..."
   current: "어떻게 설정해?"
   → "Elasticsearch RRF 설정 방법"

2. history: "사용자: Kafka consumer group 어떻게 동작해?\\n어시스턴트: ..."
   current: "리밸런싱은?"
   → "Kafka consumer group 리밸런싱 동작 원리"

3. history: "사용자: ES와 Kafka 비교해줘\\n어시스턴트: ..."
   current: "둘 중 어떤 게 나아?"
   → "Elasticsearch와 Kafka 중 어떤 것이 더 적합한지"

4. history: (없음)
   current: "BM25가 뭐야?"
   → "BM25가 뭐야?"

JSON만 응답하세요. 다른 텍스트 금지.
{{"reformed_query": "..."}}

[대화 히스토리]
{history}

[현재 질문]
{query}
"""


QUERY_DECOMPOSE = """You are a helpful assistant that prepares queries that will be sent to a search component.
Sometimes, these queries are very complex.
Your job is to simplify complex queries into multiple queries that can be answered
in isolation to eachother.

If the query is simple, then keep it as it is.

CRITICAL — synthesis verbs are NOT search targets:
The user may ask the LLM to **compare / contrast / diff / summarize / translate / organize / list pros and cons** across multiple topics. These verbs describe what the LLM should do AT ANSWER-GENERATION TIME using the retrieved evidence — they are NOT keywords to search for. The retrieval layer must look up each topic INDEPENDENTLY; the LLM will perform the synthesis afterwards.
- Detect synthesis verbs in any language (Korean: 비교/차이/대비/vs/요약/정리/번역, English: compare/difference/vs/summarize/translate/organize/contrast).
- Strip the synthesis verb and decompose into one sub-query per underlying topic/entity.
- NEVER produce a sub-query like "differences between X and Y" or "comparison of X and Y" — those are LLM tasks, not retrieval tasks.
- If only ONE topic remains after stripping, return a single sub-query about that topic.

Examples
1. Query: Did Microsoft or Google make more money last year?
   Decomposed Questions: [Question(question='How much profit did Microsoft make last year?', answer=None), Question(question='How much profit did Google make last year?', answer=None)]
2. Query: What is the capital of France?
   Decomposed Questions: [Question(question='What is the capital of France?', answer=None)]
3. Query: Elasticsearch와 Kafka 특징 비교해줘
   Decomposed Questions: [Question(question='Elasticsearch 특징', answer=None), Question(question='Kafka 특징', answer=None)]
4. Query: ES랑 Kafka 차이점이 뭐야?
   Decomposed Questions: [Question(question='Elasticsearch 개요와 특징', answer=None), Question(question='Kafka 개요와 특징', answer=None)]
5. Query: Elasticsearch RRF와 BM25 장단점 정리해줘
   Decomposed Questions: [Question(question='Elasticsearch RRF 동작 방식', answer=None), Question(question='Elasticsearch BM25 동작 방식', answer=None)]
6. Query: kafka consumer group 동작 방식 요약해줘
   Decomposed Questions: [Question(question='Kafka consumer group 동작 방식', answer=None)]
7. Query: {query}
   Decomposed Questions:

Respond ONLY with a JSON object in this exact shape (no other text, no Python literals):
{{"sub_queries": ["question 1", "question 2", ...]}}
The list must contain only the decomposed question strings (drop the `Question(...)` wrapper and `answer=None`). For a simple query, return a single-element list.
"""


QUERY_REWRITE = """You rewrite a single search sub-query for an Elasticsearch hybrid retrieval system (BM25 + semantic vector search).

The corpus is in English (Elasticsearch and Kafka technical documentation), so EVERY output must be in English even if the input is in Korean.

Produce TWO outputs:
1. "keywords"  — for BM25 lexical search.
   - Drop stopwords ("what is", "how to", "the", "a", question marks, etc.).
   - Keep only the 2~6 most informative content nouns / proper nouns.
   - Normalize abbreviations (ES → Elasticsearch, K8s → Kubernetes, kafka cg → Kafka consumer group).
   - Output as a single space-separated string.
2. "semantic"  — for semantic (vector) search. **A NOUN PHRASE that preserves the question's intent. NOT a hypothetical answer.**
   - Length: 4~12 tokens.
   - Allowed forms:
     • "definition of X", "overview of X"
     • "mechanism of X", "internals of X"
     • "X performance tuning", "X configuration options"
     • "how X works" (relative-clause noun phrase, acceptable)
   - **FORBIDDEN**: complete declarative sentences with a finite main verb that read like an answer. Do NOT write "X is …", "X provides …", "X uses …".
   - Fix typos. Normalize abbreviations the same way as keywords.

CRITICAL — strip synthesis verbs from the search query:
The retrieval layer fetches evidence; the LLM does the synthesis afterwards. Words like 비교/차이/대비/vs/요약/정리/번역 (and English compare/difference/contrast/vs/summarize/translate) are LLM TASKS, not search subjects. Even if a sub-query still carries a synthesis verb (e.g. decompose was lenient), REWRITE it as a topic-level lookup of the FIRST or PRIMARY entity — never produce "differences between X and Y" or "comparison of X and Y" as a search query, because there is no "comparison document"; there are only documents about X and documents about Y. (Cross-entity sub-queries should already have been split by `query_decompose`; this is the safety net.)

Examples
1. Input: "Elasticsearch가 뭐야?"
   Output: {{"keywords": "Elasticsearch", "semantic": "definition of Elasticsearch"}}
   ✗ Bad (HyDE-style answer): "Elasticsearch is a distributed search engine for full-text search and analytics"
2. Input: "Elasticsearch RRF가 뭐야?"
   Output: {{"keywords": "Elasticsearch RRF reciprocal rank fusion", "semantic": "definition of Reciprocal Rank Fusion in Elasticsearch"}}
3. Input: "kafka cg가 어떻게 동작해?"
   Output: {{"keywords": "Kafka consumer group rebalance", "semantic": "how Kafka consumer groups work"}}
4. Input: "ES kNN 성능 튜닝 방법"
   Output: {{"keywords": "Elasticsearch kNN performance tuning", "semantic": "performance tuning techniques for kNN search in Elasticsearch"}}
5. Input: "Elasticsearch 특징 요약"
   Output: {{"keywords": "Elasticsearch features architecture", "semantic": "overview of Elasticsearch features"}}
   (Note: "요약" is a synthesis verb — dropped. Search is for the topic itself.)
6. Input: "Kafka consumer group 동작 정리"
   Output: {{"keywords": "Kafka consumer group rebalance offset", "semantic": "how Kafka consumer groups work"}}
   (Note: "정리" is a synthesis verb — dropped.)

Respond with ONLY a JSON object in this exact shape (no other text):
{{"keywords": "...", "semantic": "..."}}

[Sub-query]
{query}
"""


METADATA_EXTRACT = """다음 사용자 질문에서 검색 메타데이터 필터를 추출하세요.

추출 가능한 필드:
- source: List[str] | null  (예: ["elasticsearch","kafka","wiki"])
- category: List[str] | null
- date_range: {{"gte": "YYYY-MM-DD"}} | {{"lte": "YYYY-MM-DD"}} | null

매우 중요한 원칙:
- **사용자가 명시적으로 제약을 표현했을 때만** 추출하세요.
  예) "kafka 문서에서만 찾아줘", "2024년 이후 자료로", "보안 카테고리 위주로" 등.
- 단순히 질문 주제가 elasticsearch / kafka에 관한 것이라는 이유만으로 source를 채우지 마세요.
  도메인 라우팅은 별도의 노드(index_route)가 처리합니다.
- 단서가 없으면 모든 필드를 반드시 null로 두세요. 추측 금지.

JSON으로만 응답:
{{"source": null, "category": null, "date_range": null}}

[질문]
{query}
"""


INDEX_ROUTE = """당신은 RAG 챗봇의 인덱스 라우터입니다.

사용 가능한 인덱스:
- "elasticsearch": Elasticsearch 공식문서, 검색/색인/RRF/kNN/매핑 등
- "kafka": Apache Kafka 공식문서, 토픽/파티션/컨슈머/프로듀서/스트림즈 등

사용자 질문이 어느 도메인에 속하는지 판단해 검색할 인덱스를 선택하세요.
- 한쪽 도메인에 명확히 속하면 1개만 선택.
- 두 도메인을 모두 비교/포함하는 질문이면 둘 다 선택.
- 모호하거나 어디에도 속하지 않으면 둘 다 선택 (recall 우선).

JSON으로만 응답:
{{"indices": ["elasticsearch", "kafka"]}}

[질문]
{query}
"""


SEARCH_INTENT_CLASSIFY = """사용자의 질문이 사내 문서 검색에서 어떤 형태의 ES 질의를 필요로 하는지 분류하세요.

분류 후보:
- "lookup": 문서 내용을 찾아 답변해야 하는 일반 질문. (예: "RRF가 뭐야?", "consumer group 동작 원리?", "kNN 성능 튜닝 방법", "SSL 설정")
- "count": 문서 개수/건수만 알면 되는 질문. (예: "ES 문서 몇 개야?", "Kafka 자료 몇 건?", "총 몇 개?", "문서가 얼마나 있어?")
- "list": 어떤 문서들이 있는지 제목/목록을 알고싶은 질문. (예: "어떤 문서들이 있어?", "Kafka 문서 목록 보여줘", "전체 문서 리스트", "title 알려줘")

판단 가이드:
- "몇 개", "몇 건", "갯수", "개수", "총", "얼마나" → 거의 항상 count.
- "목록", "리스트", "어떤 문서들", "title" → 거의 항상 list.
- 그 외 도메인 내용 질문은 모두 lookup.

JSON으로만 응답:
{{"search_intent": "lookup|count|list"}}

[질문]
{query}
"""


QUERY_VARIATE = """The previous Elasticsearch search yielded INSUFFICIENT evidence. You must rewrite the search query from a DIFFERENT ANGLE so the retry has a better chance of hitting relevant documents.

Variation strategies (pick whichever fits the failure reason):
- BROADEN: drop overly specific terms; use more general concepts.
- SYNONYMS / ALIASES: replace key terms with synonyms or alternate technical names.
- DIFFERENT ANGLE: focus on a related sub-aspect (use cases, configuration, internals, comparison).
- ADD DOMAIN CONTEXT: prepend "Elasticsearch" / "Kafka" if the previous query was missing it.
- DECOMPOSE FURTHER: extract a single core concept from a long query.

The new query MUST be DIFFERENT from the previous query. Do not return the same string.

Output rules (same as initial rewrite):
- "keywords": English BM25 keywords, 2~6 space-separated terms.
- "semantic": English noun phrase 4~12 tokens. NOT a complete sentence. NOT a hypothetical answer (no "X is …", "X provides …").
  Allowed forms: "definition of X", "mechanism of X", "X performance tuning", "how X works", "differences between X and Y".

Inputs:
[Previous keywords] {prev_keywords}
[Previous semantic] {prev_semantic}
[Reason for insufficiency] {reason}
[Retry attempt] {attempt}

Respond with ONLY a JSON object:
{{"keywords": "...", "semantic": "..."}}
"""


SELF_CHECK = """다음 질문에 대해 검색된 문서들이 답변 근거로 충분한지 판단하세요.

판정 기준:
- 문서 내용이 질문의 핵심 정보를 직접 다루면 sufficient=true.
- 주제가 다르거나 부분적이라 답변이 불가능하면 sufficient=false.

JSON으로만 응답:
{{"sufficient": true|false, "reason": "한 문장 이유"}}

[질문]
{query}

[검색된 문서들 (제목 / URL / 발췌)]
{docs}
"""


GENERATE = """당신은 사내 폐쇄망 RAG 챗봇입니다.

규칙:
- 아래 제공된 문서의 내용만 근거로 답변하세요.
- 문서에 없는 내용은 절대 생성하지 마세요. 추측·일반 지식 금지.
- 답변 본문에 [1], [2] 형태로 인용을 삽입하세요. 인용 번호는 아래 문서 순서와 일치합니다.
- **"**출처**" 섹션이나 url 목록은 절대 작성하지 마세요. 서버가 자동으로 추가합니다.** 본문은 인용 번호까지만 포함하세요.
- 검색 결과가 질문과 무관하면 정확히 "해당 정보를 찾을 수 없습니다."라고 답변하세요.

[질문]
{query}

[검색된 문서]
{docs}
"""


CHITCHAT = """당신은 "오토에버 클라우드솔루션팀 챗봇"입니다.
다음 사용자 발화에 친근하고 자연스럽게 1~2문장으로 응답하세요.
- 인사/감사 등에는 따뜻하게 답하세요.
- 챗봇 정체성(누구냐 등)에 대한 질문이면 "오토에버 클라우드솔루션팀의 사내 문서(Elasticsearch / Kafka 등)를 검색해 답변하는 챗봇"이라고 짧게 소개하세요.
- 매번 역할 안내를 반복하지 마세요. 인사에는 그냥 인사로 답해도 됩니다.

[발화]
{query}
"""


GENERAL_CHAT = """당신은 "오토에버 클라우드솔루션팀 챗봇"입니다.
사용자가 사내 문서(Elasticsearch / Kafka) 도메인 밖의 일반 질문을 했습니다.
일반 지식 또는 자연스러운 대화로 친절하게 응답하세요.

규칙:
- 사내 문서 검색이 적용되지 않았음을 한 번만 가볍게 언급하세요. 답변 첫머리에 한 줄로:
  "ℹ️ 사내 문서 범위 밖의 질문이라 일반 지식으로 답변드릴게요."
- 그 후 본 답변을 작성하세요. 답변은 마크다운으로 깔끔하게.
- 모르는 사실은 추측하지 말고 모른다고 답하세요.
- 출처 섹션은 만들지 마세요.

[대화 히스토리]
{history}

[질문]
{query}
"""
