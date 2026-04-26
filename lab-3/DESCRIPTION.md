# Лабораторная работа 3 — Мини-поисковик

## Цель

Построить поисковую систему по корпусу статей Русской Википедии (тематика: математика, машинное обучение, статистика) с поддержкой нескольких алгоритмов ранжирования и поиска.

---

## Архитектура

```
prepare_data.sh              — одноразовая подготовка данных (вне Docker)
    │
    ├── download_text.py           — стриминг статей из HuggingFace
    ├── build_index.py             — инвертированный индекс + TF-IDF граф схожести
    ├── precompute_bert.py         — предрасчёт BERT-эмбеддингов
    ├── extract_entities.py        — NER + noun-seq через spaCy → entities.pkl
    ├── build_entity_graph.py      — граф по общим сущностям → entity_graph.pkl
    ├── pagerank/mapreduce_pagerank.py
    ├── pagerank/pregel_pagerank.py
    └── pagerank/ppr.py            — Personalized PageRank (используется онлайн)

docker compose up web        — Flask-приложение на порту 5000
    └── web/
        ├── app.py
        ├── search/vectorizers.py  — TF-IDF, BM25, LSA, BERT
        ├── search/fulltext.py     — DAAT, TAAT
        ├── search/ppr_search.py   — PPR-поиск через spaCy + entity graph
        ├── search/ppr.py          — алгоритм PPR
        └── templates/index.html   — UI + D3.js граф
```

Данные хранятся в `./data/` и монтируются в контейнер как volume.

---

## Данные

**Источник:** `wikimedia/wikipedia`, конфиг `20231101.ru` (HuggingFace Datasets).  
Загрузка в режиме **streaming** — скачиваются только нужные Parquet-чанки, полный дамп (~5 ГБ) не нужен.

**Фильтрация по заголовку** — ключевые слова: `матем`, `статист`, `машинн`, `нейрон`, `вероятн`, `алгебр`, `регресс`, `градиент`, `теорем` и др.  
Итог: ~8 600 статей, первые 5 000 символов каждой.

**Два графа:**
- **TF-IDF similarity graph** (`graph.pkl`) — каждая статья ссылается на 5 наиболее похожих по TF-IDF. Используется для глобального PageRank.
- **Entity co-occurrence graph** (`entity_graph.pkl`) — статьи связаны если разделяют именованные сущности (spaCy NER). Используется для PPR.

---

## PageRank

### Реализация 1 — MapReduce (pure Python)

Файл: `indexer/pagerank/mapreduce_pagerank.py`

Итеративный алгоритм, map и reduce фазы явно разделены:

**Map:** каждый узел отдаёт свой ранг равномерно по исходящим рёбрам:
```
emit(neighbour_id, rank / out_degree)  для каждого соседа
```

**Reduce:** суммируем входящие вклады, добавляем damping:
```
new_rank(v) = (1 - d) / N  +  d * sum(contributions)
```

Dangling nodes (без исходящих рёбер) перераспределяют ранг равномерно по всем узлам.  
Map-фаза параллелизована через `multiprocessing.Pool`.  
Параметры: `d = 0.85`, до 20 итераций, сходимость при `Δ < 1e-6`.

### Реализация 2 — Pregel (BSP через networkx)

Файл: `indexer/pagerank/pregel_pagerank.py`

Модель Bulk Synchronous Parallel (BSP) — суперстепы:

1. Каждый узел **отправляет** `rank / out_degree` всем преемникам
2. Каждый узел **принимает** сообщения, обновляет ранг
3. Если `Δ < порога` — узел голосует «завершить»; останавливаемся когда все проголосовали

Реализовано через `networkx.DiGraph` + явные словари сообщений.

**Сравнение реализаций:**

| | MapReduce | Pregel |
|---|---|---|
| Парадигма | Batch итерации (map → reduce) | BSP суперстепы (send → receive → update) |
| Параллелизм | `multiprocessing.Pool` на map-фазе | Последовательный (networkx — single-thread) |
| Сходимость | Обычно 15–20 итераций | Те же итерации, иная модель остановки |

---

## Векторный поиск

Все методы реализуют интерфейс `fit(articles)` / `query(text, top_k)` и возвращают `[(doc_id, score)]`.

### TF-IDF (своя реализация)

Файл: `web/search/vectorizers.py` → `TFIDFVectorizer`

```
tf(t, d)   = count(t in d) / len(d)
idf(t)     = log((N+1) / (df(t)+1)) + 1       # сглаживание
tfidf(t,d) = tf * idf
score      = cosine(tfidf(q), tfidf(d))
```

### BM25

Файл: `web/search/vectorizers.py` → `BM25Vectorizer`  
Библиотека: `rank-bm25` (BM25Okapi)

```
score(t,d) = idf(t) * tf(t,d) * (k1+1) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))
k1 = 1.5,  b = 0.75
```

Преимущество перед TF-IDF: насыщение TF и нормализация длины документа.

### LSA / SVD

Файл: `web/search/vectorizers.py` → `LSAVectorizer`  
Библиотека: `sklearn` (TfidfVectorizer + TruncatedSVD)

1. Матрица TF-IDF `D` (~8600 × 50000)
2. `TruncatedSVD`: `D ≈ U · Σ · Vᵀ`, `k = 100` компонент
3. Поиск — косинусное сходство в латентном пространстве

Улавливает синонимию и тематическое сходство.

### BERT

Файл: `web/search/vectorizers.py` → `BERTVectorizer`  
Модель: `paraphrase-multilingual-MiniLM-L12-v2`

Эмбеддинги предрассчитываются один раз и кэшируются в `bert_embeddings.npy`.  
При старте Flask модель загружается в фоновом потоке (`threading.Thread`).  
Поиск: матричное произведение `embeddings @ query_emb.T`.

---

## Полнотекстовый поиск

Используют инвертированный индекс `{term: [(doc_id, tf), ...]}`.

### DAAT — Document-At-A-Time

```
candidates = intersection(posting_lists для всех термов запроса)
score(d) = sum(tf(t, d) для t в запросе)
```

Возвращает документы, содержащие **все** слова запроса (AND).

### TAAT — Term-At-A-Time

```
for term in query:
    for (doc_id, tf) in index[term]:
        scores[doc_id] += tf
```

Возвращает документы с **любым** словом запроса (OR), больше результатов.

| | DAAT | TAAT |
|---|---|---|
| Критерий | AND | OR |
| Обход | по документам | по термам |
| Результатов | меньше | больше |

---

## PPR — Personalized PageRank (вдохновлено HippoRAG)

Файлы: `web/search/ppr_search.py`, `web/search/ppr.py`

### Идея

Обычный PageRank не зависит от запроса. PPR персонализирует обход графа: стартовая вероятность сосредоточена в документах, релевантных запросу («seed nodes»), и распространяется по графу сущностей.

### Пайплайн

```
Запрос
  │
  ▼  spaCy ru_core_news_md (NER + NOUN-sequences)
Сущности запроса: ["градиентный спуск", "нейронная сеть"]
  │
  ▼  поиск в entities.pkl
Seed-документы (статьи, содержащие эти сущности)
  │
  ▼  персонализированный вектор p
seeds → высокий вес, остальные → 0
  │
  ▼  итерации PPR
r = (1-d)*p  +  d * A_weighted * r
  │
  ▼
Ранжированные документы
```

### Entity graph

Строится в `build_entity_graph.py`:
- spaCy извлекает именованные сущности и NOUN-последовательности из каждой статьи
- Инвертированный индекс: `entity → [doc_ids]`
- Ребро `doc_A → doc_B` с весом = кол-во общих сущностей
- Слишком редкие (< 2 документов) и слишком частые (> 500) сущности игнорируются

### PPR формула

```
r_new(v) = (1-d) * p(v)  +  d * Σ_{u→v} r(u) * w(u,v) / Σ_w w(u,w)
d = 0.85, до 50 итераций, сходимость при Δ < 1e-6
```

### Связь с HippoRAG

HippoRAG 2 использует ту же идею PPR, но строит граф через OpenIE-триплеты с LLM. Мы заменяем LLM на spaCy NER — дешевле, работает офлайн, достаточно для лабораторных целей.

---

## Визуализация графа (D3.js)

При выборе метода PPR рядом с результатами поиска отображается интерактивный граф:

- **Узлы** — статьи, размер ∝ PPR score
- **Цвет** — роль узла: seed+результат (красный), только seed (оранжевый), результат PPR (синий), сосед (серый)
- **Рёбра** — entity co-occurrence, толщина = число общих сущностей
- **Интерактив** — zoom/pan, drag узлов, hover-тултип (название + score), клик → Wikipedia

Реализовано на D3.js v7 (force-directed simulation), данные передаются из Flask как JSON.

---

## UI

Flask-приложение, порт 5000. Dropdown с группами методов:
- Векторный поиск: TF-IDF, BM25, LSA/SVD, BERT
- Полнотекстовый: DAAT, TAAT
- Графовый: PPR

Статистика запроса: метод, время в мс, "Топ N из M" или "Найдено совпадений: N".

---

## Запуск

```bash
# Один раз — подготовка данных (~30–60 мин)
./prepare_data.sh

# Поднять веб-интерфейс
docker compose up web
# → http://localhost:5000
```
