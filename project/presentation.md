# Презентация: Vector DB Benchmark

## Слайд 1. Тема

- Итоговая работа по курсу Big Data
- NoSQL и Big Data хранилища для векторного поиска
- Практика: semantic search по Airbnb listings
- Сравнение: pgvector, Qdrant, Milvus

Заметки:

В работе рассматривается векторный поиск как Big Data задача. Практическая часть
сравнивает три подхода: расширение PostgreSQL, отдельную vector database и более
масштабируемую vector-first систему.

## Слайд 2. Зачем нужен векторный поиск

- Классический поиск хорошо работает по точным словам и фильтрам
- Semantic search ищет объекты, близкие по смыслу
- Текст объявления превращается в embedding-вектор
- База ищет ближайшие векторы по cosine similarity
- Подход используется в рекомендациях, RAG и поиске по документам

Заметки:

Главная идея: похожие тексты имеют близкие embedding-векторы. Поэтому запрос
"cozy apartment near downtown" может найти релевантные объявления даже без
точного совпадения всех слов.

## Слайд 3. Pipeline эксперимента

- Raw data: CSV-файлы InsideAirbnb
- Preprocess: очистка цены, координат, описаний
- Text field: name + description + city + rooms + price
- Embeddings: all-MiniLM-L6-v2, 384 dimensions
- Indexing: загрузка в pgvector, Qdrant и Milvus
- Serving: Flask dashboard с поиском и картой

Заметки:

Один и тот же набор embeddings используется во всех трех базах. Это важно для
честного сравнения: отличается только хранилище и индекс, а не ML-модель.

## Слайд 4. Архитектура стенда

| Компонент | Роль |
| --- | --- |
| runner | ETL, embeddings, indexing, benchmark |
| PostgreSQL + pgvector | SQL-хранилище с vector extension |
| Qdrant | Отдельная vector database |
| Milvus + etcd + MinIO | Vector DB для масштабируемого сценария |
| visualization | Flask UI: поиск, карта, benchmark |

Заметки:

Стенд запускается в Docker Compose. Для Milvus нужны дополнительные компоненты:
etcd для metadata и MinIO как object storage.

## Слайд 5. Чем отличаются системы

| Система | Тип | Когда выбирать |
| --- | --- | --- |
| pgvector | PostgreSQL extension | MVP, SQL, простая интеграция |
| Qdrant | Vector-first DB | Отдельный semantic search service |
| Milvus | Scalable vector DB | Большие коллекции и высокая нагрузка |

Заметки:

pgvector проще встроить в существующее приложение. Qdrant удобен как отдельный
API-сервис. Milvus сложнее в эксплуатации, но лучше подходит для роста объема.

## Слайд 6. Данные и модель

- Источник: InsideAirbnb listings
- После очистки: 103 171 объявление
- Поля: name, description, city, price, bedrooms, bathrooms, lat/lon
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Размерность вектора: 384
- Индекс: HNSW во всех трех системах

Заметки:

Модель компактная и подходит для CPU-эксперимента. HNSW выбран потому, что это
распространенный алгоритм approximate nearest neighbor search.

## Слайд 7. Скорость индексации

| Система | Время | Примерная скорость |
| --- | ---: | ---: |
| pgvector | 1442.8 s | 72 rec/s |
| Qdrant | 165.7 s | 623 rec/s |
| Milvus | 57.0 s | 1809 rec/s |

Заметки:

На индексации специализированные vector-first системы быстрее. pgvector работает
внутри PostgreSQL, поэтому выигрывает в интеграции, но проигрывает по скорости
массовой загрузки и построения индекса.

## Слайд 8. Latency и recall

| Система | P50 | P95 | P99 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| pgvector | 16.65 ms | 27.15 ms | 38.16 ms | 1.000 |
| Qdrant | 59.69 ms | 84.85 ms | 88.30 ms | 0.889 |
| Milvus | 11.37 ms | 18.61 ms | 48.48 ms | 0.909 |

Заметки:

Milvus показал лучшую медианную задержку и P95. pgvector оказался очень
конкурентным на объеме около 100 тысяч записей. Qdrant медленнее в этом запуске,
но удобен по API и payload filtering.

## Слайд 9. Что показал поиск в UI

- Пользователь вводит текстовый запрос
- Запрос кодируется той же embedding-моделью
- Можно выбрать одну или несколько баз
- Top-K настраивается на фронте
- Карточки показывают описание, параметры, координаты и ссылку на listing

Заметки:

UI демонстрирует не только offline benchmark, но и реальный пользовательский
сценарий: запрос, embedding, поиск в базе, выдача результатов и отображение на
карте.

## Слайд 10. Облачные и российские варианты

- AWS: OpenSearch vector search
- Google Cloud: Vertex AI Vector Search
- Azure: Azure AI Search
- Managed vector DB: Pinecone, Zilliz Cloud, Qdrant Cloud
- Российский контекст: self-hosted pgvector/Qdrant/Milvus, Managed OpenSearch,
  Kubernetes или VM у локального провайдера

Заметки:

В российском контексте важны локализация данных, доступность образов и пакетов,
а также возможность on-premise или self-hosted развертывания.

## Слайд 11. Выводы

- pgvector: лучший старт, если приложение уже использует PostgreSQL
- Qdrant: хороший выбор для отдельного semantic search API
- Milvus: сильный вариант для больших объемов и масштабирования
- Выбор зависит не только от latency, но и от эксплуатации
- Важны обновления индекса, фильтры, стоимость RAM и требования к данным

Заметки:

Нет универсально лучшей базы. Для учебного стенда все три системы применимы, но
их оптимальные сценарии разные.

## Слайд 12. Что улучшить дальше

- Exact ground truth через brute force или FAISS
- Concurrent benchmark для 5/10/50 пользователей
- Измерение RAM и disk usage
- Подбор HNSW параметров M, ef_construction, ef_search
- Фильтры по городу, цене и числу комнат
- Hybrid search: keyword + vector + reranking

Заметки:

Следующий шаг - сделать benchmark ближе к production: параллельные запросы,
измерение ресурсов, подбор параметров и более строгая оценка качества.
