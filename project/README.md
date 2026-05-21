# Итоговый проект: Vector DB Benchmark

Тема: NoSQL/Big Data хранилища для векторного поиска на примере `pgvector`,
`Qdrant` и `Milvus`.

Практическая часть: semantic search по Airbnb-объявлениям из InsideAirbnb.
Модель эмбеддингов: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
Сравниваются скорость индексации, latency поиска и recall@10.



---

## Быстрый старт

### 0. Рабочая директория

```bash
cd project/
```

### 1. Собрать Docker-образы

```bash
docker-compose build --no-cache runner visualization
```

> ~5 минут (python:3.12-slim + зависимости).

### 2. Поднять все сервисы

```bash
docker-compose up -d
```

Дождаться `healthy` у всех контейнеров (~60–90 сек):

```bash
watch docker-compose ps
```

`milvus` стартует последним — ждать его.

### 3. Запустить пайплайн

Данные уже лежат в `data/raw/`. Шаги выполнять последовательно.

**3.1 Предобработка** (~1 мин):
```bash
docker exec runner python3 /scripts/preprocess.py /data/raw /data/processed/listings.csv
```

**3.2 Генерация эмбеддингов** (~30 мин на CPU):
```bash
docker exec runner python3 /scripts/generate_embeddings.py \
    /data/processed/listings.csv /data/processed/embeddings.npy
```

**3.3 Индексация** в pgvector + Qdrant + Milvus (~10–20 мин):
```bash
docker exec runner python3 /scripts/index_all.py
```

**3.4 Бенчмарк**:
```bash
docker exec runner python3 /scripts/benchmark.py
```

### 4. Открыть дашборд

```
http://localhost:5050
```

---

## Архитектура

| Сервис | Image | Порт |
|--------|-------|------|
| pgvector | `pgvector/pgvector:pg16` | 5432 |
| qdrant | `qdrant/qdrant:v1.9.2` | 6333 (HTTP), 6334 (gRPC) |
| milvus | `milvusdb/milvus:v2.4.1` | 19530 (gRPC), 9091 (UI) |
| etcd | `quay.io/coreos/etcd:v3.5.5` | — |
| minio | `minio/minio:...` | 9000 |
| runner | python:3.12-slim | — |
| visualization | python:3.12-slim (Flask) | 5050 |

---

## Полезные команды

```bash
# Логи контейнера
docker-compose logs -f runner

# Статус health check
docker inspect --format='{{.State.Health.Status}}' qdrant
docker inspect --format='{{.State.Health.Status}}' milvus

# Войти в runner для отладки
docker exec -it runner bash

# Остановить всё
docker-compose down

# Полный сброс (включая volumes)
docker-compose down -v
```

---

## Данные

Источник: [InsideAirbnb](http://insideairbnb.com/get-the-data/). В текущем
локальном наборе лежат 13 city exports, полученных из списка `download_data.sh`;
скрипт поддерживает скачивание 18 городов в свежую директорию. Скачать заново:
`bash scripts/download_data.sh`
