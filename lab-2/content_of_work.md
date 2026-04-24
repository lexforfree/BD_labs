# Лабораторная 2 — Шпаргалка

## О чём проект

Мы берём статистику NFL (американский футбол, 2014–2023, ~400k розыгрышей) и отвечаем на вопрос:

> **Как вероятность победы команды меняется в зависимости от четверти, попытки (down), позиции на поле и разницы в счёте?**

Один и тот же вопрос решается тремя разными инструментами, чтобы сравнить скорость:
`MapReduce` → `Hive` → `Spark`. Оркестрирует всё `Airflow`, результаты видны на дашборде.

---

## Данные

**Датасет:** nflverse — официальная статистика NFL по каждому розыгрышу.

После препроцессинга (`scripts/preprocess.py`) оставляем 6 колонок:

| Колонка | Что это |
|---------|---------|
| `qtr` | Четверть (1–4) |
| `down` | Попытка (1–4, в амер. футболе есть 4 попытки продвинуться на 10 ярдов) |
| `yardline_100` | Расстояние до зачётной зоны соперника (1–100 ярдов) |
| `score_differential` | Разница в счёте с точки зрения команды с мячом |
| `win` | 1 — команда с мячом выиграла матч, 0 — проиграла |

**Бакетинг** — группируем значения в диапазоны:
- `score_diff_bucket`: `le-14` / `-13to-7` / `-6to-1` / `0` / `1to6` / `7to13` / `ge14`
- `field_bucket`: `0-25` / `26-50` / `51-75` / `76-100` (ярды до зоны)

Итого **447 уникальных бакетов** × (quarter, down, score, field).

---

## Архитектура (что крутится в Docker)

```
┌─────────────┐    HDFS RPC    ┌────────────┐
│  NameNode   │◄──────────────►│  DataNode  │
│ (метаданные)│                │  (данные)  │
└─────────────┘                └────────────┘
       ▲
       │ YARN — управление ресурсами
       ▼
┌─────────────────┐   ┌──────────────────┐
│ ResourceManager │   │   NodeManager    │
│  (диспетчер)    │   │ (выполняет задачи│
└─────────────────┘   │  на воркере)     │
                      └──────────────────┘

┌──────────────────┐   ┌──────────────┐
│  Hive Metastore  │   │  HiveServer2 │  ← beeline/JDBC
│  (PostgreSQL)    │◄──│  (движок)    │
└──────────────────┘   └──────────────┘

┌──────────────────┐   ┌──────────────┐
│  Spark Master    │   │ Spark Worker │
│  (координатор)   │◄──│ (исполнитель)│
└──────────────────┘   └──────────────┘

┌──────────────────┐
│  Airflow         │  → планировщик задач
└──────────────────┘

┌──────────────────┐
│  Flask Dashboard │  → http://localhost:5050
└──────────────────┘
```

**PostgreSQL** — общая БД для двух сервисов:
- Hive metastore (схемы таблиц, пути к данным в HDFS)
- Airflow (состояние DAG-ов, логи задач)

---

## Как работает каждый инструмент

### 1. HDFS — распределённое хранилище

Файловая система, которая режет файлы на блоки (по 128 МБ) и раскладывает по датанодам.
Снаружи выглядит как обычная FS: `hdfs dfs -ls /`, `hdfs dfs -cat /data/file`.

```
/data/nfl/processed/nfl_all.csv   ← исходный файл (6.7 MB)
/output/win_probability_mr/       ← результат MapReduce
/output/hive_win_prob/            ← результат Hive
```

---

### 2. MapReduce — классический подход

**Идея:** разбить задачу на две фазы — Map (фильтрация/преобразование) и Reduce (агрегация).

```
CSV на HDFS
    │
    ▼  mapper.py (запускается на каждой строке параллельно)
    │  qtr=3|down=2|score=0|field=26-50  →  TAB  →  win=1
    │
    ▼  Hadoop сортирует и группирует по ключу
    │
    ▼  reducer.py (суммирует wins и total по каждому ключу)
    │  3|2|0|26-50  →  {"wins": 5040, "total": 9406, "win_probability": 0.5358}
    │
    ▼  HDFS: /output/win_probability_mr/part-00000
```

**Streaming** — Hadoop запускает обычные Python-скрипты через stdin/stdout.
Не нужен Java — достаточно написать маппер и редьюсер на Python.

**Время:** ~35 секунд. Медленно, потому что каждый шаг пишет на диск.

---

### 3. Hive — SQL поверх MapReduce

Hive транслирует SQL-запрос в MapReduce-задачи автоматически.
Таблица — это не настоящая таблица, а просто описание структуры файла на HDFS
(EXTERNAL TABLE — Hive читает файл, но не владеет им).

```sql
-- schema.hql — описываем структуру CSV-файла
CREATE EXTERNAL TABLE IF NOT EXISTS nfl_plays (
    qtr INT, down INT, yardline_100 INT,
    score_differential INT, win INT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
LOCATION 'hdfs://namenode:9000/data/nfl/processed/';

-- query.hql — аналитический запрос
INSERT OVERWRITE DIRECTORY 'hdfs://namenode:9000/output/hive_win_prob'
SELECT qtr, down,
       CASE WHEN score_differential <= -14 THEN 'le-14' ... END AS score_diff_bucket,
       ...
       SUM(win) AS wins,
       COUNT(*) AS total,
       ROUND(SUM(win) / COUNT(*), 4) AS win_probability
FROM nfl_plays
WHERE qtr BETWEEN 1 AND 4 AND down BETWEEN 1 AND 4
GROUP BY qtr, down, score_diff_bucket, field_bucket
HAVING COUNT(*) >= 10;
```

**Время:** ~60 секунд. Медленнее MapReduce — дополнительные накладные расходы на трансляцию SQL.

---

### 4. Spark — in-memory вычисления

Spark не пишет промежуточные результаты на диск (по умолчанию).
Данные держатся в памяти воркеров как RDD/DataFrame.

```python
# Читаем CSV из HDFS
df = spark.read.option("header", "true").csv("hdfs://namenode:9000/data/...")

# Добавляем бакеты
df = df.withColumn("score_diff_bucket", when(col("score_differential") <= -14, "le-14")...)

# Группируем и считаем
result = df.groupBy("qtr", "down", "score_diff_bucket", "field_bucket") \
           .agg(sum("win").alias("wins"), count("*").alias("total")) \
           .withColumn("win_probability", col("wins") / col("total")) \
           .withColumn("win_prob_bayes", (col("wins") + 2) / (col("total") + 4))
```

**Время:** ~5.5 секунд. В ~7× быстрее MapReduce — нет записи на диск между шагами.

---

### 5. Airflow — планировщик (DAG)

DAG (Directed Acyclic Graph) — граф задач без циклов. Каждый узел — задача.

```
upload_to_hdfs → run_mapreduce → run_hive → run_spark → generate_report
```

Каждая задача — это `BashOperator` с `docker exec <container> bash /scripts/...`.
Airflow запускает pipeline раз в неделю (или вручную через UI на :8081).

---

## Байесовская оценка (что добавили)

**Проблема MLE:** для редких ситуаций (например, 4-я попытка, выигрываем 14+ очков в 1-й четверти) может быть всего 10 розыгрышей. `9 побед / 10 = 0.90` — но это ненадёжно.

**Решение — Beta-Binomial модель:**

```
Prior: Beta(α=2, β=2) — слабое убеждение, что вероятность близка к 0.5
       эквивалентно "добавить 2 победы и 2 поражения к наблюдениям"

Posterior mean = (wins + α) / (total + α + β)
               = (wins + 2) / (total + 4)
```

| Ситуация | total | wins | MLE | Bayes |
|----------|-------|------|-----|-------|
| Редкий бакет | 10 | 9 | **0.900** | **0.786** (осторожнее) |
| Частый бакет | 9406 | 5040 | **0.5358** | **0.5358** (prior растворяется) |

При большом `total` обе оценки совпадают — prior не играет роли.
При малом `total` Байес "тянет" оценку к 0.5, уменьшая влияние шума.

---

## Итоговое сравнение инструментов

| | MapReduce | Hive | Spark |
|---|---|---|---|
| **Время** | ~35 с | ~60 с | ~5.5 с |
| **Как пишется** | Python mapper + reducer | SQL | Python DataFrame API |
| **Промежуточные данные** | Диск (HDFS) | Диск (HDFS) | Память |
| **Порог входа** | Низкий | Низкий (знаешь SQL) | Средний |
| **Гибкость** | Максимальная | Ограничена SQL | Высокая |

**Вывод:** Spark выигрывает за счёт in-memory вычислений. Hive проще писать, но медленнее даже MapReduce из-за накладных расходов транслятора.

---

## Быстрые команды

```bash
# Посмотреть файлы на HDFS
docker exec namenode hdfs dfs -ls /output/

# Посмотреть сырой вывод MapReduce
docker exec namenode hdfs dfs -cat /output/win_probability_mr/part-00000 | head -5

# Перезапустить только Spark-job
docker exec spark bash /scripts/run_spark.sh

# Логи контейнера
docker logs hiveserver2 --tail 50

# Пересобрать дашборд после правки шаблона
docker compose build visualization && docker compose up -d visualization
```
