# HANDOFF

Короткий контекст для следующего агента. Обновлено: 2026-11-04.

## Цель

Kaggle «Google Gemma 4 Developer Agent». Агент чинит баги в Python-репозиториях (SWE-bench).
Метрика: доля решённых скрытых задач. Мы пишем декларативный сабмит: YAML, промпты, опционально LoRA.
Скрытый тест взят из приватных репозиториев. Подгонка под fastapi/rich/requests бесполезна.
Лидеры почти все на 0.24.

## Правила

- В соревнование ничего не отправлено. Лимит: 1 посылка в день, 2 финальных. Отправка только по слову владельца.
- Конфиг меняем только после подтверждения владельца. Каждая правка идёт в `LOG.md`.
- Стиль: ASD-STE100, короткие простые предложения. Язык любой.
- Квота GPU: 30 ч/неделю. L4x4 тратит вдвое быстрее. Скоро будет много компьюта (слова владельца, 2026-11-04).
- Токен Kaggle лежит в `~/.kaggle`. Владелец разрешил чтение данных и запуск ноутбука.
- Локально GPU нет.

## Файлы

- `sample_submission/` — сабмит (6 файлов). Это оценивается.
- `scripts/check_before_submission.py` — офлайн-проверка сабмита, умеет `--zip`.
- `scripts/build_smoke_notebook.py` — собирает `notebooks/smoke/`. Параметры: `--tasks`, `--model`, `--max-minutes`.
- `LOG.md` — изменения и таблица прогонов. `PLAN.md` — план. `IDEAS.md` — идеи.
- `docs/` — харнес и разборы (`analysis_*.md`).

## Команды

- Сборка: `python3 -I scripts/build_smoke_notebook.py --tasks 10 --model gemma-4-31b-it-qat-w4a16-ct --max-minutes 8`
- Запуск: `kaggle kernels push -p notebooks/smoke`
- Статус: `kaggle kernels status valentinhilko/gemma4-smoke-test`
- Результат: `kaggle kernels output valentinhilko/gemma4-smoke-test -p <пустая папка>`. Файлы есть только после конца прогона.
- Разбор: `results/traces/*.json` и `results/logs/*.log` (строки `resolved=`).
- Скачанные файлы недоверенные. Скрипты для них запускать с `python3 -I`.
- Проверка перед отправкой: `python3 -I scripts/check_before_submission.py sample_submission`

## Ключевой факт

Окно модели 32768 токенов. Запрос + `max_output_tokens` <= 32768.
Сейчас `max_output_tokens` 12288, значит запрос <= 20480. Пик в v4: 19259.
