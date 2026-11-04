# Исследование соревнования: Google Gemma 4 Developer Agent

Дата сбора: 2026-11-04. Slug: `gemma-4-developer-agent`.
Сокращения: `C` = https://www.kaggle.com/competitions/gemma-4-developer-agent ; `D/N` = `C/discussion/N` ; `K:a/b` = https://www.kaggle.com/code/a/b .
Страницы соревнования прочитаны через `kaggle competitions pages list --content`. Обсуждения прочитаны через `kaggle competitions topics show`.
Метка **[НЕ ПРОВЕРЕНО]** = утверждение участника без подтверждения хоста.

## 1. Лимиты официального прогона

- Общий лимит: 12 часов на все задачи. Цитата (C/overview/evaluation): "Your agent has a limit of 12 hours to submit patches for all tasks, inclusive of sandbox setup time, but excluding time for patch validation. You may, but are not required, to set per-task time limits in your submission's `eval_config.yaml`."
- Число скрытых задач: около 120. Цитата (C/data): "There are about 120 tasks in the test set, evenly divided between the public and private splits. The test set was curated from a set of private repositories."
- Публичная часть = 58 задач. Это вывод из шагов лидерборда (усечение до 2 знаков). Источник: K:dariushafshar/gemma-4-lb-58-tasks-bronze-is-a-124-team-tie . Значит, всего примерно 116 задач.
- Public и private идут в ОДНОМ 12-часовом прогоне. Хост Ryan Holbrook (D/745805): "Both public and private tasks are evaluated during the same 12 hour run." Повторного прогона после дедлайна нет.
- Старт vLLM входит в 12 часов. Хост ответил "Yes" на вопрос 6 "Does the 12-hour limit include vLLM server start-up?" (D/745805). Время в очереди НЕ входит (D/747519, D/748443).
- Задачи идут последовательно, один агент за раз. Источник: Ayush Thakur, D/747857 ("Kaggle ran one agent at a time"). Также K:verracodeguacas/gemma-4-agent-budget-fit-single-agent ("about 120 tasks one after another"). **[НЕ ПРОВЕРЕНО хостом]**
- Железо: 4x NVIDIA L4, 96 GB VRAM, vLLM 0.19.1, `tensor_parallel_size=4`, `max_model_len=32768`. Источник: страница "Upgraded Accelerators" (C/overview), `docs/HARNESS_README.md` §3.1, K:ryanholbrook/getting-started-gemma-4-developer-agent . Песочница: 4 GiB RAM, 2 vCPU, без сети.
- `eval_config.yaml` соблюдается в фазе агента. Хост (D/748428): "Yes, the eval_config.yaml applies when the agent is running." Сверху значения не ограничиваются. Ограничение даёт только общий лимит 12 ч.
- Проверка патча (pytest) использует фиксированный таймаут 300 с. Хост (D/748428): "A participant's eval_config.yaml doesn't affect scoring... The pytest run in the scorer gets a timeout of 300 seconds". Старое мнение "`timeout_seconds` ограничивает pytest грейдера" устарело (K:hitarthjain0/... , K:verracodeguacas/...).
- Поля `eval_config.yaml`: `timeout_seconds` (одна команда), `max_tool_calls`, `max_time_minutes`, `max_turns`. По умолчанию в `inference.py`: 300 с, 100 вызовов, 60 мин, 500 ходов (HARNESS_README §7.1).
- Таймаут одной задачи: харнесс сохраняет текущий `git diff` и отправляет его на проверку (HARNESS_README §8.1). Пустое дерево = 0.
- Превышение 12 ч: вся отправка падает ("Notebook Timeout"), частичного счёта нет. Nir Aberman (D/752017): "It ended with Notebook Timeout after about 12 hours: no partial score, and the daily slot was used." Хост отказался ставить 0 только недоделанным задачам (D/749597, D/745805).
- Реальная длительность прогонов: 8.9–11.9 ч (Arya Arun, D/752017). Запас мал. **[НЕ ПРОВЕРЕНО]**
- Лимит отправок: 1 в день. Только 1 отправка может ждать в очереди (Dustin, D/747519). Очередь была до 33 ч (D/747519).
- Доступные модели: `gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct`. Одна модель на всю отправку. LoRA: PEFT `.safetensors`, rank <= 128, архив < 3 GiB (C/overview, HARNESS_README §3).

## 2. Лидерборд (скачан 2026-11-04 19:59 UTC, 4669 команд)

| Счёт | Задач из 58 | Команд |
|---|---|---|
| 0.29 | 17 | 1 (Yurnero) |
| 0.24 | 14 | 9 |
| 0.22 | 13 | 65 |
| 0.20 | 12 | 314 |
| 0.18 | 11 | 721 |
| 0.17 | 10 | 838 |
| 0.15 | 9 | 678 |
| 0.13–0.00 | 0–8 | 2043 |

Источник: `kaggle competitions leaderboard gemma-4-developer-agent -d`.
- Сейчас на 0.24 только 9 команд. Пик плотности: 0.17–0.18.
- Малый трек (E2B/E4B): лучший счёт 0.06 на 2026-10-20 (K:ryanholbrook/gemma-4-developer-agent-small-model-leaderboard).

Почему много одинаковых счетов:
1. Шаг счёта крупный. Одна задача = 0.0172. Возможны только 10–15 значений.
2. Шум большой. Один и тот же zip дал 7–13 задач из 58. SD одного прогона = 1.4–1.8 задачи (K:dariushafshar/...). Ещё пример: один zip дал 0.18 и 0.01 (D/750521).
3. Публичные конфиги совпадают. Почти все сильные публичные ноутбуки — потомки Budget-Fit / SuperAgent с одинаковыми настройками (см. §3). K:honghanhhh/gemma-4-baseline-lb-0-24 копирует конфиг байт в байт.
4. Команды много раз отправляют близкие версии. Лидерборд показывает лучший прогон. Это "максимум из N жребиев".
- Byron Galbraith (D/748051): распределение объясняется биномиальным шумом. **[НЕ ПРОВЕРЕНО]** Мы не знаем, какой конфиг у каждой из 9 команд.
- Совет K:dariushafshar/...: сравнивайте конфиги по среднему из >= 2 прогонов. Для второго финального выбора берите другой агент, не копию.

## 3. Публичные ноутбуки: сравнение конфигов

Код скачан `kaggle kernels pull`, прочитан как текст, не запускался. Голоса — на 2026-11-04. Счёт — заявка автора или перепись K:dariushafshar.

| Ноутбук (голоса) | Счёт | Модель | temp / top_p | thinking / include | max_out | Бюджет: мин / вызовы / cmd с / ходы | Агенты, LoRA |
|---|---|---|---|---|---|---|---|
| K:ryanholbrook/getting-started-gemma-4-developer-agent (2285) | — (официальный) | 31B | см. sample | — | — | sample: 1 / 10 / 300 / 50 | coder + analyzer (agent_tool); LoRA-заглушки (lora_B=0) |
| K:flexonafft/gemma-4-budget-aware-coding-agent-v2 (494) | V10: 0.22 (1 из 3 прогонов; копии 7 и 9 задач) | 31B | 0.2 / 0.95 | 4096 / true | 8192 | 8 / 28 / 240 / 80 | 1 агент, без LoRA |
| K:romanrozen/gemma-eda-baseline-for-a-start-lb-top-1 (253) | 0.12; копии 4–7 задач | 31B | 0.2 / 0.95, top_k 40, seed 42 | 4096 / false | 8192 | нет eval_config | coder + analyzer, graph-инструменты |
| K:verracodeguacas/gemma-4-agent-budget-fit-single-agent (235) | 0.18; 10 прогонов, среднее 0.166 | 31B | 0.2 / 0.95 | 4096 / true | 8192 | 8 / 28 / 240 / 80 | 1 агент |
| K:lucifer19/black-cat-swe-agent-pack-instinct (143) | якорь v10: 0.12 | 31B | — | — | — | 5 / 100 / 300 / 250 | coder + analyzer + skill локализации |
| K:prvsiyan/gemma-and-the-shape-of-doubt (129) | 0.18 | 31B | 0.2 / 0.95 | 4096 / true | 8192 | 8 / 28 / 240 / 80 | копия flexonafft V10 |
| K:dmitriigluzdov/gemma-4-task-soundness-serving-and-evaluation (116) | 0.18 | 31B | 0.2 / 0.95 | 4096 / true | 8192 | 8 / 28 / 240 / 80 | Budget-Fit без изменений |
| K:zhukovoleksiy/gemma-4-walkthrough-findings-submission (99) | v2: 0.10; v3: таймаут 12 ч | 31B | 1.0 / 0.95, top_k 64 | 4096 / true | 8192 | 7 / 28 / 240 / 80 | 1 агент (v4) |
| K:jvnhonglei/gemma-4-superagent-reproduction (81), K:matterhorn3838/gemma-4-superagent (67) | SuperAgent: 0.18 (по словам flexonafft) | 31B | 0.2 / 0.95 | 4096 / true | 8192 | 8 / 28 / 240 / 80 | 1 агент |
| K:honghanhhh/gemma-4-baseline-lb-0-24 (66) | 0.24 (в заголовке) | 31B | 0.2 / 0.95 | 4096 / true | 8192 | 8 / 28 / 240 / 80 | 1 агент; только правки промпта |
| K:leoprovorov/single-agent-anytime-patch (17) | не оценён | 31B | 0.2 / 0.95 | 4096 / true | 8192 | 8 / 28 / 240 / 80 | 1 агент |
| K:boristown/gemma-4-e2b-lora-fine-tuning-swe-agent-baseline (40) | лучший 0.06 | E4B | 0.2 / 0.95 | 4096 / true | 8192 | ? / ? / 240 / 80 | LoRA r=8, alpha=8 |
| **Наш** `sample_submission/` | — | 31B | 0.2 / 0.95 | **6144** / true | **12288** | 8 / **100** / 300 / 200 | coder + analyzer |

Общие идеи промптов в сильных ноутбуках (K:flexonafft/..., K:verracodeguacas/..., K:honghanhhh/...):
- План по вызовам. Вызовы 1–8: найти и прочитать. Правка исходника обязательна до вызова 12. После вызова 20: не искать новое.
- "Contract first". Скопировать из issue точные имена, текст ошибок, тип исключения, HTTP-коды.
- Graph-инструменты запрещены в промпте. Причина: они принимают только полные id и обычно ничего не возвращают.
- Тесты не трогать. Временные файлы класть в `/tmp`. Не запускать весь `pytest`. Запускать один файл с `-k`.
- Перед `submit_patch` выполнить `git status --short && git diff`. Проверить `patch_size > 0`.
- Менять все параллельные реализации: sync/async, Client/AsyncClient, `__all__` (K:honghanhhh/...).
- Anytime patch: первая правка до вызова 8; `read_file` диапазонами <= 80 строк; короткое мышление во время чтения (K:leoprovorov/single-agent-anytime-patch).

## 4. Обсуждения: баги, трюки, что помогает

Что помогло (локальные абляции на 129 задачах, Ayush Thakur, D/746250):
- Thinking off: 23/129. Thinking on: 48–51/129. Скрыть thoughts (`include_thoughts: false`): 35/129. Мышление и `include_thoughts: true` обязательны.
- Temperature 0.0/0.2/0.4/1.0: 47/48/51/51. Это шум.
- Thinking budget 2048/4096/6144: 50/48/48. 6144 не дал выигрыша.
- Бюджет 50 → 25 вызовов: −14 задач, медиана времени −29%. Analyzer sub-agent: +1 задача (шум). Graph retrieval: пользы нет.
- Калибровка: 5.5 мин на Kaggle ~ 4.4 мин на B40. 5.5 мин x 120 задач = 11 ч (D/747857).
- Одна модель сильнее не помогает. DeepSeek V4 Pro и Qwen3.6-27B в том же харнессе: +4–5 задач из 124 (D/747168).
- 12B хуже 31B в 2 раза и не быстрее. 12B с Budget-Fit превысил 12 ч (D/750974, K:dmitriigluzdov/gemma-4-public-agent-ideas-and-evidence).
- Один агент лучше пары coder+analyzer: 27/70 против 16/70 (K:dmitriigluzdov/gemma-4-task-soundness-serving-and-evaluation).
- Без правки до вызова 12 задачи падают в 87% (K:verracodeguacas/...). 38% задач с базовым промптом не меняют ни один файл (D/749198).
- Пустой патч в 28% прогонов. Переполнение контекста: 75 случаев, все с пустым патчем (K:leoprovorov/single-agent-anytime-patch, данные Dmitrii Gluzdov).

Баги харнесса:
- Мысли терялись между вызовами (vLLM читал не то поле). Исправлено около 2026-10-01 (D/744354).
- Wheelhouse v29+: `include_thoughts: false` больше не отключает мышление. Отключает только `thinking_budget: 0` (D/747914, хост подтвердил).
- Переполнение контекста раньше выбрасывало патч (D/744692). По K:zhukovoleksiy/... в wheelhouse 29–31 ошибка больше не теряет патч. **[НЕ ПРОВЕРЕНО]**
- Сжатие контекста срабатывает при `token_threshold=14336` (Getting Started; Ayush Thakur в D/748584). Настроить его нельзя. Резюме теряет вызовы инструментов (D/754783).
- Парсер инструментов vLLM 0.19.1 может зависнуть на аргументе-массиве с `]`. Тогда все следующие задачи дают 0 и прогон может превысить 12 ч (D/753973). **[НЕ ПРОВЕРЕНО хостом]**
- Вызов инструмента внутри незакрытого thought-канала теряется (~1% прогонов, D/753521).
- `edit_file` с `"""` в `old_string` иногда ломается парсером. Модель повторяет вызов до 27 раз (D/747168).
- Nudge в `SequentialAgent` перезапускает первого под-агента. Многоэтапные схемы проигрывают (D/754636).
- Часть публичных задач fastapi не проходит даже с gold patch. Хост: все скрытые задачи проходят с gold patch, репозитории не пересекаются (D/747248, D/752848).
- Skills: были отчёты о 0.00 со skill (D/750350). Другие участники получали ненулевой счёт. **[НЕ ПРОВЕРЕНО]**

LoRA:
- Сообщений о стабильном выигрыше LoRA нет. SFT на траекториях DeepSeek/Qwen: loss падает, решаемость не растёт (D/751596, D/750961, D/753793, D/750713).
- LoRA замедляет декодирование на 15–20% (D/751596). LoRA уменьшает KV-кэш с ~46k до ~7.6k токенов (K:zhukovoleksiy/...). **[НЕ ПРОВЕРЕНО]**
- Две отправки с LoRA r=16 и r=4 дали 0.00. Тот же конфиг без LoRA дал 0.18 (D/755677). Другие получили > 0 с LoRA r=32. Причина неизвестна.
- PEFT не цепляется к QAT-чекпойнту (`Gemma4ClippableLinear`). Участники учат на bf16-копии (D/750713, D/755677).
- Хост: адаптеры применяются. Демо-адаптер покрывает только слой 0 (D/749528).

## 5. Сроки (C/overview/timeline, все в 23:59 UTC)

- 2026-11-12: дедлайн статьи (paper track, необязательно).
- 2026-11-25: Entry Deadline и Team Merger Deadline.
- 2026-12-02: Final Submission Deadline. Можно выбрать 2 финальные отправки.
- Призы: 37k / 18k / 10k USD. Бонус 10k USD за лучший E2B/E4B (D/751104).

## 6. Выводы для нашего конфига (наш анализ, не проверено отправкой)

- Риск таймаута 12 ч. У нас 8 мин и 100 вызовов. Budget-Fit укладывается в 12 ч только из-за лимита 28 вызовов: многие задачи заканчиваются рано. 116 x 8 мин = 15.5 ч. Таймаут даёт 0 всей отправке.
- `thinking_budget 6144` не дал выигрыша в D/746250. Больше мышления = больше времени на вызов.
- `max_output_tokens 12288` оставляет для промпта <= 20480 токенов. Переполнение даёт пустой патч чаще (D/744692).
- Analyzer sub-agent не дал измеримого выигрыша (D/746250, K:dmitriigluzdov/...). Он тратит время и контекст.
- Лидер (0.29) не раскрыл метод. Мы не знаем, что дало +3 задачи.
