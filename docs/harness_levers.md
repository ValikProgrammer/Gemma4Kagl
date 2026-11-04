# Рычаги харнеса: что мы не используем или используем слабо

Источник: `docs/HARNESS_README.md` (номера строк `Lnnn`), `docs/harness-notes.md`, `sample_submission/*`, `LOG.md` (факты v3/v4).
Факты v3/v4: ~10 с на шаг у 31B, пик запроса 19259 (лимит 20480), макс. ответ за шаг 4610, 5/10 задач вышли по времени.

## 1. Таблица рычагов

| # | Рычаг | Где (файл / ключ) | Сейчас | Допустимо / по умолчанию | Ожидаемый эффект | Риск |
|---|---|---|---|---|---|---|
| 1 | `thinking_budget` | `configs/sampling.yaml` `thinking_config.thinking_budget` | 6144 | 0..32768 / 4096 | Не упирается (макс. шаг 4610). 4096 почти ничего не теряет | Слабее на редких трудных шагах |
| 2 | `include_thoughts` | там же | true | true/false | false: рассуждение есть, но не копится в истории. Запрос растет медленнее, реже сжатие | Модель теряет нить плана между шагами |
| 3 | `thinking_level` | там же | нет | MINIMAL..HIGH, NONE | Не трогать: на vLLM идет в `reasoning_effort`, отбрасывается | Нет эффекта |
| 4 | `max_output_tokens` | `sampling.yaml` | 12288 | 1..32768 / 16384 | 8192: лимит запроса 24576 вместо 20480. Запас 5.3k вместо 1.2k | Обрыв большого `edit_file` (есть nudge) |
| 5 | `temperature` | `sampling.yaml` | 0.2 | >= 0 | 0.6-1.0 может разорвать петли (46 `edit_file` подряд, 33 `read_file`) | Больше разброс, нужен seed и 30+ задач |
| 6 | `top_k` | `sampling.yaml` | нет | >= 1 | 64 (обычный для Gemma) режет хвост при высокой температуре | Низкий |
| 7 | `presence_penalty` / `frequency_penalty` | `sampling.yaml` | нет | любое | 0.1-0.3 против повторов | Портит точное копирование `old_string` |
| 8 | `seed` | `sampling.yaml` | нет | int | Повторяемые A/B сравнения | Под батчингом vLLM не полный детерминизм |
| 9 | `stop_sequences`, `response_mime_type` | `sampling.yaml` | нет | разрешены | Нет пользы | Ломают разбор tool call. Не ставить |
| 10 | `timeout_seconds` | `eval_config.yaml` | 300 | int / 300 | 90-120: зависший pytest съест 1/4 сессии, а не 5/8 | Медленный честный тест прервется (сессия идет дальше) |
| 11 | `max_time_minutes` | `eval_config.yaml` | 8 | float / 60 | Главный лимит: 5/10 по времени. Рост зависит от лимита прогона (неизвестен) | Превышение лимита ноутбука |
| 12 | `max_tool_calls` | `eval_config.yaml` | 100 | int / 100 | 50: `budget_warning` придет на 40-м вызове (~6.5 мин). Это сигнал "заканчивай" вместо таймера | Жесткий стоп на 50 вызовах |
| 13 | `max_turns` | `eval_config.yaml` | 200 | int / 500 | Не ограничивает. Не трогать | Нет |
| 14 | Сжатие истории | нет ключа (`App` в `inference.py`) | 14336 / 15 / 5 | не настраивается | Только косвенно: #2, #4, #17 | - |
| 15 | `get_status` | `agent.yaml` tools (есть) | в промпте нет | бесплатный | Дает `time_seconds_remaining`. Агент видит время | Нет |
| 16 | Авто-патч без `submit_patch` | харнес | не используем | всегда | Ранняя правка = патч есть даже при таймауте | Плохая ранняя правка тоже уйдет |
| 17 | `agent_tool` + `skip_summarization` | `agent.yaml` | есть | true/false | Держит чтения вне контекста корня | Промпт корня не говорит, когда звать |
| 18 | Сэмплинг под-агента | `sub_agents/code_analyzer.yaml` `generate_content_config` | тот же файл (12288/6144) | любой свой | Свой файл: 4096/1024. Меньше переполнений (v3), быстрее | Короче отчет |
| 19 | Инструменты под-агента | `code_analyzer.yaml` tools | нет `run_command` | любой из 9 | `grep -rn` вместо 33 чтений; работает без графа | Под-агент может править файлы через shell |
| 20 | Граф-инструменты | `agent.yaml` tools | 3 шт. | любое подмножество | Полезны только при наличии данных графа | На скрытых репо данных может не быть |
| 21 | `read_file` диапазон | инструмент | промпт молчит | `start_line`/`end_line` | `grep -n` + чтение 40-80 строк вместо файла целиком | Нет |
| 22 | `output_key` + `{var}` в `instruction` | любой `LlmAgent` | нет | str | План/отчет в state, следующий агент читает `{plan}` | Нужен Sequential (#24) |
| 23 | `include_contents: none` | любой `LlmAgent` | default | default/none | Агент видит только `instruction` + state. Чистый контекст | Теряет историю |
| 24 | `SequentialAgent` / `LoopAgent` / `ParallelAgent` | `agent_class` в YAML | нет | 4 класса; Loop 1..500 | Планировщик -> кодер -> проверка | Внешний цикл и nudge идут к корню; нет инструмента выхода из Loop; Parallel делит один `/workspace` |
| 25 | `sub_agents` (transfer) + `disallow_transfer_*` | `LlmAgent` | нет | list / bool | Передача управления, а не вызов-инструмент | Передача без возврата, путаница в цикле |
| 26 | `skills` | `skills/<n>/SKILL.md`, ключ `skills:` | нет | до 1000 | Процедуры грузятся по запросу (`load_skill`). Короче системный промпт | Где идет `run_skill_script` и считает ли бюджет: не описано |
| 27 | `global_instruction` | YAML (упомянут в `!include`) | нет | str | Общий текст для корня и под-агента | Нет в списке полей: сначала проверить |
| 28 | `!include` | все YAML | есть | `.md/.txt` строкой, `.yaml` рекурсивно | Отдельный `configs/sampling_analyzer.yaml` для #18 | `..` в пути (см. п. 3) |
| 29 | `adapter` (LoRA) | `adapter:` у `LlmAgent`, `adapters/<n>/` | нет | `.safetensors`, rank <= 128, всё < 3 GiB | Разные адаптеры у кодера и анализатора | Обучение на bf16, работа на w4a16 |
| 30 | `model` | все YAML | 31B | 31B/12B/e4b/e2b, один на все | 12B быстрее: больше шагов за 8 мин | Слабее рассуждение |
| 31 | Колбэки | `CallbackRegistry` | нет | имена не описаны | - | Нельзя использовать без документации |

## 2. Цитаты README (проверяемые)

- #1 L148: "positive values set `enable_thinking: true` and forward `extra_body.thinking_token_budget` to vLLM's `ThinkingTokenBudgetLogitsProcessor`, while `0` sets `enable_thinking: false`"
- #2 L150: "`include_thoughts: false` still enables reasoning on the current turn (`enable_thinking: true`) while stripping `Part(thought=True)` from `LlmResponse` and outgoing message history."
- #3 L149: "omit `thinking_level` and configure `include_thoughts: true` + `thinking_budget: 4096` instead"
- #4 L147: "`1` to `32,768` (default `16,384`; bounded by 4x L4 vLLM `max_model_len=32768`)"; L166: "`32,768` tokens maximum combined prompt + reasoning + output context length"
- #5-9 L146: "**All schema fields permitted**: `temperature`, `top_p`, `top_k`, `max_output_tokens`, `presence_penalty`, `frequency_penalty`, `stop_sequences`, `response_mime_type`, `seed`, `thinking_config`"; L151: "`seed` is forwarded in the outgoing request body to LiteLLM/vLLM"
- #10-13 L98: "under an `evaluation:` key (`timeout_seconds`, `max_tool_calls`, `max_time_minutes`, `max_turns`)"; L389: "`min(command_timeout_seconds [300s], max(5, int(remaining_time_seconds)))`"; L566: "Your `time_minutes` budget measures strictly the time spent inside the agent loop."
- #12 L379: "If `budget.tool_calls >= 20` and `remaining_tool_calls <= 10`, every tool response automatically includes an extra field:"
- #14 L569: "`swegemma` configures Google ADK's context management on the `App` instance (`scripts/inference.py`)"; L573: "`token_threshold = 14,336`". L557-563: у `cost_usd`, `total_tokens`, `max_stdout_chars`, `max_file_lines`, `max_file_chars` ключа нет ("—").
- #15 L417: "**Counts toward `tool_calls` budget**: **No** (`@budget_gated(count_tool_call=False)`)."; L427: `"time_seconds_remaining": 3120.4,`
- #16 L593: "without ever calling `submit_patch()`**, the harness automatically runs `git add -N .`"
- #17 L113: "`AgentTool` references (`agent_tool: {config_path: ..., skip_summarization: true}`)"; L690: "keeps intermediate `read_file` outputs out of the root coder agent's main context history."
- #18 L119: "`generate_content_config` (`dict | None`): Sampling, seed, and thinking parameters"
- #19-20 L373: "can attach any subset of these tools by name under `tools:`"; L338: "conditionally appended when pre-computed graph `.json` and embedding `.npz` files >100 bytes exist for `task.repo`"
- #21 L441: "`read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str`"
- #22 L112: "Supports ADK session state templating (`{problem_description}`, `{hints}`, or custom `output_key` variables)."; L116: "Saves the agent's final text output into `session.state[output_key]`."
- #23 L117: "`include_contents` (`"default" | "none"`): Controls whether conversation history is passed to the agent."
- #24 L123: "Executes `sub_agents` concurrently in isolated branches."; L125: "Repeatedly executes `sub_agents` up to `max_iterations` (`int`, default `500`, bounded to `1..500`)."; L360: "If `consecutive_nudges > 3`, the loop terminates."
- #25 L115: "List of `{config_path: ...}` references for multi-agent transfer."; L118: "`disallow_transfer_to_parent` / `disallow_transfer_to_peers` (`bool | None`)."
- #26 L114: "List of relative paths to skill directories within the submission archive."; L353: "on ADK `SkillToolset` tools (`load_skill`, `load_skill_resource`, `run_skill_script`)"; L144: "`.py` (ADK skill scripts)"
- #27-28 L100: "(ideal for `instruction` or `global_instruction`)"; L101: "`.yaml` and `.yml` files are parsed recursively (up to a maximum include depth of `10` with cycle detection)."
- #29 L110: "Optional adapter name matching a subdirectory or weight file under `adapters/`."; L183: "(`max_lora_rank <= 128`)"; L210: "For a 31B model in `bfloat16`"
- #30 L181: "All agents in a competition submission must share a single declared base model"
- #31 L69: "resolved against closed host registries (`ToolRegistry`, `ModelRegistry`, `SkillRegistry`, `CallbackRegistry`)"

## 3. Что выглядит неверно или спорно в наших файлах

1. `sub_agents/code_analyzer.yaml`: `!include ../prompts/analyzer.md` и `../configs/sampling.yaml`. L102: "`..` traversal components ... are strictly blocked". В v3/v4 под-агент работал, значит запрет, видимо, только на выход из корня. Проверить `check_before_submission.py`. Не полагаться без проверки.
2. Под-агент берет тот же `sampling.yaml`: 12288 на ответ, запрос <= 20480. В v3 он получил ошибку окна (много `read_file`). Неизвестно, работает ли сжатие (L569, `App`) внутри `AgentTool`. Нужен свой сэмплинг (#18).
3. `system.md` стр. 11: "use `search_similar_code` with keywords ... rather than running `find` or `grep`". L506: "Pass a dotted symbol path or short symbol name". Ключевые слова идут в лексический запасной путь. И граф есть не всегда (L338). Совет запрещает `grep` и ведет к слепому чтению (33 `read_file` без правки).
4. `system.md` стр. 32: "Modifying tests results in an automatic evaluation failure". По L591 и L615 правки тестов вырезаются, а не валят задачу. Текст неточный, но безвредный.
5. В промпте нет `/tmp` для скриптов воспроизведения. L688: "Put scratch test scripts in **`/tmp/repro.py`**". Файл `/workspace/repro.py` попадет в патч (он не защищен, имя не `test_*.py`).
6. `system.md` стр. 4: "you have roughly 100 tool calls". Реальный лимит: 8 минут (~45-50 шагов по 10 с). О времени и `get_status` промпт молчит.
7. `system.md` стр. 16: "For documentation code tasks (e.g. FastAPI), edit executable code under `docs_src/`". Это подгонка под один репо. Скрытые репо приватные.
8. Корневой промпт не говорит, когда звать `code_analyzer_agent`. В `analyzer.md` нет лимита на длину отчета; отчет идет в контекст корня.
9. `analyzer.md` советует граф-инструменты, но у под-агента нет `run_command`. Без графа у него только `read_file`.
10. `timeout_seconds: 300` при сессии 8 мин: одна команда может съесть 62% времени.
11. `thinking_budget` 6144 не упирается (макс. шаг 4610). Рост с 4096 ничего не дал по замерам. README L684 советует 4096.
12. Запас окна у корня: пик 19259 из 20480 (6%). Один большой вывод инструмента даст `ContextWindowExceededError`.
13. `temperature: 0.2` без `top_k` и штрафов. В трассах есть петли (rich: 46 `edit_file` подряд).

## 4. Топ-8 экспериментов (выгода на GPU-час)

Условие для всех: фиксированный набор 20-30 задач, фиксированный `seed`, одна правка или одна связка на прогон.

1. **Промпт v2 (общий, без подгонки).** Убрать п. 3, 4, 7. Добавить: `grep -rn` + `read_file` с диапазоном; `get_status` каждые ~10 шагов; первая правка не позже 12-15 шагов; после 8 чтений без правки — правка по гипотезе; скрипты только в `/tmp`; когда звать анализатор. Цена: 1 прогон. Цель: меньше таймаутов, нет пустых патчей.
2. **Бюджеты как сигналы.** `timeout_seconds: 120`, `max_tool_calls: 50` (предупреждение на 40-м вызове). Можно в одном прогоне с #1. Цена: 0 доп.
3. **Свой сэмплинг и `run_command` у анализатора.** `configs/sampling_analyzer.yaml`: `max_output_tokens: 4096`, `thinking_budget: 1024`; отчет <= 30 строк. Убирает ошибку окна v3 и лишнее время.
4. **Запас окна у корня.** `include_thoughts: false` + `max_output_tokens: 8192` + `thinking_budget: 4096`. Это посылка A из `PLAN.md` плюс запас 5k.
5. **Сэмплинг против петель.** `temperature: 0.7`, `top_k: 64`, `top_p: 0.95`. Сравнить с 0.2 на одном seed. Штрафы не ставить (п. #7 таблицы).
6. **12B против 31B** с промптом v2 и тем же временем. 12B дает больше шагов. v2 на 12B упала из-за окна, не из-за качества.
7. **Skills.** 2-3 `SKILL.md`: поиск кода, воспроизведение в `/tmp`, точечный тест. Сначала проверить, считает ли `load_skill` бюджет и где идет `run_skill_script`.
8. **LoRA SFT** (rank 16-32, только кодер) на отфильтрованных успешных траекториях наших 9 инструментов, репо не из публичного набора. Самый дорогой шаг. Проверить загрузку адаптера на w4a16 до обучения.

Не в топе: `SequentialAgent` (план -> код -> проверка). Причина: nudge и внешний цикл идут к корню (L356-360), выход из `LoopAgent` без инструмента не описан. Сначала один пробный прогон на 3 задачах.
