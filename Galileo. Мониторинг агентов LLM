from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import logging
# Здесь важно, что ReviewAgent находится в том же или доступном модуле
from .review_agent import ReviewAgent
# Предполагается наличие этих функций, если они используются
# from utils.article_parser import parse_article
# from utils.text_cleaner import remove_think_blocks

logger = logging.getLogger("AgenticAIPipeline")

class SummaryAgent:
    def init(self, model_name: str = "qwen3:1.7b", search_tool=None):
        """
        Инициализирует SummaryAgent.
        :param model_name: Название модели LLM для использования.
        :param search_tool: Инструмент для поиска (по умолчанию DuckDuckGoSearchAPIWrapper).
        """
        self.llm = OllamaLLM(
            model=model_name,
            system="Ты - эксперт по искусственному интеллекту, задача которого - создавать точные и информативные summary на основе предоставленных текстов. Отвечай всегда только на русском языке, не выдумывай информацию."
        )
        self.search_tool = search_tool if search_tool else DuckDuckGoSearchAPIWrapper()
        self.reviewer = ReviewAgent() # Инициализация ReviewAgent

    def run(self, state: dict) -> dict:
        """
        Координирует сбор и суммирование информации по подпунктам,
        взаимодействуя с ReviewAgent для оценки.
        :param state: Словарь состояния, содержащий 'topic' (основную тему)
                      и 'subtopics' (список подпунктов).
        :return: Обновленный словарь состояния с 'subtopic_summaries'.
        """
        topic = state.get("topic")
        subtopics = state.get("subtopics", [])
        subtopic_summaries = []

        if not topic or not subtopics:
            logger.error("SummaryAgent: Отсутствует основная тема или подпункты в состоянии.")
            state["subtopic_summaries"] = []
            return state

        for subtopic in subtopics:
            logger.info(f"SummaryAgent: Обработка подпункта: '{subtopic}' для темы '{topic}'")
            max_attempts = 2  # Максимальное количество попыток собрать больше информации
            current_n_articles = 3 # Начальное количество статей для поиска

            for attempt in range(max_attempts):
                articles = []
                try:
                    search_query = f"{topic} {subtopic}"
                    logger.info(f"SummaryAgent: Поиск {current_n_articles} статей по запросу: '{search_query}'")
                    search_results = self.search_tool.results(search_query, current_n_articles)

                    for art_meta in search_results:
                        # В реальной реализации здесь должен быть парсинг веб-страницы.
                        # Для примера, используем заглушку или простую имитацию.
                        # text = parse_article(art_meta.get("link"), art_meta.get("title"))
                        # Если parse_article не определен, используем заглушку:
                        text = f"Текст статьи из URL: {art_meta.get('link', 'N/A')}\nЗаголовок: {art_meta.get('title', 'N/A')}\n...\n"
                        articles.append({"url": art_meta.get("link"), "title": art_meta.get("title"), "text": text})

                except Exception as e:
                    logger.error(f"SummaryAgent: Ошибка при поиске статей для подпункта '{subtopic}': {e}")
                    # Если поиск не удался, завершаем попытки для текущего подпункта
                    break

                if not articles:
                    logger.warning(f"SummaryAgent: Не удалось найти статьи для подпункта '{subtopic}' после {attempt+1} попыток. Пропускаем.")
                    break # Переходим к следующему подпункту, если статьи не найдены

                # Формируем текст для суммаризации
                collected_texts = "\n\n".join(
                    f"URL: {a.get('url', 'N/A')}\nЗаголовок: {a.get('title', 'N/A')}\nТекст: {a.get('text', 'N/A')}"
                    for a in articles
                )

                prompt = (
                    f"Ты - эксперт по искусственному интеллекту. Создай подробное summary (краткое изложение) "
                    f"по подпункту '{subtopic}' в контексте основной темы '{topic}'. "
                    f"Используй ТОЛЬКО факты из предоставленных текстов ниже, не добавляй ничего выдуманного. "
                    f"Summary должно быть на русском языке, объемом 1-2 абзаца, с примерами, если они явно присутствуют в текстах.\n\n"
                    f"Тексты для анализа:\n{collected_texts}"
                )
                msg = HumanMessage(content=prompt)
                try:
                    raw_summary_output = self.llm.invoke([msg])
                    summary = raw_summary_output if isinstance(raw_summary_output, str) else getattr(raw_summary_output, "content", str(raw_summary_output))
                    # summary = remove_think_blocks(summary) # Применить, если есть функция remove_think_blocks
                except Exception as e:
                    logger.error(f"SummaryAgent: Ошибка при генерации summary для подпункта '{subtopic}': {e}")
                    summary = "Не удалось сгенерировать summary из-за ошибки."

                # --- Вызов ReviewAgent для оценки summary ---
                if summary == "Не удалось сгенерировать summary из-за ошибки.":
                    need_more = True # Считаем, что нужно больше информации, если произошла ошибка
                else:
                    need_more = self.reviewer.need_more_info(subtopic, summary)

                if not need_more or attempt == max_attempts - 1:
                    # Если информации достаточно или это последняя попытка, сохраняем summary
                    subtopic_summaries.append({"subtopic": subtopic, "summary": summary})
                    logger.info(f"SummaryAgent: Summary для подпункта '{subtopic}' сгенерировано и принято (попытка {attempt+1}, статей: {current_n_articles}).")
                    break # Переходим к следующему подпункту
                else:
                    logger.info(f"SummaryAgent: Агент-оценщик считает информацию недостаточной для '{subtopic}' - повторяем сбор с большим количеством статей.")
                    current_n_articles += 5 # Увеличиваем количество статей для следующей попытки
                    if current_n_articles > 15: # Ограничиваем максимальное количество статей
                        current_n_articles = 15
                        logger.warning(f"SummaryAgent: Достигнут лимит статей для '{subtopic}', принимаем текущее summary.")
                        subtopic_summaries.append({"subtopic": subtopic, "summary": summary})
                        break

        state["subtopic_summaries"] = subtopic_summaries
        return state
