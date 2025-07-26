from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger("AgenticAIPipeline")

class ReportAgent:
    def init(self, model_name: str = "qwen3:1.7b"):
        """
        Инициализирует ReportAgent с указанной моделью LLM.
        :param model_name: Название модели LLM для использования.
        """
        self.llm = OllamaLLM(
            model=model_name,
            system="Ты - эксперт по искусственному интеллекту, задача которого - писать связные и информативные доклады. Отвечай всегда только на русском языке. Строго используй только предоставленную информацию."
        )

    def run(self, state: dict) -> dict:
        """
        Генерирует финальный доклад на основе суммированных подпунктов.
        :param state: Словарь состояния, содержащий 'topic' (основную тему)
                      и 'subtopic_summaries' (список словарей с 'subtopic' и 'summary').
        :return: Обновленный словарь состояния с 'final_report'.
        """
        topic = state.get("topic", "неизвестной теме")
        subtopic_summaries = state.get("subtopic_summaries", [])

        if not subtopic_summaries:
            logger.warning(f"ReportAgent: Не найдено суммированных подпунктов для темы '{topic}'. Отчет будет пустым.")
            state["final_report"] = "Не удалось сгенерировать отчет из-за отсутствия данных."
            return state

        # Формируем блоки текста для каждого подпункта
        subtopic_blocks = []
        for item in subtopic_summaries:
            subtopic_name = item.get("subtopic", "Без названия")
            subtopic_summary = item.get("summary", "Нет данных.")
            subtopic_blocks.append(f"### {subtopic_name}\n{subtopic_summary}")

        # Объединяем блоки в один текст для LLM
        combined_summaries_text = "\n\n".join(subtopic_blocks)

        prompt = (
            f"Ты - эксперт по искусственному интеллекту. Напиши подробный и связный доклад на тему '{topic}' "
            f"на основе ТОЛЬКО СЛЕДУЮЩИХ SUMMARIES:\n\n"
            f"{combined_summaries_text}\n\n"
            "Доклад должен быть логически структурирован, состоять из 4-5 абзацев, связывая информацию между подпунктами, "
            "включая примеры, если они есть в предоставленных summary. "
            "В самом конце доклада, после основного текста, добавь раздел 'Оглавление' "
            "со списком всех рассмотренных подпунктов, как в начале документа. "
            "Не придумывай информацию, которой нет в summary. Пиши только на русском языке!"
        )

        msg = HumanMessage(content=prompt)
        try:
            raw_output = self.llm.invoke([msg])
            # Убедимся, что получаем строку
            final_report = raw_output if isinstance(raw_output, str) else getattr(raw_output, "content", str(raw_output))
            logger.info(f"ReportAgent: Финальный доклад по теме '{topic}' сгенерирован.")
            state["final_report"] = final_report
        except Exception as e:
            logger.error(f"ReportAgent: Ошибка при генерации доклада для темы '{topic}': {e}")
            state["final_report"] = f"Произошла ошибка при генерации отчета: {e}"

        return state
