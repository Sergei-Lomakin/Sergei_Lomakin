from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger("AgenticAIPipeline")

class ReviewAgent:
    def init(self):
        self.llm = OllamaLLM(
            model="qwen3:4b",
            system="Ты - эксперт-оценщик качества summary. Оценивай строго, отвечай только 'ok' или 'more'."
        )

    def need_more_info(self, subtopic, summary) -> bool:
        prompt = (
            f"Вот краткое summary по теме '{subtopic}':\n\n"
            f"{summary}\n\n"
            "Оцени, достаточно ли это информативно для качественного доклада. "
            "Если информации мало, нет примеров, нет конкретики или фактов, то напиши 'Повторить сбор (more)'. "
            "Если summary достаточно информативно и покрывает все важные аспекты, напиши 'Достаточно (ok)'. "
            "Пиши только 'ok' или 'more'!"
        )
        logger.info(f" Анализ summary для подпункта: '{subtopic}'\nТекст summary:\n{summary}\n")
        msg = HumanMessage(content=prompt)
        
        try:
            # Для OllamaLLM invoke принимает список сообщений
            raw_output = self.llm.invoke([msg])
            # Ensure the output is a string for comparison, handling different types of responses
            out_content = raw_output if isinstance(raw_output, str) else getattr(raw_output, "content", str(raw_output))
        except Exception as e:
            logger.error(f"ReviewAgent: Ошибка при вызове LLM для оценки summary: {e}")
            out_content = "more" # Assume more info is needed in case of an error

        logger.info(f" Ответ LLM-оценщика для подпункта '{subtopic}': {out_content}")
        return 'more' in out_content.lower()
