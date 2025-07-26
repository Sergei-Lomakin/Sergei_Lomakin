from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
import logging
import re

logger = logging.getLogger("AgenticAIPipeline")

class PlannerAgent:
    def init(self):
        self.llm = OllamaLLM(
            model="qwen3:4b",
            system="Ты - научный эксперт и планировщик. Твоя задача - разбить тему на наиболее информативные и важные подпункты для доклада. Отвечай на русском."
        )

    def run(self, topic: str) -> dict:
        """
        Разбивает заданную тему на подпункты.
        :param topic: Основная тема для разбиения.
        :return: Словарь с ключом 'subtopics', содержащим список сгенерированных подпунктов.
        """
        prompt = (
            f"Разбей тему '{topic}' на 4-7 информативных и важных подпунктов (аспектов) для подробного доклада. "
            "Каждый подпункт должен быть самостоятельной подтемой (например: определение, области применения, примеры, риски, перспективы и т.п.). "
            "Выведи подпункты в виде простого нумерованного списка, без пояснений и примечаний. Пиши только на русском!"
        )
        msg = HumanMessage(content=prompt)

        try:
            raw_output = self.llm.invoke([msg])
            # Извлекаем содержимое ответа, так как invoke может возвращать Message object
            out = raw_output if isinstance(raw_output, str) else getattr(raw_output, "content", str(raw_output))
        except Exception as e:
            logger.error(f"PlannerAgent: Ошибка при вызове LLM для генерации подпунктов: {e}")
            out = "" # В случае ошибки возвращаем пустую строку

        subtopics = []
        for line in out.splitlines():
            m = re.match(r"^\d+\s*[.)-]?\s*(.+)", line.strip()) # Улучшенный regex для парсинга нумерованного списка
            if m:
                subtopics.append(m.group(1).strip())

        # Если LLM не вернул нумерованный список, но вернул какой-то текст,
        # пробуем использовать каждую непустую строку как подпункт (без нумерации)
        if not subtopics and out.strip():
            subtopics = [s.strip() for s in out.splitlines() if s.strip()]

        logger.info(f"LLM выделил следующие подпункты для раскрытия темы '{topic}':\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(subtopics)))

        # Возвращаем словарь состояния, включающий исходную тему и сгенерированные подпункты
        return {"topic": topic, "subtopics": subtopics}
