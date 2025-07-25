# --- Шаг 1. Определение функций-компонентов конвейера (Pipeline) ---

def preprocess(raw_data: dict) -> dict:
    """
    Имитирует предварительную обработку данных.
    Например: очистка, нормализация, удаление пропусков.
    """
    print("-> Шаг 1. Выполняется предварительная обработка данных...")
    if not raw_data or "raw_text" not in raw_data:
        # Имитируем обработку пустого ввода
        return {"processed_text": ""}
        
    # Простое преобразование: приводим текст к нижнему регистру
    processed_data = raw_data.copy()
    processed_data["processed_text"] = raw_data["raw_text"].lower()
    print(f"   Результат: '{processed_data['processed_text']}'")
    return processed_data

def run_model(processed_data: dict) -> dict:
    """
    Имитируем запуск основной модели (например, нейронной сети).
    """
    print("-> Шаг 2. Запуск основной модели анализа...")
    
    text_to_analyze = processed_data.get("processed_text", "")
    
    # Ну просто имитируем логику модели, подсчитываем количество слов
    word_count = len(text_to_analyze.split())
    
    results = {
        "source_text": text_to_analyze,
        "word_count": word_count,
        "confidence": 0.95 # Имитация уверенности модели
    }
    print(f"   Результат: {results}")
    return results

def postprocess(model_results: dict) -> str:
    """
    Имитирует финальную обработку результатов.
    Например: форматирование вывода, приведение к нужному формату.
    """
    print("-> Шаг 3. Выполняется финальная обработка результатов...")
    
    # Создаем человекочитаемый отчет
    final_report = (
        f"Отчет по анализу текста:\n"
        f" - Исходный текст (обработанный): '{model_results.get('source_text', 'N/A')}'\n"
        f" - Количество слов: {model_results.get('word_count', 0)}\n"
        f" - Уверенность модели: {model_results.get('confidence', 0.0):.0%}"
    )
    print("   Результат: Отчет сформирован.")
    return final_report

# --- Шаг 2. Тут у нас определение основной функции-оркестратора (Workflow) ---

def complex_workflow(initial_data: dict) -> str:
    """
    Координирует все шаги обработки в правильной последовательности.
    """
    # Вызов шагов конвейера один за другим
    processed_data = preprocess(initial_data)
    model_results = run_model(processed_data)
    final_results = postprocess(model_results)
    
    return final_results

# --- Шаг 3. А здесь уже пример использования с реальными данными (точка входа) ---

if __name__ == "__main__":
    # Здесь у нас блок, который выполняется при запуске скрипта.
    # Если просто - определяем что на входе и запускаем весь процесс.
    
    print("--- ЗАПУСК КОМПЛЕКСНОГО ПРОЦЕССА ОБРАБОТКИ ---\n")

    # Определяем исходные данные для обработки
    sample_data = {"raw_text": "Это Пример Входного Текста для Демонстрации."}
    
    # Запускаем полный рабочий процесс с нашими данными
    final_output = complex_workflow(sample_data)
    
    print("\n--- ПРОЦЕСС ЗАВЕРШЕН ---\n")
    
    # Печатаем финальный результат, который был возвращен процессом
    print("Итоговый результат:")
    print(final_output)
