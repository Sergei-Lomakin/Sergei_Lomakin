from patronus import traced

@traced()
def generate_response(prompt: str) -> str:
    # Ваш LLM вызов или логика обработки здесь
    return f"Ответ на: {prompt}"

# Вызов трассируемой функции
result = generate_response("Расскажи мне о машинном обучении")
print(result)
