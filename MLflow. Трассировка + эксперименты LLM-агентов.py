import os
import mlflow
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Если код выполняется вне записной книжки Databricks (например, локально),
# раскомментируйте и установите следующие переменные среды, указывающие на вашу рабочую область Databricks:
# os.environ["DATABRICKS_HOST"] = "https://your-workspace.cloud.databricks.com"
# os.environ["DATABRICKS_TOKEN"] = "your-personal-access-token"

mlflow.langchain.autolog() # LangGraph использует autolog LangChain
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/langgraph-tracing-demo")

@tool
def get_weather(city: str):
    """Используйте это для получения информации о погоде."""
    return f"В {city} может быть облачно"

llm = ChatOpenAI(model="gpt-4o-mini")
graph = create_react_agent(llm, [get_weather])

# Внимание: В оригинальном коде отсутствует список сообщений для invoke.
# Добавляю пример списка сообщений, чтобы код был функциональным.
result = graph.invoke({"messages": [("user", "Какая погода в Нью-Йорке?")]})
print(result)

# Просмотр трассировки в пользовательском интерфейсе MLflow
