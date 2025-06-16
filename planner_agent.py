import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import load_tools
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

from langchain.agents.agent_types import AgentType

load_dotenv()

# 🔐 Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)



# 🛠️ Tools (e.g. calculator, etc.)
tools = load_tools(["llm-math"], llm=llm)

# ⚙️ Create planner and executor
planner = load_chat_planner(llm=llm)
executor = load_agent_executor(llm=llm, tools=tools)

# 🧠 Plan-and-execute agent
plan_and_execute = PlanAndExecute(planner=planner, executor=executor)
