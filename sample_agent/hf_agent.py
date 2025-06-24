import os, asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.genai import types
import litellm






import os
from huggingface_hub import InferenceClient


client = InferenceClient(
   provider="auto",
   #api_key=os.getenv("HF_TOKEN"),     #"here should have a hf key",
)


completion = client.chat.completions.create(
   model="deepseek-ai/DeepSeek-V3-0324",
   messages=[
       {
           "role": "user",
           "content": "How many 'G's in 'huggingface'?"
       }
   ],
)


print(completion.choices[0].message)




"""


llm = LiteLlm(model="huggingface/together/deepseek-ai/DeepSeek-R1-0528",
             api_key="hf_key")




agent = LlmAgent(
   model=llm,
   name="hf_agent",
   description="Agent that uses a Hugging Face model",
   instruction="Answer user queries using google_search tool",
   tools=[],
)


async def main():
   session_service = InMemorySessionService()
   session = await session_service.create_session(
       app_name="hf_app", user_id="u", session_id="s"
   )


   runner = Runner(agent=agent, app_name="hf_app", session_service=session_service)


   prompt = f"what time is it?"


   content = types.Content(role="user", parts=[types.Part(text=prompt)])


   async for event in runner.run_async(session_id=session.id, user_id="u", new_message=content):
       if event.is_final_response():
           print("Agent replied:", event.content.parts[0].text)


asyncio.run(main())


"""
