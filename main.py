from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_cohere import ChatCohere
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")
    information="""
    Margaret Colleen Hoover (née Fennell; born December 11, 1979) is an American author who primarily writes novels in the romance and young adult fiction genres.[3][1] She is best known for her 2016 novel It Ends with Us. Many of her works were self-published before they were picked up by a publishing house. As of October 2022, Hoover has sold approximately 20 million books.[4] She was named one of the 100 most influential people in the world by Time magazine in 2023.[5]
Early and personal life

Hoover was born on December 11, 1979,[6] in Sulphur Springs, Texas, to Vannoy Fite[1] and Eddie Fennell. She grew up in Saltillo, Texas,[7] and she graduated from Saltillo High School in 1998.[8] She married Heath Hoover in 2000,[9] and they have three sons.[10] She graduated from Texas A&M University–Commerce with a degree in social work.[11] She worked in various social work and teaching jobs before she started her career as an author.[12] 
    """
    summary_template=f"""
    given the information {information} about a person, I want you to create:
    1. a short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatCohere(temperature=0, model="command-a-03-2025")
    # llm = ChatOllama(temperature=0, model="gpt-oss:120b-cloud")
    chain = summary_prompt_template | llm
    response = chain.invoke(input = {"information" : information})
    print(response.content)

if __name__ == "__main__":
    main()
