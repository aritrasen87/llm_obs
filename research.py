import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage


### Research Agent for Web Search
tavily_tool = TavilySearchResults(max_results=5)
def web_search(query: str) -> str:
    """Search the web for information."""
    docs = tavily_tool.invoke({"query": query})
    web_results = "\n".join([d["content"] for d in docs])
    return web_results



class AppAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        loader = PyPDFLoader('sample_doc.pdf')
        docs = loader.load()
        ###  BGE Embddings
        model_name = "BAAI/bge-base-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        ### Creating Retriever using Vector DB
        self.db = Chroma.from_documents(docs, embeddings)
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})
        self.graph = self.app()

    def app(self):
        """Create a research agent with web search and RAG capabilities."""
        research_agent = create_react_agent(
            model=self.llm,tools=[web_search],name="research_expert",
            prompt="You are a world class researcher with access to web search.")
        def rag_search(query:str):
            "Function to do RAG search"
            docs = self.retriever.invoke(
                    query,
                )
            return "\nRetrieved documents:\n" + "".join(
                [
                    f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                    for i, doc in enumerate(docs)
                ]
            )

        rag_agent = create_react_agent(
            model=self.llm,
            tools=[rag_search],
            name="rag_expert",
            prompt="You are a RAG tool with access to transformer applications on Deep Learning related tasks."
        )
        workflow = create_supervisor(
            agents=[research_agent, rag_agent],model=self.llm,
            prompt=(
                "You are a supervisor managing a web search expert and a RAG search expert. "
                "For current events and information, use research_agent."
                "For transformer related information , use rag_agent."
            ))

        # Compile and run
        app = workflow.compile()
        return app
    def invoke(self, user_input: str, config: dict = None):
        """
        Invokes the LangGraph agent with the given user input.

        Args:
            user_input (str): The user's message.
            config (dict, optional): Configuration for the graph (e.g., thread_id for memory). Defaults to None.


        Returns:
            dict: The final state of the agent after processing the input.
        """
        # Create an initial state with the user's message
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        return self.graph.invoke(initial_state, config=config)



