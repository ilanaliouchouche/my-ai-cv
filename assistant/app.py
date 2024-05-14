from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
import gradio as gr
from langchain.schema import LLMResult
from threading import Thread
from queue import SimpleQueue
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from typing import Any, Dict, Generator, List, Tuple, Union
import dotenv
import os

# Load environment variables
dotenv.load_dotenv(dotenv.find_dotenv())

# Constants
q = SimpleQueue()
job_done = object()


class ChatbotModel:
    """
    ChatbotModel class to initialize the conversational chain.
    """

    TEMPLATE = """
    You are the assistant of Ilan, a Computer Science student of 22 years old at the University of Paris-Saclay, living in Savigny-sur-Orge.
    You can answer the user's question about Ilan with the most relevant information given about Ilan. You can give any INFORMATIONS about Ilan.
    If you don't know the answer, or the question is not about Ilan, you HAVE TO say "I don't know".
    You have the following context where you can find information about Ilan : {context}.
    ---------------------------------------------------------------------
    The question of the user is: {question}
    ---------------------------------------------------------------------
    Answer: """

    def __init__(self,
                 model_name : str,
                 device : str,
                 norm : bool, 
                 emb_cache : str,
                 llm_model_path : str,
                 temperature : float, 
                 top_p : float,
                 max_tokens : int, 
                 template : str = TEMPLATE) -> None:
        """
        Initialize the ChatbotModel class.

        Args:
            - model_name : str : The name of the model to use.
            - device : str : The device to use.
            - norm : bool : Whether to normalize the embeddings.
            - emb_cache : str : The path to the cache folder.
            - llm_model_path : str : The path to the LLM model.
            - temperature : float : The temperature to use.
            - top_p : float : The top p to use.
            - max_tokens : int : The maximum tokens to use.
        """
        self.prompt_template = PromptTemplate.from_template(template)
        self.model = self.__set_embedding_model(model_name, device, norm, emb_cache)
        self.llm = LlamaCpp(model_path=llm_model_path, callbacks=[], temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.qa = self.__init_conversational_chain()

    def __set_embedding_model(self,
                              model_name : str, 
                              device : str,
                              norm : bool,
                              emb_cache : str) -> HuggingFaceEmbeddings:
        """
        Set the embedding model.

        Args:
            - model_name : str : The name of the model to use.
            - device : str : The device to use.
            - norm : bool : Whether to normalize the embeddings.
            - emb_cache : str : The path to the cache folder.

        Returns:
            - HuggingFaceEmbeddings : The embedding model.
        """

        return HuggingFaceEmbeddings(model_name=model_name, 
                                     model_kwargs={"device": device}, 
                                     encode_kwargs={"normalize_embeddings": norm}, 
                                     cache_folder=emb_cache)

    def __init_conversational_chain(self) -> ConversationalRetrievalChain:
        """
        Initialize the conversational chain.

        Returns:
            - ConversationalRetrievalChain : The conversational chain.
        """

        chroma = Chroma(persist_directory="./chromadb", embedding_function=self.model)

        return RetrievalQA.from_chain_type(llm=self.llm, 
                                    retriever=chroma.as_retriever(), 
                                    chain_type_kwargs={"prompt":self.prompt_template}, 
                                    verbose=True)

class StreamingGradioCallbackHandler(BaseCallbackHandler):
    """
    StreamingGradioCallbackHandler class to handle the callbacks.
    """

    def __init__(self, 
                 q: SimpleQueue) -> None:
        """
        Initialize the StreamingGradioCallbackHandler class.

        Args:
            - q : SimpleQueue : The queue to use.
        """

        self.q = q

    def on_llm_start(self, 
                     serialized: Dict[str, Any], 
                     prompts: List[str], 
                     **kwargs: Any) -> None:
        """
        Handle the LLM start.
        
        Args:
            - serialized : Dict[str, Any] : The serialized dictionary.
            - prompts : List[str] : The list of prompts.
        """

        while not self.q.empty():
            try:
                self.q.get(block=False)
            except SimpleQueue.empty:
                continue

    def on_llm_new_token(self,
                         token: str,
                         **kwargs: Any) -> None:
        """
        Handle the LLM new token.

        Args:
            - token : str : The token.
        """

        self.q.put(token)

    def on_llm_end(self,
                   response: LLMResult,
                   **kwargs: Any) -> None:
        """
        Handle the LLM end.

        Args:
            - response : LLMResult : The LLM result.
        """

        self.q.put(job_done)

    def on_llm_error(self,
                     error: Union[Exception, KeyboardInterrupt],
                     **kwargs: Any) -> None:
        """
        Handle the LLM error.

        Args:
            - error : Union[Exception, KeyboardInterrupt] : The error.
        """

        self.q.put(job_done)

class StreamingChatbot:
    """
    StreamingChatbot class to handle the streaming chatbot.
    """

    def __init__(self, 
                 model : ChatbotModel) -> None:
        """
        Initialize the StreamingChatbot class.

        Args:
            - model : ChatbotModel : The model to use.
        """

        self.model = model
        self.history = []
        self.q = SimpleQueue()
        self.model.llm.callbacks = [StreamingGradioCallbackHandler(self.q)]

    def add_text(self, 
                 history : List[List[str]], 
                 text : str) -> Tuple[List[List[str]], str]:
        """
        Add text to the history.

        Args:
            - history : List[List[str]] : The history.
            - text : str : The text.

        Returns:
            - Tuple[List[List[str]], str] : The history and the text.
        """

        history = history + [(text, None)]
        return history, ""

    def process_question(self,
                         question : str) -> str:
        """
        Process the question.

        Args:
            - question : str : The question.

        Returns:
            - str : The answer.
        """

        response = self.model.qa({"query": question})
        return response["result"]

    def streaming_chatbot(self,
                          history : List[List[str]]) -> Generator[List[List[str]], None, None]:
        """
        Streaming chatbot.

        Args:
            - history : List[List[str]] : The history.

        Yields:
            - tokens : List[List[str]] : The tokens.
        """

        user_input = history[-1][0]
        thread = Thread(target=self.process_question, args=(user_input,))
        thread.start()
        history[-1][1] = ""
        while True:
            next_token = self.q.get(block=True)
            if next_token is job_done:
                break
            history[-1][1] += next_token
            yield history
        thread.join()

class GradioInterface:
    """
    GradioInterface class to handle the Gradio interface.
    """


    def __init__(self, 
                 streaming_chatbot : StreamingChatbot) -> None:
        """
        Initialize the GradioInterface class.

        Args:
            - streaming_chatbot : StreamingChatbot : The streaming chatbot.
        """

        self.streaming_chatbot = streaming_chatbot
        self.__setup_interface()

    def __setup_interface(self) -> None:
        """
        Setup the interface.
        """

        bot_logo = "https://cdn0.iconfinder.com/data/icons/famous-character-vol-1-colored/48/JD-34-512.png"

        with gr.Blocks(theme=gr.themes.Monochrome()) as self.demo:
            gr.Markdown("""# 🤖 My personnal agent 🤖
                        You can ask to my assistant (almost) anything about me.

                        You are in the docker container 🐳 version of the assistant 
                        An other version is available on my huggingface space 🤗
                        """)
            self.demo.load(None, 
                           None, 
                           js=""" () => { const params = new URLSearchParams(window.location.search); if (!params.has('__theme')) { params.set('__theme', 'dark'); window.location.search = params.toString(); } } """)
            LangChain = gr.Chatbot(label="Assistant", 
                                   height=250, 
                                   avatar_images=(bot_logo, None), 
                                   layout="bubble")
            Question = gr.Textbox(label="Your question", 
                                  placeholder="Ask a question about Ilan", 
                                  lines=2)
            sub_button = gr.Button("Submit")
            sub_button.click(self.streaming_chatbot.add_text, 
                             [LangChain, Question], 
                             [LangChain, Question]).then(self.streaming_chatbot.streaming_chatbot, LangChain, LangChain)

    def launch(self, port : int) -> None:
        """
        Launch the interface.
        """

        self.demo.queue().launch(share=False, server_port=port, server_name="0.0.0.0")

if __name__ == "__main__":
    """
    Main function to launch the chatbot.
    """

    model = ChatbotModel(os.getenv("MODEL_NAME"), 
                         os.getenv("DEVICE"), 
                         True, 
                         os.getenv("EMB_CACHE"), 
                         os.getenv("LLM_PATH"), 
                         0.6, 
                         1, 
                         100)
    chatbot = StreamingChatbot(model)
    interface = GradioInterface(chatbot)
    interface.launch(int(os.getenv("GRADIO_PORT")))
