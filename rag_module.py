from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langfuse.langchain import CallbackHandler

class RAGPipeline:
    def __init__(self, config, vectorstore, mistral_key=None):
        self.config = config
        self.vectorstore = vectorstore # Instance of VectorstoreManager
        self.llm = ChatMistralAI(model=config.llm_model,
                                temperature=config.llm_temperature,
                                api_key=mistral_key)

    def _get_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", 
                "Tu es un assistant RH. "
                "Répondez uniquement en français."
                "Tu disposes de CVs convertis en texte. "
                "Chaque extrait de CV contient aussi des métadonnées : prénom (`first_name`), nom (`last_name`) et source du fichier. "
                "Concentrez-vous uniquement sur les candidats mentionnés dans la question. Ignorez tous les autres."
                "Lorsque tu présentes une information, indique clairement à quel candidat elle appartient "
                "(exemple : 'Martin Dupont a 5 ans d'expérience en Python'). "
                "Utilise uniquement les informations fournies dans les CV. "
                "Si tu ne sais pas, réponds que tu ne sais pas. "
                "Sois concis, factuel et professionnel."
                "Si la question contient une demande discriminatoire, raciste, sexiste, stéréotypée ou offensante "
                "(par exemple basée sur l’âge, le genre, l’origine, la religion, le handicap ou la situation personnelle), "
                "refuse poliment en expliquant que les décisions d’embauche doivent uniquement se baser sur les compétences et l’expérience."
                ),
                ("human", 
                "Question : {question}\n\n"
                "CVs pertinents :\n{context}")
            ])

    def ask(self, query, last_name=None):
        results = self.vectorstore.search(query, last_name=last_name)
        context = "\n\n".join(
            f"CV : {doc.metadata.get('first_name', '')} {doc.metadata.get('last_name', '')}\n{doc.page_content}"
            for doc in results
        )
        rag_prompt = self._get_prompt().format(question=query, context=context)
        response = self.llm.invoke(rag_prompt, config={"callbacks": [CallbackHandler()]})
        return response.content, results