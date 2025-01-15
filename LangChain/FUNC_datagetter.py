urls1 = ["https://cosmosdbdatagetter.azurewebsites.net/data",]
urls2 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=%E3%83%88%E3%83%A9%E3%83%95%E3%82%A3%E3%83%83%E3%82%AF%E3%82%BD%E3%83%BC%E3%82%B9%E9%96%A2%E9%80%A3%E6%83%85%E5%A0%B1",]
urls3 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=%E3%83%A6%E3%83%BC%E3%82%B6%E3%83%BC%E8%A1%8C%E5%8B%95%E9%96%A2%E9%80%A3%E6%83%85%E5%A0%B1",]
urls4 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=%22%E3%83%A6%E3%83%BC%E3%82%B6%E3%83%BC%E8%A1%8C%E5%8B%95%E9%96%A2%E9%80%A3%E6%83%85%E5%A0%B1%22",]
urls5 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=%E3%82%B5%E3%82%A4%E3%83%88%E5%86%85%E6%A4%9C%E7%B4%A2%E9%96%A2%E9%80%A3%E6%83%85%E5%A0%B1",]
urls6 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=%E3%83%87%E3%83%90%E3%82%A4%E3%82%B9%E3%81%8A%E3%82%88%E3%81%B3%E3%83%A6%E3%83%BC%E3%82%B6%E3%83%BC%E5%B1%9E%E6%80%A7%E9%96%A2%E9%80%A3%E6%83%85%E5%A0%B1",]
urls7 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=%E6%99%82%E9%96%93%E5%B8%AF%E9%96%A2%E9%80%A3%E6%83%85%E5%A0%B1",]

def data_get(urls):
    loader = langchain_community.document_loaders.SeleniumURLLoader(urls=urls)  # 修正
    documents = loader.load() 


    # 読込した内容を分割する
    text_splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
    )
    docs = text_splitter.split_documents(documents)

    # OpenAIEmbeddings の初期化
    embedding = OpenAIEmbeddings()

    def get_embedding(text, model):
        text = text.replace("\n", " ")
        res = openai.embeddings.create(input = [text], model=model).data[0].embedding
        return res
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding
    )
    return vectorstore

    """
    ###############
    # LLMに質問する #
    ###############

    # プロンプトを準備
    template = """
    #<bos><start_of_turn>system
    #次の文脈を使用して、最後の質問に答えてください。
    #{context}
    #<end_of_turn><start_of_turn>user
    #{query}
    #<end_of_turn><start_of_turn>model
    """
    prompt = langchain.prompts.PromptTemplate.from_template(template)

    # チェーンを準備
    chain = (
        prompt
        | llm
    )

    query = "記事の{activeUsers}を教えてください"

    # 検索する
    search = vectorstore.similarity_search(query=query, k=3)

    content = "\n".join([f"Content:\n{doc.page_content}" for doc in search])

    # 推論を実行
    answer = chain.invoke({'query': query, 'context': content})
    print(answer)
    """

