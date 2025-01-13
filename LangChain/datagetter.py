def data_get():
    documents_url = ["https://cosmosdbdatagetter.azurewebsites.net/data?data_range=2024-9-28 to 2025-1-1",]


    loader = langchain_community.document_loaders.SeleniumURLLoader(urls=documents_url)  # 修正
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

