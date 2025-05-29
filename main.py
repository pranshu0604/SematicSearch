# Commented out code is for older reference and can be uncommented to run the respective sections.

# loading doc ------------------------------------------------


# from langchain_community.document_loaders import PyPDFLoader


# file_path = "./pdfs/fightclubsummary.pdf"
# loader = PyPDFLoader(file_path)

# docs = loader.load()

# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

# print(len(docs))

# splitting doc ------------------------------------------------

# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200, add_start_index=True
# )
# all_splits = text_splitter.split_documents(docs)

# print(len(all_splits))

# embedding doc ------------------------------------------------

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# vector_1 = embeddings.embed_query(all_splits[0].page_content)
# vector_2 = embeddings.embed_query(all_splits[1].page_content)

# assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(f"first ten elements:\n {vector_1[:10]}")


# creating vector store ---------------------------------------

from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# ids = vector_store.add_documents(documents=all_splits)

# results = vector_store.similarity_search(
#     "What does tyler durden say about the nature of one's being?",
# )

# print(results[0])



# self written retriever --------------------------------------


# from typing import List

# from langchain_core.documents import Document
# from langchain_core.runnables import chain


# @chain
# def retriever(query: str) -> List[Document]:
#     return vector_store.similarity_search(query, k=1)


# output = retriever.batch(
#     [
#         "What is the conclusion of Fight Club?",
#         "How does Tyler die?",
#     ],
# )

# print(output)


# using langchain retriever -----------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

output = retriever.batch(
    [
        "What is the conclusion of Fight Club?",
        "How does Tyler die?",
    ],
)

print(output)