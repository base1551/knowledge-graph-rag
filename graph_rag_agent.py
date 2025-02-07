from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain
from langchain.schema import Document
from pymilvus import Collection, connections
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from utils import connect_milvus, connect_neo4j

# 環境変数の読み込み
load_dotenv()

class GraphRAGAgent:
    def __init__(self):
        # LLMの初期化
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Embeddingsの初期化
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # データベース接続
        self._init_databases()

        # Neo4j QAチェーンの初期化
        self.graph_qa = GraphCypherQAChain.from_llm(
            cypher_llm=self.llm,
            qa_llm=self.llm,
            graph=self.neo4j_driver,
            verbose=True
        )

    def _init_databases(self):
        """データベース接続の初期化"""
        # Milvus接続
        if not connect_milvus():
            raise ConnectionError("Milvusへの接続に失敗しました")

        # Neo4j接続
        self.neo4j_driver = connect_neo4j()
        if not self.neo4j_driver:
            raise ConnectionError("Neo4jへの接続に失敗しました")

    def vector_search(self, query: str, collection_name: str, top_k: int = 3) -> List[Document]:
        """Milvusでベクトル検索を実行"""
        try:
            # クエリのembedding取得
            query_embedding = self.embeddings.embed_query(query)

            # コレクションの取得
            collection = Collection(collection_name)
            collection.load()

            # 検索の実行
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text"]
            )

            # 結果をDocumentに変換
            documents = []
            for hits in results:
                for hit in hits:
                    doc = Document(
                        page_content=hit.entity.get('text'),
                        metadata={"score": hit.score}
                    )
                    documents.append(doc)

            return documents
        except Exception as e:
            print(f"ベクトル検索中にエラーが発生しました: {str(e)}")
            return []

    def graph_search(self, query: str) -> str:
        """Neo4jでグラフ検索を実行"""
        try:
            result = self.graph_qa({"query": query})
            return result["result"]
        except Exception as e:
            print(f"グラフ検索中にエラーが発生しました: {str(e)}")
            return ""

    def query(self, user_query: str, collection_name: str) -> Dict:
        """統合検索を実行"""
        # ベクトル検索の実行
        vector_results = self.vector_search(user_query, collection_name)

        # グラフ検索の実行
        graph_result = self.graph_search(user_query)

        # コンテキストの結合
        context = "\n\n".join([
            "【ベクトル検索結果】",
            *[doc.page_content for doc in vector_results],
            "\n【グラフ検索結果】",
            graph_result
        ])

        # 最終的な回答の生成
        final_prompt = PromptTemplate(
            template="""
            以下のコンテキストと質問に基づいて、包括的な回答を生成してください。

            コンテキスト:
            {context}

            質問:
            {query}

            回答:""",
            input_variables=["context", "query"]
        )

        final_response = self.llm.predict(
            final_prompt.format(context=context, query=user_query)
        )

        return {
            "answer": final_response,
            "vector_results": [doc.page_content for doc in vector_results],
            "graph_result": graph_result
        }

    def close(self):
        """リソースのクリーンアップ"""
        try:
            connections.disconnect("default")  # Milvus接続を閉じる
            if self.neo4j_driver:
                self.neo4j_driver.close()     # Neo4j接続を閉じる
        except Exception as e:
            print(f"クリーンアップ中にエラーが発生しました: {str(e)}")
