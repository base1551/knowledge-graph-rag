from graph_rag_agent import GraphRAGAgent
from utils import create_milvus_collection, init_neo4j_schema
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

def insert_sample_data(agent, collection_name: str):
    """サンプルデータの登録"""
    # Milvusにサンプルデータを登録
    collection = create_milvus_collection(collection_name)
    if not collection:
        return False

    # サンプルデータ
    sample_texts = [
        "GraphRAGは、グラフデータベースとベクトルデータベースを組み合わせた新しいRAGアプローチです。",
        "Neo4jは、エンティティ間の関係性を効率的に表現できるグラフデータベースです。",
        "Milvusは、大規模なベクトルデータの類似度検索に特化したベクターデータベースです。"
    ]

    # Embeddingsの作成
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # データの準備
    vectors = embeddings.embed_documents(sample_texts)
    entities = [
        {
            "id": i,
            "text": text,
            "embedding": vector
        }
        for i, (text, vector) in enumerate(zip(sample_texts, vectors))
    ]

    # Milvusへの登録
    collection.insert(entities)
    collection.flush()
    print("✅ Milvusにサンプルデータを登録しました")

    # Neo4jにサンプルデータを登録
    with agent.neo4j_driver.session() as session:
        # ノードの作成
        session.run("""
            CREATE (g:Topic {name: 'GraphRAG'})
            CREATE (n:Topic {name: 'Neo4j'})
            CREATE (m:Topic {name: 'Milvus'})
            CREATE (v:Topic {name: 'Vector DB'})
            CREATE (d:Topic {name: 'Graph DB'})

            CREATE (g)-[:USES]->(n)
            CREATE (g)-[:USES]->(m)
            CREATE (n)-[:IS_A]->(d)
            CREATE (m)-[:IS_A]->(v)
        """)
    print("✅ Neo4jにサンプルデータを登録しました")
    return True

def main():
    try:
        # GraphRAGAgentのインスタンス化
        agent = GraphRAGAgent()
        print("✅ GraphRAGAgentを初期化しました")

        # Neo4jスキーマの初期化
        init_neo4j_schema(agent.neo4j_driver)

        # サンプルデータの登録
        collection_name = "sample_collection"
        if not insert_sample_data(agent, collection_name):
            raise Exception("サンプルデータの登録に失敗しました")

        # テストクエリの実行
        test_queries = [
            "GraphRAGとは何ですか？",
            "Neo4jとMilvusの違いは何ですか？",
            "GraphRAGではどのようなデータベースが使用されていますか？"
        ]

        print("\n=== テストクエリの実行 ===")
        for query in test_queries:
            print(f"\n質問: {query}")
            result = agent.query(query, collection_name)
            print(f"回答: {result['answer']}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ エラーが発生しました: {str(e)}")
    finally:
        # リソースのクリーンアップ
        if 'agent' in locals():
            agent.close()

if __name__ == "__main__":
    main()
