import os
from dotenv import load_dotenv
from pymilvus import connections
from neo4j import GraphDatabase

# 環境変数の読み込み
load_dotenv()

def connect_milvus():
    """Milvusへの接続を確立する"""
    try:
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530")
        )
        print("✅ Milvusに接続しました")
        return True
    except Exception as e:
        print(f"❌ Milvusへの接続に失敗しました: {str(e)}")
        return False

def connect_neo4j():
    """Neo4jへの接続を確立する"""
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password123")
            )
        )
        # 接続テスト
        with driver.session() as session:
            session.run("RETURN 1")
        print("✅ Neo4jに接続しました")
        return driver
    except Exception as e:
        print(f"❌ Neo4jへの接続に失敗しました: {str(e)}")
        return None

def create_milvus_collection(collection_name, dim=1536):
    """Milvusにコレクションを作成する"""
    from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

    # スキーマの定義
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields=fields)

    # コレクションの作成
    try:
        collection = Collection(name=collection_name, schema=schema)
        # インデックスの作成
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"✅ Milvusコレクション '{collection_name}' を作成しました")
        return collection
    except Exception as e:
        print(f"❌ Milvusコレクションの作成に失敗しました: {str(e)}")
        return None

def init_neo4j_schema(driver):
    """Neo4jのスキーマを初期化する"""
    try:
        with driver.session() as session:
            # 制約の作成
            session.run("""
                CREATE CONSTRAINT paper_id IF NOT EXISTS
                FOR (p:Paper) REQUIRE p.id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT topic_name IF NOT EXISTS
                FOR (t:Topic) REQUIRE t.name IS UNIQUE
            """)
            print("✅ Neo4jスキーマを初期化しました")
            return True
    except Exception as e:
        print(f"❌ Neo4jスキーマの初期化に失敗しました: {str(e)}")
        return False
