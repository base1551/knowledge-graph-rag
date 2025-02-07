# GraphRAG Agent

Neo4j（グラフデータベース）とMilvus（ベクトルデータベース）を組み合わせた、LLMを用いたGraphRAGエージェントの実装です。

## 環境構築

### 前提条件
- Docker と Docker Compose がインストールされていること
- Python 3.8以上がインストールされていること
- OpenAI APIキー（または互換のAPIキー）があること

### セットアップ手順

1. リポジトリをクローンまたはダウンロード
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Python仮想環境の作成と有効化
```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
```

3. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

4. 環境変数の設定
`.env`ファイルを編集し、必要な環境変数を設定してください：
```
OPENAI_API_KEY=your_api_key_here
```

5. Dockerコンテナの起動
```bash
docker-compose up -d
```

## 使用方法

1. サンプルプログラムの実行
```bash
python main.py
```

2. APIとしての使用
```python
from graph_rag_agent import GraphRAGAgent

# エージェントの初期化
agent = GraphRAGAgent()

# クエリの実行
result = agent.query(
    "GraphRAGとは何ですか？",
    collection_name="sample_collection"
)

print(result["answer"])

# 終了時にリソースをクリーンアップ
agent.close()
```

## プロジェクト構成

- `docker-compose.yml`: MilvusとNeo4jのDockerコンテナ設定
- `requirements.txt`: Pythonパッケージの依存関係
- `.env`: 環境変数設定ファイル
- `utils.py`: ユーティリティ関数（DB接続など）
- `graph_rag_agent.py`: GraphRAGエージェントの主要実装
- `main.py`: 使用例とテストコード

## 機能

- Milvusを使用したセマンティック検索
- Neo4jを使用したグラフベースの知識検索
- LangChainを使用したLLMとの統合
- ベクトル検索とグラフ検索の結果を組み合わせた回答生成

## 注意事項

- 本実装はローカル環境での実行を想定しています
- 実運用環境では適切なセキュリティ設定を行ってください
- 大規模なデータセットを扱う場合は、環境に応じたチューニングが必要です

## ライセンス

MITライセンス
