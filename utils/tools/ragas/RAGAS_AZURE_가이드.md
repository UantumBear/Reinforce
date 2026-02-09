```python
# 출처: 26.02.09 기준 
# https://docs.ragas.io/en/v0.1.21/howtos/customisations/azure-openai.html

from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from ragas import evaluate

azure_model = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["model_deployment"],
    model=azure_configs["model_name"],
    validate_base_url=False,
)

# init the embeddings for answer_relevancy, answer_correctness and answer_similarity
azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["embedding_deployment"],
    model=azure_configs["embedding_name"],
)

```

# 이슈

직접 사용해보니, 가이드대로 호출함에도, 아래와 같은 오류가 났다.  (ragas==0.3.1)
```bash
BadRequestError(Error code: 400 - {'error': {'message': "Unsupported value: 'temperature' does not support 1E-8 with this model. Only the default (1) value is supported.", 'type': 'invalid_request_error', 'param': 'temperature', 'code': 'unsupported_value'}})
```

찾아보면 많이들 겪는 오류 같다. 혹시 해결이 되었을까? 라이브러리를 업그레이드 해보았다.  
이미 최신이었다 ㅜ  래퍼 클래스로 강제 주입하나 보통?  

```bash
uv lock --upgrade-package ragas
uv sync
```
