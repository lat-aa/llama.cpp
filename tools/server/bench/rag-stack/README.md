# RAG-stack benches

Single harness for embed / looped-gen / e2e gates. See [BASELINE.md](BASELINE.md).

```bash
python rag_stack_bench.py embed      --url http://127.0.0.1:8080 --out embed.json
python rag_stack_bench.py loop-depth -m /path/model.gguf --require --expect 2
python rag_stack_bench.py gen-perf   --bin ../../../../build/bin/Release/llama-bench.exe -m /path/nb.gguf
python rag_stack_bench.py e2e        --embed-url http://127.0.0.1:8080 --gen-url http://127.0.0.1:8081
python rag_stack_bench.py sparse     --hf BAAI/bge-m3
```
