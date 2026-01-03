FROM cr.metax-tech.com/public-ai-release/maca/vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64


WORKDIR /app

ENV OMP_NUM_THREADS=4

COPY requirements.txt  .
COPY download_model.py .

ENV PATH="/opt/conda/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt

RUN python download_model.py \
        --model_name qinglandesu/GPUclass_qwen4 \
        --cache_dir /app \
        --revision master

EXPOSE 8000

COPY . .

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]