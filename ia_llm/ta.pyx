#! /usr/bin/env python

#Pillar       ,Stack      ,"""Necessary Extension""",Purpose
#Semantic     ,FAISS + SQL,Relational Linking       ,Traditional RAG; links facts to sources.
#Episodic     ,FAISS + SQL,Temporal Indexing        ,"Chat History"
#Reflective   ,FAISS + SQL,Contradiction Detection  ,"Identity & Narrative"
#Executive    ,FAISS + SQL,Dependency Graph         ,"Tracks sub-tasks. RAG here helps find ""similar past solutions"" to current problems."
#Metacognitive,FAISS + SQL,                         ,"Logic Logs & Error Correction for reflection, self-correction, hyper params, meta prompting, generating test cases, fine-tuning."

# TODO tool discovery
# TODO auto-retrieval & prompt generation

# TODO temperature

from copy import deepcopy
from dataclasses import dataclass, field
import datetime
from enum        import Enum
import inspect
from inspect     import Parameter
import json
import logging
import math
import multiprocessing
import os
from pathlib     import Path
import random
import requests
import shutil
import subprocess
from subprocess  import Popen
import sys
from types       import *
from typing      import *

import faiss
from faiss       import IndexFlatL2, IndexIDMap
import huggingface_hub
from llama_cpp   import Llama, LlamaGrammar
from llama_cpp   import llama_backend_init
import llama_cpp
import numpy     as np
from numpy       import ndarray
import psutil
from pydantic    import BaseModel, ValidationError
from pydantic    import create_model
import sqlite3

def setup_logging(name: str, level: int = logging.INFO) -> None:
    assert not logging.getLogger().hasHandlers()
    logging.basicConfig(
        level=level,
        format=f'%(asctime)s [%(levelname)s] [{name.upper()}] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

##
#
##

class NumCtxCap(Enum):
    HIGH_PERFORMANCE  = 16384
    STANDARD          =  4096
    MINIMAL           =  2048
    EMERGENCY         =   512
    CAP_EMBEDDING     =  8192
    CAP_NON_EMBEDDING = 32768
    CAP_FALLBACK      =  4096

def get_system_num_ctx_capacity() -> int:
    available_gb = psutil.virtual_memory().available / (1024**3) # Available RAM in GB

    # We want to leave at least 2GB for the OS and other processes
    # A 4B Q6 model takes ~3.5GB.
    if available_gb   > 8:
        return NumCtxCap.HIGH_PERFORMANCE.value
    if available_gb > 4:
        return NumCtxCap.STANDARD.value
    if available_gb > 2:
        return NumCtxCap.MINIMAL.value
    return NumCtxCap.EMERGENCY.value

def get_model_num_ctx_train(model_path: Path, embedding:bool) -> int:
    try:
        temp_llm           :Llama = Llama(model_path=str(model_path), vocab_only=True, verbose=False) # Load ONLY metadata (lightning fast, uses almost no RAM)
        model_train_cap    :int   = temp_llm.context_params.n_ctx

        if model_train_cap > 0:
            return model_train_cap
        logging.warning('no n_ctx_train')
        if embedding:
            return NumCtxCap.CAP_EMBEDDING.value
        return NumCtxCap.CAP_NON_EMBEDDING.value
    except Exception as error:
        logging.error(error)
        return NumCtxCap.CAP_FALLBACK.value

def get_optimal_num_ctx(model_path: Path, embedding: bool) -> int:
    logging.info(f'get_optimal_num_ctx(model_path={model_path}, embedding={embedding})')
    # Heuristic: ~128MB per 1024 tokens for 4B models.
    system_cap             :int   = get_system_num_ctx_capacity()
    model_train_cap        :int   = get_model_num_ctx_train(model_path=model_path, embedding=embedding)
    
    logging.info(f'system_cap     : {system_cap}')
    logging.info(f'model_train_cap: {model_train_cap}')

    return min(system_cap, model_train_cap)

class GpuOffloadConfig(Enum): # TODO uniq
    OFFLOAD_ALL = -1
    CPU_ONLY    =  0

def get_gpu_offload_config() -> int:
    if llama_cpp.llama_supports_gpu_offload():
        return GpuOffloadConfig.OFFLOAD_ALL.value
    return GpuOffloadConfig.CPU_ONLY.value

def get_llama(
        model_path  :Path,
        lora_path:Path|None=None,
        embedding   :bool     =False,
        n_batch     :int      =512,
        verbose     :bool     =False,
)->Llama:
    logging.info(f'get_llama(model_path={model_path}, lora_path={lora_path}, embedding={embedding}, n_batch={n_batch}, verbose={verbose})')
    n_ctx       :int      = get_optimal_num_ctx(model_path, embedding=embedding)
    n_gpu_layers:int      = get_gpu_offload_config()
    logging.info(f'n_ctx       : {n_ctx}')
    logging.info(f'n_gpu_layers: {n_gpu_layers}')
    if lora_path and not lora_path.exists():
        logging.warning(f'lora path dne: {lora_path}')
        lora_path         = None
    assert not lora_path or lora_path.is_file()
    lora_str    :str|None = str(lora_path) if lora_path else None
    return Llama(
            embedding   =embedding,
            lora_path   =lora_str,
            model_path  =str(model_path),
            n_batch     =n_batch,
            n_ctx       =n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose     =verbose,)

def pull_model(repo_id:str, model_path:Path)->Path:
    logging.info(f'pull_model(repo_id={repo_id}, model_path={model_path})')
    assert not model_path.exists()
    local_dir   :Path          = model_path.parent
    local_dir.mkdir(parents=True, exist_ok=True)
    cache_path  :str           = huggingface_hub.hf_hub_download(
        repo_id  =repo_id,
        filename =model_path.name,
        local_dir=local_dir,)
    logging.info(f'cache_path: {cache_path}')
    assert model_path.is_file()
    return Path(cache_path)

def pull_model_if_necessary(repo_id:str, model_path:Path)->bool:
    logging.info(f'pull_model_if_necessary(repo_id={repo_id}, model_path={model_path})')
    if model_path.exists():
        assert model_path.is_file()
        return False
    assert not model_path.exists()
    cache_path  :Path          = pull_model(repo_id, model_path)
    assert model_path.is_file()
    assert cache_path.resolve() == model_path.resolve()
    return True

def pull_llama(repo_id:str, model_path:Path, lora_path:Path, embedding:bool=False, n_batch:int=512, verbose:bool=False)->Llama:
    pull_model_if_necessary(repo_id=repo_id, model_path=model_path)
    return get_llama(model_path=model_path, lora_path=lora_path, embedding=embedding, n_batch=n_batch, verbose=verbose)

##
#
##

class ContextStats(BaseModel):
    used     :int
    remaining:int
    n_ctx    :int
    ratio    :float

def get_context_stats(llm: Llama, prompt: str, encoding:str='utf-8') -> ContextStats:
    if not prompt:
        logging.warning(f'no prompt')
    prompt_tokens     = llm.tokenize(prompt.encode(encoding))
    n_past       :int = len(prompt_tokens)
    assert (not prompt) or n_past
    n_ctx        :int = llm.n_ctx()
    assert n_ctx >= n_past
    remaining    :int = n_ctx - n_past
    if not remaining:
        logging.warning(f'no remaining tokens')
    return ContextStats(
            used     =n_past,
            remaining=remaining,
            n_ctx    =n_ctx,
            ratio    =float(n_past) / n_ctx,
    )

class ContextFullError(ValueError):
    """Context window is full. Need to compress or summarize"""

def calculate_safe_max_tokens(llm: Llama, prompt: str, requested: int = 500, threshold:int=10, encoding:str='utf-8') -> int:
    logging.info(f'calculate_safe_max_tokens(requested={requested}, threshold={threshold}, encoding={encoding})')

    stats:ContextStats = get_context_stats(llm, prompt, encoding=encoding)
    logging.info(f'context stats: {stats}')
    
    if stats.remaining <= threshold:
        raise ContextFullError(str(threshold - stats.remaining))
    return min(requested, stats.remaining)

##
#
##

def faiss_create_embedding_db(dimension: int) -> IndexIDMap:
    logging.info(f'faiss_create_embedding_db(dimension={dimension})')
    base_index = IndexFlatL2(dimension)
    return IndexIDMap(base_index)

def faiss_dump_embedding_db(index: IndexIDMap, state: dict, path: Path)->None:
    logging.info(f'faiss_dump_embedding_db(path={path})')
    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))

def faiss_load_embedding_db(path: Path)->IndexIDMap:
    logging.info(f'faiss_load_embedding_db(path={path})')
    index = faiss.read_index(str(path))
    return index

def faiss_load_or_create_embedding_db(path:Path, dimension:int)->IndexIDMap:
    logging.info(f'faiss_load_or_create_embedding_db(path={path}, dimension={dimension})')
    if not path.exists():
        return faiss_create_embedding_db(dimension)
    assert path.is_file()
    index:IndexIDMap = faiss_load_embedding_db(path)
    #assert dimension == index.dimension # TODO
    return index

def faiss_create_embedding(index: IndexIDMap, vector: List[float], doc_id: int) -> int:
    logging.info(f'faiss_create_embedding(doc_id={doc_id})')
    v_array = np.array([vector]).astype('float32')
    ids_array = np.array([doc_id]).astype('int64')
    index.add_with_ids(v_array, ids_array)
    return doc_id

def faiss_retrieve_embedding(
    index       :IndexIDMap,
    query_vector:List[float],
    timestamps  :Dict[int, float] |None=None, # Map of doc_id -> unix_timestamp
    k           :int                   =5,
    decay_lambda:float                 =0.00,
    now         :datetime.datetime|None=None,
) -> Tuple[ndarray, ndarray]:
    logging.info(f'faiss_retrieve_embedding(timestamps={timestamps}, k={k}, decay={decay_lambda})')
    timestamps     :Dict[str,float]        = timestamps or {}

    # 1. Search a larger pool than k to allow for re-ranking
    # If we only search k, we might miss a slightly more distant but much newer memory
    distances, indices                     = index.search(np.array([query_vector]).astype('float32'), k * 3)

    now            :float                  = now or datetime.datetime.now().timestamp()
    adjusted_scores:List[Tuple[float,int]] = []

    for dist, doc_id in zip(distances[0], indices[0]):
        if doc_id == -1: continue

        # 2. Get the age of the memory (default to 'now' if unknown)
        msg_time   :float                  = timestamps.get(doc_id, now)
        age_hours  :float                  = (now - msg_time) / 3600

        # 3. Penalize distance: d_new = d * e^(lambda * age)
        # For L2, lower is better. As age increases, adjusted distance grows.
        penalty    :float                  = math.exp(decay_lambda * age_hours)
        adjusted_scores.append((dist * penalty, doc_id))

    # 4. Re-sort by adjusted score and return the top k
    adjusted_scores.sort(key=lambda x: x[0])
    final_scores   :List[float]            = [s[0] for s in adjusted_scores[:k]]
    final_indices  :List[int]              = [s[1] for s in adjusted_scores[:k]]

    # TODO log number of results
    return np.array(final_scores), np.array(final_indices)

def faiss_update_embedding(index: IndexIDMap, vector: List[float], doc_id: int) -> int:
    logging.info(f'faiss_update_embedding(doc_id={doc_id})')
    faiss_delete_embedding(index, doc_id)
    return faiss_create_embedding(index, vector, doc_id)

def faiss_delete_embedding(index: IndexIDMap, doc_id: int) -> int:
    logging.info(f'faiss_delete_embedding(doc_id={doc_id})')
    ids_to_remove = np.array([doc_id], dtype='int64')
    index.remove_ids(ids_to_remove)
    return doc_id

##
#
##

def sqlite_create_text_db(path:Path, alias:str='sqlite_create_text_db')->None:
    logging.info(f'{alias}(path={path})')
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

def sqlite_dump_text_db(path:Path)->None:
    logging.info(f'sqlite_dump_text_db(path={path})')
    pass # noop

def sqlite_load_text_db(path:Path)->None:
    sqlite_create_text_db(path, alias='sqlite_load_text_db') # compat

def sqlite_load_or_create_text_db(path: Path):
    sqlite_create_text_db(path, alias='sqlite_load_or_create_text_db') # compat

def sqlite_create_text(path: Path, content: str, metadata: Dict[str, Any] | None = None) -> int:
    """Inserts text, returns rowid (1-based)."""
    logging.info(f'sqlite_create_text(path={path})')
    with sqlite3.connect(path) as conn:
        cursor     = conn.cursor() # TODO typehint
        cursor.execute(
            "INSERT INTO documents (content, metadata) VALUES (?, ?)",
            (content, json.dumps(metadata or {}))
        )
        row_id:int = cursor.lastrowid
    logging.info('row_id: {row_id}')
    return row_id

def filter_faiss_padding(doc_ids:List[int])->List[int]:
    return [int(i) for i in doc_ids if i != -1]

def sqlite_retrieve_text(path: Path, doc_ids: List[int]) -> List[Dict[str, Any]]:
    logging.info(f'sqlite_retrieve_text(path={path}, doc_ids={doc_ids})')
    if not doc_ids:
        logging.warning('no doc ids')
        return []
    valid_ids = filter_faiss_padding(doc_ids)
    placeholders = ', '.join(['?'] * len(valid_ids))
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            f'SELECT id, content, metadata FROM documents WHERE id IN ({placeholders})',
            valid_ids
        )
        # TODO log number of results
        return [dict(row) for row in cursor.fetchall()]

def sqlite_update_text(path: Path, doc_id: int, content: str, metadata: Dict[str, Any] | None = None)->None:
    logging.info(f'sqlite_update_text(path={path}, doc_id={doc_id})')
    with sqlite3.connect(path) as conn:
        conn.execute(
            'UPDATE documents SET content = ?, metadata = ? WHERE id = ?',
            (content, json.dumps(metadata or {}), doc_id)
        )

def sqlite_delete_text(path: Path, doc_id: int)->None:
    logging.info(f'sqlite_delete_text(path={path}, doc_id={doc_id})')
    with sqlite3.connect(path) as conn:
        conn.execute('DELETE FROM documents WHERE id = ?', (doc_id,))

##
#
##

def db_create(dimension:int, sqlite_path:Path)->None:
    logging.info(f'db_create(dimension={dimension}, sqlite_path={sqlite_path})')
    faiss_create_embedding_db(dimension)
    sqlite_create_text_db    (sqlite_path)

def db_load_or_create(faiss_path:Path, dimension:int, sqlite_path:Path)->None:
    logging.info(f'db_load_or_create(faiss_path={faiss_path}, dimension={dimension}, sqlite_path={sqlite_path})')
    faiss_load_or_create_embedding_db(faiss_path, dimension)
    sqlite_load_or_create_text_db(sqlite_path)

def db_dump(faiss_path:Path, sqlite_path:Path)->None:
    logging.info(f'db_dump(faiss_path={faiss_path}, sqlite_path={sqlite_path})')
    faiss_dump_embedding_db(faiss_path)
    sqlite_dump_text_db    (sqlite_path)

def db_load(faiss_path:Path, sqlite_path:Path)->None:
    logging.info(f'db_load(faiss_path={faiss_path}, sqlite_path={sqlite_path})')
    faiss_load_embedding_db(faiss_path)
    sqlite_load_text_db    (sqlite_path)

def db_create_document(index: IndexIDMap, db_path: Path, vector: List[float], content: str, metadata: dict = None) -> int:
    logging.info(f'db_create_document(db_path={db_path})')
    doc_id = sqlite_create_text(db_path, content, metadata)
    faiss_create_embedding(index, vector, doc_id)
    return doc_id

def db_retrieve_document(
        index       : IndexIDMap,
        db_path     : Path,
        query_vector: List[float],
        timestamps  :Dict[int,float]  |None=None,
        k           : int                  = 5,
        decay_lambda:float                 =0.0,
        now         :datetime.datetime|None=None,
) -> List[Dict[str, Any]]:
    logging.info(f'db_retrieve_document(db_path={db_path}, k={k})')
    distances, indices = faiss_retrieve_embedding(index=index, query_vector=query_vector, timestamps=timestamps, k=k, decay_lambda=decay_lambda, now=now)
    return sqlite_retrieve_text(db_path, indices.tolist())

def db_update_document(index: IndexIDMap, db_path: Path, doc_id: int, vector: List[float], content: str, metadata: dict = None):
    logging.info(f'db_update_document(db_path={db_path}, doc_id={doc_id})')
    sqlite_update_text(db_path, doc_id, content, metadata)
    faiss_update_embedding(index, vector, doc_id)

def db_delete_document(index: IndexIDMap, db_path: Path, doc_id: int):
    logging.info(f'db_delete_document(db_path={db_path}, doc_id={doc_id})')
    faiss_delete_embedding(index, doc_id)
    sqlite_delete_text(db_path, doc_id)

##
#
##

def generate_embedding(text: str, llm: Llama) -> list[float]:
    logging.debug(f'generate_embedding()')
    return llm.embed(text)

def generate_search_embedding(text:str, llm: Llama, prefix:str='search_query: ')->list[float]:
    logging.info(f'generate_search_embedding(prefix={prefix})')
    text = f'{prefix}{text}'
    return generate_embedding(text, llm)

def generate_document_embedding(text:str, llm:Llama, prefix:str='search_document: ')->list[float]:
    logging.info(f'generate_document_embedding(prefix={prefix})')
    text = f'{prefix}{text}'
    return generate_embedding(text, llm)

def rag_create_document(content: str, embed_llm: Llama, index: IndexIDMap, db_path: Path, metadata: dict = None, prefix:str='search_document: ') -> int:
    logging.info(f'rag_create_document(db_path={db_path})')
    vector = generate_document_embedding(content, embed_llm, prefix=prefix)
    return db_create_document(index, db_path, vector, content, metadata)

def rag_retrieve_document(
        query       : str,
        embed_llm   : Llama,
        index       : IndexIDMap,
        db_path     : Path,
        prefix      :str                   ='search_query: ',
        timestamps  :Dict[int,float]  |None=None,
        k           :int                   = 5,
        decay_lambda:float                 =0.0,
        now         :datetime.datetime|None=None,
) -> List[Dict[str, Any]]:
    logging.info(f'rag_retrieve_document(db_path={db_path}, k={k})')
    query_vec = generate_search_embedding(query, embed_llm, prefix)
    return db_retrieve_document(index=index, db_path=db_path, query_vector=query_vec, timestamps=timestamps, k=k, decay_lambda=decay_lambda, now=now)

def rag_update_document(doc_id: int, content: str, embed_llm: Llama, index: IndexIDMap, db_path: Path, metadata: dict = None, prefix:str='search_document: ')->None:
    logging.info(f'rag_update_document(doc_id={doc_id}, db_path={db_path})')
    new_vector = generate_document_embedding(content, embed_llm, prefix=prefix)
    db_update_document(index, db_path, doc_id, new_vector, content, metadata)

def rag_delete_document(doc_id: int, index: IndexIDMap, db_path: Path)->None:
    logging.info(f'rag_delete_document(doc_id={doc_id}, db_path={db_path})')
    db_delete_document(index, db_path, doc_id)

# TODO data cleaning ?
# TODO file contents -type aware chunking
# TODO data cleaning ?

##
#
##

class Exchange():
    system_prompt:str
    user_msg     :str
    thought      :str|None
    assistant_msg:str

def format_exchange_for_training(
        exchange     :Exchange,
        #system_prompt:str,
        #user_msg     :str,
        #thought      :str|None,
        #assistant_msg:str,
        sample_start :str='<s>',
        im_start     :str='<|im_start|>',
        im_end       :str='<|im_end|>',
        think        :str='<|thought|>',
)->str:
    fmt_system   :str = '{im_start}system\n{system_prompt}{im_end}\n'
    fmt_user     :str = '{im_start}user\n{user_msg}{im_end}\n'
    fmt_think    :str = '{im_start}assistant\n{think}\n{thought}\n{think}\n{assistant_msg}{im_end}\n'
    fmt_no_think :str = '{im_start}assistant\n{assistant_msg}{im_end}\n'
    fmt_assistant:str = fmt_think if thought is not None else fmt_no_think
    msg_system   :str = fmt_system   .format(
            system_prompt=exchange.system_prompt,
            im_start     =im_start,
            im_end       =im_end)
    msg_user     :str = fmt_user     .format(
            user_msg     =exchange.user_msg,
            im_start     =im_start,
            im_end       =im_end)
    msg_assistant:str = fmt_assistant.format(
            assistant_msg=exchange.assistant_msg,
            think        =think,
            thought      =exchange.thought,
            im_start     =im_start,
            im_end       =im_end,)
    return (sample_start + msg_system + msg_user + msg_assistant + sample_start)

def clean_sample_for_training(sample:str)->str:
    fmt            :str = '{sample}\n'
    sample              = sample.replace('\r', '')
    sample              = sample.replace('\n', ' ')
    sample              = sample.strip()
    return fmt.format(sample=sample)

def get_bytes_token_len(prompt:bytes, llm:Llama)->int:
    assert isinstance(prompt,bytes)
    return llm.tokenize(prompt)

def get_str_token_len(prompt:str, llm:Llama, encoding:str='utf-8')->int:
    assert isinstance(prompt,str)
    b:bytes = prompt.encode(encoding)
    return get_bytes_token_len(prompt=b, llm=llm)

def get_sample_sort_key(llm:Llama, encoding:str='utf-8')->Callable[[str],int]:
    def wrapped(prompt:str)->int:
        return get_str_token_len(prompt=prompt, llm=llm, encoding=encoding)
    return wrapped # TODO maybe use joblib caching

def shuffle_sample_buckets_for_training(buckets:List[List[str]])->None:
    logging.info(f'shuffle_sample_buckets_for_training(n_buckets={len(buckets)})')
    for bucket in buckets:
        logging.debug(f'n_sample: {len(bucket)}')
        random.shuffle(bucket)

def get_sample_buckets_for_training(
    samples   :List[str],
    llm       :Llama,
    group_size:int= 64, # Alignment with hardware/batch multiples
    encoding  :str='utf-8',
) -> List[List[str]]:
    logging.info(f'format_exchanges_for_training(n_samples={len(samples)}, group_size={group_size})')
    samples                     = list(map(clean_sample_for_training, samples))
    # TODO drop samples that are bigger than llm's num ctx
    key    :Callable[[str],int] = get_sample_sort_key(llm=llm, encoding=encoding)
    samples.sort(key=key)
    buckets:List[List[str]]     = [
            samples[i:i + group_size]
            for i in range(0, len(samples), group_size)]
    shuffle_sample_buckets_for_training(buckets)
    return buckets

def format_exchanges_for_training(
    exchanges    :List[Exchange],
    llm          :Llama,
    sample_start :str='<s>',
    im_start     :str='<|im_start|>',
    im_end       :str='<|im_end|>',
    think        :str='<|thought|>',
    group_size   :int= 64, # Alignment with hardware/batch multiples
    encoding     :str='utf-8',
) -> List[List[str]]:
    logging.info(f'format_exchanges_for_training('
        f'n_exchanges={len(exchanges)}, '
        f'sample_start={sample_start}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'think={think}, '
        f'group_size={group_size}, '
        f'encoding={encoding})')
    samples:List[str] = [
            format_exchange_for_training(
                exchange    =exchange,
                sample_start=sample_start,
                im_start    =im_start,
                im_end      =im_end,
                think       =think,)
            for exchange in exchanges]
    return get_sample_buckets_for_training(
        samples   =samples,
        llm       =llm,
        group_size=group_size,
        encoding  =encoding,)

def dump_sample_buckets_for_training(training_data:Path, buckets:List[List[str]], encoding:str='utf-8', clobber:bool=False)->None:
    logging.info(f'dump_sample_buckets_for_training(training_data={training_data}, n_buckets={len(buckets)}, encoding={encoding}, clobber={clobber})')
    assert clobber or (not training_data.exists())
    with open(training_data, 'w', encoding=encoding) as f:
        for bucket in buckets:
            logging    .debug(f'n_sample  : {len(bucket)}')
            for sample in bucket:
                logging.debug(f'sample_len: {len(sample)}')
                f.write(sample)
    assert training_data.is_file()

def get_finetune_cmd(
    model_base  :Path, 
    train_data  :Path, 
    lora_out    :Path, 
    threads     :int|None=None,
    adam_iter   :int     =256,   # Standard for small LoRAs
    batch       :int     =4,     # Small batch size for CPU
    sample_start:str     ='<s>', # The "magic" delimiter # TODO <|endoftext|> ???
) -> List[str]:
    logging.info(f'get_finetune_cmd('
        f'model_base={model_base}, '
        f'train_data={train_data}, '
        f'lora_out={lora_out}, '
        f'threads={threads}, '
        f'adam_iter={adam_iter}, '
        f'batch={batch}, '
        f'sample_start={sample_start}')
    llama_finetune:str|None = shutil.which('llama-finetune')
    logging.info(f'llama finetune: {llama_finetune}')
    assert llama_finetune
    threads       :int      = threads or multiprocessing.cpu_count()
    checkpoint_in :Path     = lora_out.with_suffix('.chkpt') # For resuming
    return [
        llama-finetune,
        '--model-base',    str(model_base),
        '--train-data',    str(train_data),
        '--lora-out',      str(lora_out),
        '--threads',       str(threads),
        '--adam-iter',     str(adam_iter),
        '--batch',         str(batch),
        '--sample-start',  sample_start,
        '--checkpoint-in', str(checkpoint_in),
    ]

class ModelCollapseError(Exception):
    """Raised when loss becomes NaN or hits 0.0 too early."""

def graduated_termination(process:Popen, timeout:int=5)->None:
    logging.info(f'graduated_termination(timeout={timeout})')
    if not process:
        logging.warning('no process')
        return
    if process.poll() is not None:
        logging.info('already dead')
        return
    logging.warning('terminating')
    process.terminate()
    logging.info('waiting...')
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        logging.error(error)
        process.kill()

def watch_finetune_proc(process:Popen)->None:
    logging.info(f'watch_finetune_proc()')
    for line in process.stdout:
        line_clean:str        = line.strip()
        line_lower:str        = line_clean.lower()
        if ('loss' not in line_lower):
            continue
        logging.info(f'llama-finetune: {line_clean}')
        is_nan    :bool       = ('nan'      in line_lower)
        is_zero   :bool       = ('0.000000' in line_lower)
        if (not is_nan) and (not is_zero):
            continue
        raise ModelCollapseError('Loss is NaN. Check learning rate or data quality.')

def run_finetune_cmd(
    model_base  :Path,
    train_data  :Path,
    lora_out    :Path,
    threads     :int|None=None,
    adam_iter   :int     =256,
    batch       :int     =4,
    sample_start:str     ='<s>', # TODO <|endoftext|> ???
    timeout     :int     =5,
)->None:
    logging.info(f'run_finetune_cmd('
        f'model_base={model_base}, '
        f'train_data={train_data}, '
        f'lora_out={lora_out}, '
        f'threads={threads}, '
        f'adam_iter={adam_iter}, '
        f'batch={batch}, '
        f'sample_start={sample_start})')
    command           :List[str]  = get_finetune_cmd(
        model_base  =model_base,
        train_data  =train_data,
        lora_out    =lora_out,
        threads     =threads,
        adam_iter   =adam_iter,
        batch       =batch,
        sample_start=sample_start,)
    logging.info(f'command: {command}')
    process           :Popen|None = None
    try:
        process       :Popen      = Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,)
        watch_finetune_proc(process)
        process.wait()
    finally:
        graduated_termination(process, timeout=timeout)
    if (process.returncode != 0):
        raise subprocess.CalledProcessError(process.returncode, command)

def sqlite_create_exchange_db(path: Path)->None:
    logging.info(f'sqlite_create_exchange_db(path={path})')
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exchanges (
                id INTEGER PRIMARY KEY,
                adapter TEXT,
                system_prompt TEXT,
                user_msg TEXT,
                thought TEXT,
                assistant_msg TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

def sqlite_create_exchange(path:Path, adapter:str, exchange:Exchange, created_at:datetime.datetime)->None:
    logging.info(f'sqlite_create_exchange(path={path}, adapter={adapter}, created_at={created_at})')
    with sqlite3.connect(path) as conn:
        conn.execute("""
            INSERT INTO exchanges (adapter, system_prompt, user_msg, thought, assistant_msg, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (adapter, exchange.system_prompt, exchange.user_msg, exchange.thought, exchange.assistant_msg, created_at))

def sqlite_retrieve_exchanges(path: Path, adapter: str | None = None) -> List[Exchange]:
    logging.info(f'sqlite_retrieve_exchanges(path={path}, adapter={adapter})')
    query = 'SELECT system_prompt, user_msg, thought, assistant_msg FROM exchanges'
    params = []
    if adapter:
        query += ' WHERE adapter = ?'
        params.append(adapter)
    
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        # TODO log number of results
        return [
            Exchange(
                system_prompt=row['system_prompt'],
                user_msg=row['user_msg'],
                thought=row['thought'],
                assistant_msg=row['assistant_msg']
            ) for row in cursor.fetchall()
        ]

def sqlite_update_exchange()->...:
    logging.info(f'sqlite_update_exchange()')
    raise NotImplementedError() # TODO

def sqlite_delete_exchanges_by_adapter(path: Path, adapter: str) -> None:
    logging.info(f'sqlite_delete_exchanges_by_adapter(path={path}, adapter={adapter})')
    with sqlite3.connect(path) as conn:
        conn.execute('DELETE FROM exchanges WHERE adapter = ?', (adapter,))
    # TODO log result

def finetune_adapter(
    exchange_db  :Path,
    adapter      :str,
    # format_exchanges_for_training
    llm          :Llama,
    # dump_sample_buckets_for_training
    train_data   :Path,
    # run_finetune_cmd
    model_base   :Path,
    lora_out     :Path,
    # format_exchanges_for_training
    sample_start :str     ='<s>', # TODO <|endoftext|> ???
    im_start     :str     ='<|im_start|>',
    im_end       :str     ='<|im_end|>',
    think        :str     ='<|thought|>',
    group_size   :int     = 64, # Alignment with hardware/batch multiples
    encoding     :str     ='utf-8',
    # dump_sample_buckets_for_training
    clobber      :bool    =False,
    # run_finetune_cmd
    threads      :int|None=None,
    adam_iter    :int     =256,
    batch        :int     =4,
    timeout      :int     =5,
) -> None:
    """ High-level primitive to turn an exchange DB into a LoRA adapter.  """
    logging.info(f'finetune_adapter('
                 f'exchange_db={exchange_db} '
                 f'adapter={adapter} '
                 f'model_base={model_base} '
                 f'lora_out={lora_out} '
                 f'sample_start={sample_start} '
                 f'im_start={im_start} '
                 f'im_end={im_end} '
                 f'think={think} '
                 f'group_size={group_size} '
                 f'encoding={encoding} '
                 f'threads={threads} '
                 f'adam_iter={adam_iter} '
                 f'batch={batch} '
                 f'timeout={timeout})')
    if train_data.exists() and (not clobber):
        raise FileExistsError(f'Training file already exists at {train_data}. Use clobber=True to overwrite.')
    exchanges     :List[Exchange]  = sqlite_retrieve_exchanges(path=exchange_db, adapter=adapter)
    if not exchanges:
        raise ValueError(f"No exchanges found in {exchange_db} for adapter: '{adapter}'")
    buckets   :List[List[str]] = format_exchanges_for_training(
            exchanges   =exchanges,
            llm         =llm,
            sample_start=sample_start,
            im_start    =im_start,
            think       =think,
            group_size  =group_size,
            encoding    =encoding,)
    dump_sample_buckets_for_training(
            training_data=train_data,
            buckets      =buckets,
            encoding     =encoding,
            clobber      =clobber,)
    run_finetune_cmd(
            model_base  =model_base,
            train_data  =train_data,
            lora_out    =lora_out,
            threads     =threads,
            adam_iter   =adam_iter,
            batch       =batch,
            sample_start=sample_start,
            timeout     =timeout,)

def finetune_llm_adapter(
    exchange_db  :Path,
    adapter      :str,
    rag          :RAG,
    train_data   :Path,
    sample_start :str     ='<s>', # TODO <|endoftext|> ???
    im_start     :str     ='<|im_start|>',
    im_end       :str     ='<|im_end|>',
    think        :str     ='<|thought|>',
    group_size   :int     = 64, # Alignment with hardware/batch multiples
    encoding     :str     ='utf-8',
    clobber      :bool    =False,
    threads      :int|None=None,
    adam_iter    :int     =256,
    batch        :int     =4,
    timeout      :int     =5,
)->None:
    finetune_adapter(
            exchange_db =exchange_db,
            adapter     =adapter,
            llm         =rag.llm,
            train_data  =train_data,
            model_base  =rag.llm_path,
            lora_out    =rag.llm_lora,
            sample_start=sample_start,
            im_start    =im_start,
            im_end      =im_end,
            think       =think,
            group_size  =group_size,
            encoding    =encoding,
            clobber     =clobber,
            threads     =threads,
            adam_iter   =adam_iter,
            batch       =batch,
            timeout     =timeout,)

def finetune_embed_adapter(
    exchange_db  :Path,
    adapter      :str,
    rag          :RAG,
    train_data   :Path,
    sample_start :str     ='<s>', # TODO <|endoftext|> ???
    im_start     :str     ='<|im_start|>',
    im_end       :str     ='<|im_end|>',
    think        :str     ='<|thought|>',
    group_size   :int     = 64, # Alignment with hardware/batch multiples
    encoding     :str     ='utf-8',
    clobber      :bool    =False,
    threads      :int|None=None,
    adam_iter    :int     =256,
    batch        :int     =4,
    timeout      :int     =5,
)->None:
    finetune_adapter(
            exchange_db =exchange_db,
            adapter     =adapter,
            llm         =rag.embed,
            train_data  =train_data,
            model_base  =rag.embed_path,
            lora_out    =rag.embed_lora,
            sample_start=sample_start,
            im_start    =im_start,
            im_end      =im_end,
            think       =think,
            group_size  =group_size,
            encoding    =encoding,
            clobber     =clobber,
            threads     =threads,
            adam_iter   =adam_iter,
            batch       =batch,
            timeout     =timeout,)

def finetune_semantic_llm_adapter(
    exchange_db  :Path,
    rag          :RAG,
    train_data   :Path,
    sample_start :str     ='<s>', # TODO <|endoftext|> ???
    im_start     :str     ='<|im_start|>',
    im_end       :str     ='<|im_end|>',
    think        :str     ='<|thought|>',
    group_size   :int     = 64, # Alignment with hardware/batch multiples
    encoding     :str     ='utf-8',
    clobber      :bool    =False,
    threads      :int|None=None,
    adam_iter    :int     =256,
    batch        :int     =4,
    timeout      :int     =5,
)->None:
    raise NotImplementedError()  # TODO application-specific logic
    finetune_llm_adapter(
            exchange_db =exchange_db,
            adapter     =adapter,
            rag         =rag,
            train_data  =train_data,
            sample_start=sample_start,
            im_start    =im_start,
            im_end      =im_end,
            think       =think,
            group_size  =group_size,
            encoding    =encoding,
            clobber     =clobber,
            threads     =threads,
            adam_iter   =adam_iter,
            batch       =batch,
            timeout     =timeout,)

def finetune_semantic_embed_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_semantic_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_episodic_llm_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_episodic_embed_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_episodic_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_reflective_llm_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_reflective_embed_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_reflective_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_executive_llm_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_executive_embed_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_executive_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_metacognitive_llm_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_metacognitive_embed_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic
def finetune_metacognitive_adapter()->None:
    raise NotImplementedError()  # TODO application-specific logic

##
#
##

class ConversationTurn(Enum): # TODO uniq
    SYSTEM    = 'system'
    USER      = 'user'
    ASSISTANT = 'assistant'
    TOOL      = None

    # TODO raise error if anyone tries to get the value of TOOL

class Message(BaseModel):
    turn    :ConversationTurn
    text    :str
    ct_tool :str='user'
    im_start:str='<|im_start|>'
    im_end  :str='<|im_end|>'

    def format(self, ct_tool:str|None=None, im_start: str|None=None, im_end:str|None=None)->str:
        ct_tool :str = ct_tool  or self.ct_tool
        im_start:str = im_start or self.im_start
        im_end  :str = im_end   or self.im_end
        turn    :str = self.turn.value if self.turn.value != ConversationTurn.TOOL else self.ct_tool
        return f'{im_start}{self.turn.value}\n{self.text}\n{im_end}'

class LazyStr(BaseModel):
    value:str|Callable[[],str]

    def __str__(self)->str:
        if isinstance(self.value,str):
            return self.value
        logging.info(f'LazyStr.__str__() resolving: {self.value}')
        if not inspect.iscoroutinefunction(self.value):
            return self.value()
        return asyncio.run(self.value())

class LazyMessage(BaseModel):
    text    :LazyStr|str
    turn    :ConversationTurn
    ct_tool :str='user'
    im_start:str='<|im_start|>'
    im_end  :str='<|im_end|>'

    def resolve(self, ct_tool:str|None=None, im_start:str|None=None, im_end:str|None=None)->Message:
        ct_tool :str = ct_tool  or self.ct_tool
        im_start:str = im_start or self.im_start
        im_end  :str = im_end   or self.im_end
        text    :str = str(self.text)
        return Message(
                turn    =self.turn,
                text    =text,
                ct_tool =ct_tool,
                im_start=im_start,
                im_end  =im_end,)

    def format(self, ct_tool:str|None=None, im_start: str|None=None, im_end:str|None=None)->str:
        message:Message = self.resolve(ct_tool=ct_tool, im_start=im_start, im_end=im_end)
        return message.format(ct_tool=ct_tool, im_start=im_start, im_end=im_end)

@dataclass
class Conversation():
    system_prompt:str|Lazystr|None          = None
    instructions :List[str|LazyStr]         = field(default_factory=list)
    history      :List[Message|LazyMessage] = field(default_factory=list)
    ct_tool      :str                       = 'user'
    im_start     :str                       = '<|im_start|>'
    im_end       :str                       = '<|im_end|>'

    def get_system_prompt(self, ct_tool:str|None=None, im_start:str|None=None, im_end:str|None=None)->Message|None:
        logging.info(f'get_system_prompt(ct_tool={ct_tool}, im_start={im_start}, im_end={im_end})')
        messages:List[str|LazyStr] = []
        if self.system_prompt:
            messages.append(self.system_prompt)
        messages.extend(self.instructions)
        logging.info(f'n_system_prompt: {len(messages)}')
        if not messages:
            return None
        message :str               = '\n'.join(messages)
        logging.info(f'system prompt: {message}')
        ct_tool :str = ct_tool  or self.ct_tool
        im_start:str = im_start or self.im_start
        im_end  :str = im_end   or self.im_end
        if not message:
            logging.warn(f'no system prompt')
        return Message(
                turn    =ConversationTurn.SYSTEM,
                text    =message,
                ct_tool =ct_tool,
                im_start=im_start,
                im_end  =im_end,)

    def append_user_prompt(self, user_prompt:str|LazyStr, ct_tool:str|None=None, im_start:str|None=None, im_end:str|None=None)->None:
        ct_tool :str = ct_tool  or self.ct_tool
        im_start:str = im_start or self.im_start
        im_end  :str = im_end   or self.im_end
        message:LazyMessage= LazyMessage(
                turn    =ConversationTurn.USER,
                text    =user_prompt,
                ct_tool =ct_tool,
                im_start=im_start,
                im_end  =im_end,)
        self.history.append(message)

    def append_agent_response(self, agent_response:str|LazyStr, ct_tool:str|None=None, im_start:str|None=None, im_end:str|None=None):
        ct_tool :str = ct_tool  or self.ct_tool
        im_start:str = im_start or self.im_start
        im_end  :str = im_end   or self.im_end
        message:LazyMessage= LazyMessage(
                turn    =ConversationTurn.ASSISTANT,
                text    =agent_response,
                ct_tool =ct_tool,
                im_start=im_start,
                im_end  =im_end,)
        self.history.append(message)

    def append_tool_result(self, result:str, ct_tool:str|None=None, im_start:str|None=None, im_end:str|None=None)->None:
        ct_tool :str = ct_tool  or self.ct_tool
        im_start:str = im_start or self.im_start
        im_end  :str = im_end   or self.im_end
        message:LazyMessage= LazyMessage(
                turn    =ConversationTurn.TOOL,
                text    =result,
                ct_tool =ct_tool,
                im_start=im_start,
                im_end  =im_end,)
        self.history.append(message)

    def edit_user_message(self, i:int, user_prompt:str|LazyStr, ct_tool:str|None=None, im_start:str|None=None, im_end:str|None=None)->None:
        """replace history[i] with a user-formatted message"""
        logging.info(f'edit_user_message(i={i}, ct_tool={ct_tool}, im_start={im_start}, im_end={im_end})')
        ct_tool :str = ct_tool  or self.ct_tool
        im_start:str = im_start or self.im_start
        im_end  :str = im_end   or self.im_end
        message:LazyMessage= LazyMessage(
                turn    =ConversationTurn.USER,
                text    =user_prompt,
                ct_tool =ct_tool,
                im_start=im_start,
                im_end  =im_end,)
        self.history[i]    = message

    def edit_agent_message(self, i:int, agent_response:str|LazyStr, ct_tool:str|None=None, im_start:str|None=None, im_end:str|None=None)->None:
        """replace history[i] with an agent-formatted message"""
        logging.info(f'edit_agent_message(i={i}, ct_tool={ct_tool}, im_start={im_start}, im_end={im_end})')
        ct_tool :str = ct_tool  or self.ct_tool
        im_start:str = im_start or self.im_start
        im_end  :str = im_end   or self.im_end
        message:LazyMessage= LazyMessage(
                turn    =ConversationTurn.ASSISTANT,
                text    =agent_response,
                ct_tool =ct_tool,
                im_start=im_start,
                im_end  =im_end,)
        self.history[i]    = message

    def edit_messages(self, i:int, j:int, history:List[Message|LazyMessage])->None:
        logging.info(f'edit_message(i={i}, j={j})')
        self.history[i:j]  = history

    def _current_turn(self, user_prompt:str|None)->ConversationTurn|None:
        if user_prompt:
            return ConversationTurn.USER
        if not self.history:
            logging.warning(f'no messages')
            return None
        return self.history[-1].turn

    def to_llm_prompt(self, user_prompt:str|None=None, ct_tool:str|None=None, im_start:str|None=None, im_end:str|None=None)->str:
        logging.info(f'to_llm_prompt(ct_tool={ct_tool}, im_start={im_start}, im_end={im_end})')

        turn    :ConversationTurn = self._current_turn(user_prompt=user_prompt)
        if turn not in [ConversationTurn.USER, ConversationTurn.TOOL]:
            logging.warning(f'skipped user prompt: {turn}')

        ct_tool :str              = ct_tool  or self.ct_tool
        im_start:str              = im_start or self.im_start
        im_end  :str              = im_end   or self.im_end
        if user_prompt:
            self.append_user_prompt(user_prompt=user_prompt, ct_tool=ct_tool, im_start=im_start, im_end=im_end)
        history :List[str]        = [
                m.format(ct_tool=ct_tool, im_start=im_start, im_end=im_end)
                for m in self.history]
        system_prompt:Message|None = self.get_system_prompt(ct_tool=ct_tool, im_start=im_start, im_end=im_end)
        system :List[str]         = (
                [system_prompt.format(ct_tool=ct_tool, im_start=im_start, im_end=im_end)]
                if system_prompt
                else [])
        assist :str               = '\n{im_start}assistant\n'
        prompts:List[str]         = system + history
        logging.info(f'n_prompts: {len(prompts)}')
        prompts                  += [assist]
        return '\n'.join(prompts)

    def deepcopy(self):
        return deepcopy(self)

    def simple_delta(self, conversation:'Conversation', prefer:'Conversation'=self)->'Conversation':
        logging.info(f'simple_delta()')
        assert (prefer == self or prefer == conversation)
        i_self :int           = len(self        .history)
        i_other:int           = len(conversation.history)
        assert i_self <= i_other
        if i_self == i_other:
            logging.warning(f'no dela')
        if self.system_prompt != conversation.system_prompt:
            logging.warning(f'disparate system prompts')
        if len(self.instructions) != len(conversation.instructions):
            logging.warning(f'disparate numbers of instructions')
        if self.ct_tool != conversation.ct_tool:
            logging.warning(f'disparate tool turn resolution')
        if self.im_start != conversation.im_start:
            logging.warning(f'disparate message start delims')
        if self.im_end != conversation.im_end:
            logging.warning(f'disparate message end delims')
        delta  :List[Message] = conversation.history[i_self:] # TODO convert ct_tool, im_start, im_end ?
        logging.info(f'n_delta: {len(delta)}')
        return Conversation(
                system_prompt=prefer.system_prompt,
                instructions =prefer.instructions,
                history      =delta,
                ct_tool      =prefer.ct_tool,
                im_start     =prefer.im_start,
                im_end       =prefer.im_end,)

    def __len__(self)->int:
        return len(self.history)

def _get_llm_response(
    llm           : Llama,
    prompt        : str,
    grammar       : LlamaGrammar,
    im_start      : str='<|im_start|>',
    im_end        : str='<|im_end|>',
    requested     : int = 500,
    threshold     : int = 10,
    encoding      : str = 'utf-8',
) -> dict: # TODO typehint
    logging.info(f'_get_llm_response(im_start={im_start}, im_end={im_end}, requested={requested}, threshold={threshold}, encoding={encoding})')
    safe_limit      :int       = calculate_safe_max_tokens(
        llm, prompt, requested=requested, threshold=threshold, encoding=encoding,
    )
    response        :Dict      = llm( # TODO typehint
        prompt    =prompt,
        grammar   =grammar,
        max_tokens=safe_limit,
        stop      =[im_end, im_start], # </s> ???
    )
    # TODO how to use non-zero indices? would we potentially want decision logic ?
    return response['choices'][0]

def get_llm_response(
    llm           : Llama,
    conversation  : Conversation,
    grammar       : LlamaGrammar,
    user_prompt   : str|None=None,
    ct_tool       : str|None=None,
    im_start      : str|None=None,
    im_end        : str|None=None,
    requested     : int     =500,
    threshold     : int     =10,
    encoding      : str     ='utf-8',
) -> str:
    logging.info(f'get_llm_response('
        f'n_convo={len(conversation)}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'encoding={encoding})')
    ct_tool         :str       = ct_tool  or conversation.ct_tool
    im_start        :str       = im_start or conversation.im_start
    im_end          :str       = im_end   or conversation.im_end
    prompt          :str       = conversation.to_llm_prompt(user_prompt=user_prompt, ct_tool=ct_tool, im_start=im_start, im_end=im_end,)
    response        :Dict      = _get_llm_response( # TODO typehint
            llm           =llm,
            prompt        =prompt,
            grammar       =grammar,
            im_start      =im_start,
            im_end        =im_end,
            requested     =requested,
            threshold     =threshold,
            encoding      =encoding,)
    finish_reason   :str|None  = response.get('finish_reason')
    if finish_reason != 'stop':
        logging.warning('unnatural response termination')
    # TODO use other fields ??? 
    return response['text']

def get_llm_response_grammatical(
        llm         :Llama,
        conversation:Conversation,
        grammar     :str,
        user_prompt :str|None=None,
        ct_tool     :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int=500,
        threshold   :int=10,
        encoding    :str = 'utf-8',
) -> str:
    logging.info(f'get_llm_response_grammatical('
        f'n_convo={len(conversation)}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'encoding={encoding})')
    _grammar:LlamaGrammar = LlamaGrammar.from_string(grammar)
    text        :str          = get_llm_response(
            llm         =llm,
            user_prompt =user_prompt,
            conversation=conversation,
            grammar     =_grammar,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            encoding    =encoding,)
    return text

def request_grammars(
    url:str = 'https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
)->List[str]:
    logging.info(f'request_grammars(url={url})')
    response                   = requests.get(url) # TODO typehint
    response.raise_for_status()
    files       :Dict[str,Any] = response.json() # TODO JSON
    grammars    :List[str]     = [ # Extract names like 'python' from 'python.gbnf'
            f['name'].replace('.gbnf', '')
            for f in files
            if f['name'].endswith('.gbnf') ]
    logging.info(f'n_grammars: {len(grammars)}')
    assert grammars
    return grammars

def read_grammar_list(grammar_list:Path, encoding:str='utf-8')->List[str]:
    logging.info(f'read_grammar_list(grammar_list={grammar_list}, encoding={encoding})')
    assert grammar_list.is_file()
    grammar :str           = grammar_list.read_text(encoding=encoding)
    grammars:List[str]     = grammar.splitlines()
    grammars               = [
            line.strip()
            for line in grammars
            if line.strip()]
    logging.info(f'n_grammars: {len(grammars)}')
    return grammars

def _write_grammar_list_sanity_check(grammar_list:Path, grammars:List[str], encoding:str='utf-8')->bool:
    logging.info(f'_write_grammar_list_sanity_check(grammar_list={grammar_list}, grammars={grammars}, encoding={encoding})')
    with grammar_list.open(encoding=encoding) as f:
        lines:int = sum(1 for _ in f)
    return (lines == len(grammars))

def write_grammar_list(grammar_dir:Path, grammar_list:Path, grammars:List[str], encoding:str='utf-8')->None:
    logging.info(f'write_grammar_list(grammar_dir={grammar_dir}, grammar_list={grammar_list}, n_grammars={len(grammars)}, encoding={encoding})')
    grammar_dir.mkdir(parents=True, exist_ok=True)
    grammar     :str           = '\n'.join(grammars)
    grammar_list.write_text(grammar, encoding=encoding)
    assert grammar_list.is_file()
    assert _write_grammar_list_sanity_check(grammar_list=grammar_list, grammars=grammars, encoding=encoding)

def request_grammars_if_necessary(
    grammar_dir:Path,
    url        :str = 'https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
    clobber    :bool=False,
    encoding   :str ='utf-8',
)->List[str]:
    logging.info(f'request_grammars_if_necessary(grammar_dir={grammar_dir}, url={url}, clobber={clobber}, encoding={encoding})')
    grammar_list:Path          = grammar_dir / 'grammars.lst'
    if (not clobber) and grammar_list.exists():
        logging.info(f'already exists: {grammar_list}')
        return read_grammar_list(grammar_list=grammar_list, encoding=encoding)
    assert clobber or (not grammar_list.exists())
    grammars    :List[str]     = request_grammars(url=url)
    write_grammar_list(grammar_dir=grammar_dir, grammar_list=grammar_list, grammars=grammars, encoding=encoding)
    return grammars

def request_grammar(
    lang       :str,
    #api        :str = 'https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
    url        :str = 'https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
)->str:
    logging.info(f'request_grammar(lang={lang}, url={url})')
    #assert lang in request_grammars_if_necessary(grammar_dir, url, clobber=False, encoding=encoding)
    _url        :str           = f'{url}/{lang}.gbnf' # TODO urljoin
    response                   = requests.get(_url) # TODO typehint
    response.raise_for_status()
    return response.text

def request_grammar_if_necessary(
    lang       :str,
    grammar_dir:Path,
    api        :str = 'https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
    url        :str = 'https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
    clobber    :bool=False,
    encoding   :str ='utf-8',
)->str:
    logging.info(f'request_grammar_if_necessary(lang={lang}, grammar_dir={grammar_dir}, api={api}, url={url}, clobber={clobber}, encoding={encoding})')
    assert lang in request_grammars_if_necessary(grammar_dir, api, clobber=clobber, encoding=encoding)
    grammar_path:Path          = grammar_dir / f'{lang}.gbnf'
    if (not clobber) and grammar_path.exists():
        logging.info(f'already exists: {grammar_path}')
        assert grammar_path.is_file()
        return grammar_path.read_text(encoding=encoding)
    assert clobber or (not grammar_path.exists())
    grammar     :str           = request_grammar(lang=lang, url=url)
    grammar_dir.mkdir(parents=True, exist_ok=True)
    grammar_path.write_text(grammar, encoding=encoding)
    return grammar

def get_llm_response_lang(
        llm         :Llama,
        conversation:Conversation,
        lang        :str,
        grammar_dir:Path,
        user_prompt :str|None=None,
        ct_tool     :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int     =500,
        threshold   :int     =10,
        api         :str     ='https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
        url         :str     ='https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
        clobber     :bool    =False,
        encoding    :str     ='utf-8',
)->str:
    logging.info(f'get_llm_response_lang('
        f'n_convo={len(conversation)}, '
        f'lang={lang}, '
        f'grammar_dir={grammar_dir}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'api={api}, '
        f'url={url}, '
        f'clobber={clobber}, '
        f'encoding={encoding})')
    grammar     :str           = request_grammar_if_necessary(lang, grammar_dir=grammar_dir, api=api, url=url, clobber=clobber, encoding=encoding)
    return get_llm_response_grammatical(
            llm         =llm,
            user_prompt =user_prompt,
            conversation=conversation,
            grammar     =grammar,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            encoding    =encoding,)

def get_llm_response_arithmetic(
        llm         :Llama,
        conversation:Conversation,
        grammar_dir:Path,
        ct_tool     :str|None=None,
        user_prompt :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int     =500,
        threshold   :int     =10,
        api         :str     ='https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
        url         :str     ='https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
        clobber     :bool    =False,
        encoding    :str     ='utf-8',
)->str:
    return get_llm_response_lang(
            llm         =llm,
            user_prompt =user_prompt,
            conversation=conversation,
            lang        ='arithmetic',
            grammar_dir =grammar_dir,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            api         =api,
            url         =url,
            clobber     =clobber,
            encoding    =encoding,)

def get_llm_response_c(
        llm         :Llama,
        conversation:Conversation,
        grammar_dir:Path,
        user_prompt :str|None=None,
        ct_tool     :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int     =500,
        threshold   :int     =10,
        api         :str     ='https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
        url         :str     ='https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
        clobber     :bool    =False,
        encoding    :str     ='utf-8',
)->str:
    return get_llm_response_lang(
            llm         =llm,
            user_prompt =user_prompt,
            conversation=conversation,
            lang        ='c',
            grammar_dir =grammar_dir,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            api         =api,
            url         =url,
            clobber     =clobber,
            encoding    =encoding,)

def get_llm_response_chess(
        llm         :Llama,
        conversation:Conversation,
        grammar_dir:Path,
        user_prompt :str|None=None,
        ct_tool     :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int     =500,
        threshold   :int     =10,
        api         :str     ='https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
        url         :str     ='https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
        clobber     :bool    =False,
        encoding    :str     ='utf-8',
)->str:
    return get_llm_response_lang(
            llm         =llm,
            user_prompt =user_prompt,
            conversation=conversation,
            lang        ='chess',
            grammar_dir =grammar_dir,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            api         =api,
            url         =url,
            clobber     =clobber,
            encoding    =encoding,)

def get_llm_response_english(
        llm         :Llama,
        conversation:Conversation,
        grammar_dir:Path,
        user_prompt :str|None=None,
        ct_tool     :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int     =500,
        threshold   :int     =10,
        api         :str     ='https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
        url         :str     ='https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
        clobber     :bool    =False,
        encoding    :str     ='utf-8',
)->str:
    return get_llm_response_lang(
            llm         =llm,
            user_prompt =user_prompt,
            conversation=conversation,
            lang        ='english',
            grammar_dir =grammar_dir,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            api         =api,
            url         =url,
            clobber     =clobber,
            encoding    =encoding,)

def get_llm_response_japanese(
        llm         :Llama,
        conversation:Conversation,
        grammar_dir:Path,
        user_prompt :str|None=None,
        ct_tool     :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int     =500,
        threshold   :int     =10,
        api         :str     ='https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
        url         :str     ='https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
        clobber     :bool    =False,
        encoding    :str     ='utf-8',
)->str:
    return get_llm_response_lang(
            llm         =llm,
            user_prompt =user_prompt,
            conversation=conversation,
            lang        ='japanese',
            grammar_dir =grammar_dir,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            api         =api,
            url         =url,
            clobber     =clobber,
            encoding    =encoding,)

def get_llm_response_json(
        llm         :Llama,
        conversation:Conversation,
        grammar_dir:Path,
        user_prompt :str|None=None,
        ct_tool     :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int     =500,
        threshold   :int     =10,
        api         :str     ='https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
        url         :str     ='https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
        clobber     :bool    =False,
        encoding    :str     ='utf-8',
)->str:
    return get_llm_response_lang(
            llm         =llm,
            user_prompt =user_prompt,
            conversation=conversation,
            lang        ='json',
            grammar_dir =grammar_dir,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            api         =api,
            url         =url,
            clobber     =clobber,
            encoding    =encoding,)

def get_llm_response_json_arr(
        llm         :Llama,
        conversation:Conversation,
        grammar_dir:Path,
        user_prompt :str|None=None,
        ct_tool     :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int     =500,
        threshold   :int     =10,
        api         :str     ='https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
        url         :str     ='https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
        clobber     :bool    =False,
        encoding    :str     ='utf-8',
)->str:
    return get_llm_response_lang(
            llm         =llm,
            user_prompt =user_prompt,
            conversation=conversation,
            lang        ='json_arr',
            grammar_dir =grammar_dir,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            api         =api,
            url         =url,
            clobber     =clobber,
            encoding    =encoding,)

def get_llm_response_list(
        llm         :Llama,
        conversation:Conversation,
        grammar_dir:Path,
        user_prompt :str|None=None,
        ct_tool     :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int     =500,
        threshold   :int     =10,
        api         :str     ='https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
        url         :str     ='https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
        clobber     :bool    =False,
        encoding    :str     ='utf-8',
)->str:
    return get_llm_response_lang(
            llm         =llm,
            user_prompt =user_prompt,
            conversation=conversation,
            lang        ='list',
            grammar_dir =grammar_dir,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            api         =api,
            url         =url,
            clobber     =clobber,
            encoding    =encoding,)

M:TypeVar = TypeVar('M', bound=BaseModel)

def get_llm_response_pydantic(
        llm         :Llama,
        conversation:Conversation,
        output_type :Type[M],
        user_prompt :str|None=None,
        ct_tool     :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int=500,
        threshold   :int=10,
        encoding    :str='utf-8',
) -> M:
    logging.info(f'get_llm_response_pydantic('
        f'n_convo={len(conversation)}, '
        f'output_type={output_type}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'encoding={encoding})')
    model       :Dict[str,Any] = output_type.model_json_schema() # TODO JSON type
    schema      :str           = json.dumps(model)
    grammar     :LlamaGrammar  = LlamaGrammar.from_json_schema(schema)
    text        :str           = get_llm_response(
            llm         =llm,
            user_prompt =user_prompt,
            conversation=conversation,
            grammar     =grammar,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            encoding    =encoding,)
    return output_type.model_validate_json(text)

class Tool(BaseModel):
    name       : str
    description: str|None
    parameters : Dict[str, Any] # TODO JSON
    # TODO return type ?

    def format(self)->str:
        params:str = json.dumps(self.parameters)
        desc  :str = f': {self.description}' if self.description else ''
        return f'{self.name}{desc}. Params: {params}'

# TODO NoneTool for noops ?

class ToolCall(BaseModel):
    thought    : str
    action     : str
    #params     : Dict[str,str]
    params     : Dict[str,Any]

def get_tools_description(tools:List[Tool])->str|None:
    if not tools:        
        return None
    descriptions :List[str]    = [tool.format() for tool in tools]
    descriptions               = [f'- {d}' for d in descriptions]
    descriptions.insert(0, 'Available Tools:')
    return '\n'.join(descriptions)

def set_tools_description(conversation:Conversation, tools:List[Tool])->None:
    description:str|None = get_tools_description(tools)
    conversation.instructions.append(description)

def get_llm_response_tool_call(
        llm         :Llama,
        conversation:Conversation,
        tools       :List[Tool],
        user_prompt :str|None=None,
        ct_tool     :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int     =500,
        threshold   :int     =10,
        encoding    :str     ='utf-8',
)->ToolCall:
    logging.info(f'get_llm_response_tool_call('
        f'n_convo={len(conversation)}, '
        f'n_tools={len(tools)}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'encoding={encoding})')
    if not tools:
        logging.warning(f'no tools')
    _conversation:Conversation = conversation.deepcopy()
    #descriptions :List[str]    = [tool.format() for tool in tools]
    #descriptions               = [f'- {d}' for d in descriptions]
    #descriptions.insert(0, 'Available Tools:')
    #description  :str          = '\n'.join(descriptions)
    #_conversation.instructions.append(description)
    set_tools_description(conversation=_conversation, tools=tools)
    tool         :ToolCall     = get_llm_response_pydantic(
            llm         =llm,
            conversation=_conversation,
            output_type =ToolCall,
            user_prompt =user_prompt,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            encoding    =encoding,)
    logging.info(f'tool: {tool}')
    return tool

#def format_function_parameter(name: str, param: inspect.Parameter) -> Dict[str, Any]:
#    metadata: Dict[str, Any] = {}
#    annotation = param.annotation
#
#    if annotation != inspect._empty:
#        # Handle Union types (e.g., str | None or Optional[int])
#        if get_origin(annotation) is UnionType or get_origin(annotation) is Union:
#            # Filter out NoneType to find the 'real' type
#            args = [arg for arg in get_args(annotation) if arg is not type(None)]
#            if args:
#                # Take the first non-None type for the prompt description
#                metadata['type'] = getattr(args[0], '__name__', str(args[0]))
#            else:
#                metadata['type'] = 'Any'
#        else:
#            metadata['type'] = getattr(annotation, '__name__', str(annotation))
#
#    # Always include a hint about it being optional if it has a default
#    if param.default != inspect._empty:
#        metadata['optional'] = True
#        if param.default is not None:
#             metadata['default'] = str(param.default)
#
#    return metadata
def format_function_parameter(name: str, param: Parameter) -> Dict[str, Any]:
    _param: Dict[str, Any] = {}
    annotation = param.annotation

    if annotation != inspect._empty:
        # Check if it's a Union (e.g. str | int) or Optional (str | None)
        origin = get_origin(annotation)
        if origin is UnionType or origin is Union:
            # Get all types in the Union, filtering out None
            sub_types = [t for t in get_args(annotation) if t is not type(None)]
            # Tell the LLM the primary type
            _param['type'] = sub_types[0].__name__ if sub_types else "Any"
            _param['description'] = f"Can be {', '.join([t.__name__ for t in sub_types])}"
        else:
            _param['type'] = getattr(annotation, '__name__', str(annotation))

    if param.default != inspect._empty:
        _param['default'] = str(param.default)
        _param['optional'] = True

    return _param

def function_to_tool(func: Callable[[...],Any]) -> Tool:
    signature            = inspect.signature(func) # TODO typehint
    params:Dict[str,Any] = { # TODO JSON
        name: format_function_parameter(name, param)
        for name, param in signature.parameters.items()
    }
    doc   :str|None      = func.__doc__
    doc                  = doc.strip() if doc else None
    return Tool(
        name=func.__name__,
        description=doc,
        parameters=params
    )

def create_function_signature_model(func:Callable[[...],Any])->BaseModel:
    name  :str           = f'{func.__name__}_{id(func)}'
    sig                  = inspect.signature(func) # TODO typehint
    fields:Dict[str,Any] = {}
    for name, param in sig.parameters.items():
        annotation       = ( # TODO typehint
                param.annotation
                if param.annotation != inspect._empty
                else Any)
        default          = ( # TODO typehint
                param.default
                if param.default != inspect._empty
                else ...)
        fields[name]     = (annotation, default)
    return create_model(name, **fields)

@dataclass
class FunctionCall():
    tool_call  :ToolCall
    #thought    : str
    #action     : str

    @property
    def thought(self)->str:
            return self.tool_call.thought

    @property
    def action(self)->str:
        return self.tool_call.action

    @property
    def params(self)->Dict[str,Any]: # TODO JSON
        return self.tool_call.params

@dataclass
class FunctionCallResult(FunctionCall):
    result     :Any          |None
    indent     :int                = 2
    width      :int                = 80
    depth      :int                = 5
    requested  :int                = 500
    threshold  :int                = 10
    encoding   :str                = 'utf-8'

    def format(self, llm:Llama,
               indent:int|None=None, width:int|None=None, depth:int|None=None,
               requested:int|None=None, threshold:int|None=None, encoding:str|None=None,)->str:
        indent   :int = indent if indent is not None else self.indent
        width    :int = width  if width  is not None else self.width
        depth    :int = depth  if depth  is not None else self.depth
        requested:int = requested or self.requested
        threshold:int = threshold or self.threshold
        encoding :str = encoding  or self.encoding
        prompt          :str       = pprint.pformat(self.result, indent=indent, width=width, depth=depth)
        safe_limit      :int       = calculate_safe_max_tokens(
            llm, prompt, requested=requested, threshold=threshold, encoding=encoding,)
        if (safe_limit <= threshold):
            limit:int              = (threshold - safe_limit)
            assert (limit >= len(prompt))
            prompt                 = prompt[:limit]
        return prompt

@dataclass
class FunctionCallUnknown(FunctionCall):
    pass

@dataclass
class FunctionCallError(FunctionCall):
    error      : BaseException

@dataclass
class FunctionCallNone(FunctionCall):
    pass

#def _get_llm_response_funcion_call_handle_error(
#        error         :BaseException,
#        tool_call     :ToolCall,
#        raise_on_error:bool,
#)->FunctionCallError:
#        if raise_on_error:
#            raise error
#        logging.error(error)
#        return FunctionCallError(
#            thought=tool_call.thought,
#            action =tool_call.action,
#            error  =error,)

class FunctionCallUnknownError(ValueError):
    def __init__(self, *args,
                 #thought:str, action:str,
                 tool_call:ToolCall,
                 **kwargs)->None:
        super().__init__(*args, **kwargs)
        #self.thought:str = thought
        #self.action :str = action
        self.tool_call:ToolCall = tool_call

        @property
        def thought(self)->str:
            return self.tool_call.thought

        @property
        def action(self)->str:
            return self.tool_call.action

        @property
        def params(self)->Dict[str,Any]: # TODO JSON
            return self.tool_call.params

class FunctionCallValidationError(ValidationError):
    def __init__(self, *args,
                 #thought:str, action:str,
                 error    :BaseException,
                 tool_call:ToolCall,
                 **kwargs)->None:
        super().__init__(*args, **kwargs)
        #self.thought:str = thought
        #self.action :str = action
        self.error    :BaseException = error
        self.tool_call:ToolCall      = tool_call

        @property
        def thought(self)->str:
            return self.tool_call.thought

        @property
        def action(self)->str:
            return self.tool_call.action

        @property
        def params(self)->Dict[str,Any]: # TODO JSON
            return self.tool_call.params

class FunctionCallRetriesError(Exception):
    def __init__(self, *args,
                 #thought:str, action:str,
                 tool_call:ToolCall,
                 **kwargs)->None:
        super().__init__(*args, **kwargs)
        #self.thought:str = thought
        #self.action :str = action
        self.tool_call:ToolCall = tool_call

        @property
        def thought(self)->str:
            return self.tool_call.thought

        @property
        def action(self)->str:
            return self.tool_call.action

        @property
        def params(self)->Dict[str,Any]: # TODO JSON
            return self.tool_call.params

class FunctionCallException(Exception):
    def __init__(self, *args,
                 #thought:str, action:str,
                 error    :BaseException,
                 tool_call:ToolCall,
                 **kwargs)->None:
        super().__init__(*args, **kwargs)
        #self.thought:str = thought
        #self.action :str = action
        self.error    :BaseException = error
        self.tool_call:ToolCall      = tool_call

        @property
        def thought(self)->str:
            return self.tool_call.thought

        @property
        def action(self)->str:
            return self.tool_call.action

        @property
        def params(self)->Dict[str,Any]: # TODO JSON
            return self.tool_call.params

FunctionCallBaseException:TypeAlias = FunctionCallUnknownError|FunctionCallValidationError|FunctionCallRetriesError|FunctionCallException

def get_llm_response_function_call(
        llm           :Llama,
        conversation  :Conversation,
        funcs         :List[Callable[[...],Any]],
        user_prompt   :str|None=None,
        allow_none    :bool    =False,
        raise_on_error:bool    =True,
        ct_tool       :str|None=None,
        im_start      :str|None=None,
        im_end        :str|None=None,
        requested     :int     =500,
        threshold     :int     =10,
        encoding      :str     ='utf-8',
)->FunctionCall:
    logging.info(f'get_llm_response_function_call('
        f'n_convo={len(conversation)}, '
        f'n_func={len(funcs)}, '
        f'allow_none={allow_none}, '
        f'raise_on_error={raise_on_error}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'encoding={encoding})')
    if not funcs:
        logging.warning(f'no funcs')
    names     :Dict[str,Callable]       = {f.__name__: f for f in funcs}
    tools     :List[Tool]               = list(map(function_to_tool, funcs))
    tool_call :ToolCall                 = get_llm_response_tool_call(
            llm         =llm,
            conversation=conversation,
            tools       =tools,
            user_prompt =user_prompt,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            encoding    =encoding,)
    if allow_none and (tool_call.action.lower() == 'none'):
        logging.info(f'no action')
        return FunctionCallNone(
            #thought=tool_call.thought,
            #action =tool_call.action,)
            tool_call=tool_call,)
    func      :Callable[[...],Any]|None = names.get(tool_call.action)
    if (not func) and raise_on_error:
        raise FunctionCallUnknownError(f'unknown action: {tool_call.action}', tool_call=tool_call)#thought=tool_call.thought, action=tool_call.action)
    if (not func):
        logging.warning(f'unknown action: {tool_call.action}')
        return FunctionCallUnknown(
            #thought=tool_call.thought,
            #action =tool_call.action,)
            tool_call=tool_call,)
    Model     :Type[BaseModel]          = create_function_signature_model(func)
    logging.debug(f'Model : {Model}')
    try:
        model :BaseModel                = Model(**tool_call.params)
    except ValidationError as error:
        #error                           = FunctionCallValidationError(thought=tool_call.thought, action=tool_call.action, error)
        #return _get_llm_response_function_call_handle_error(error=error, tool_call=tool_call, raise_on_error=raise_on_error)
        if raise_on_error:
            raise FunctionCallValidationError(error=error, tool_call=tool_call,)#thought=tool_call.thought, action=tool_call.action)
        logging.error(error)
        return FunctionCallError(
            #thought=tool_call.thought,
            #action =tool_call.action,
            tool_call=tool_call,
            error  =error,)
    logging.debug(f'model : {model}')
    params    :Dict[str,Any]            = model.model_dump() # TODO JSON
    logging.debug(f'params: {params}')
    # TODO log function call at INFO level
    try:
        result:Any|None                 = (
                asyncio.run(func(**params))
                if inspect.iscoroutinefunction(func)
                else func(**params))
        logging.info(f'result: {result}')
    except Exception as error:
        #return _get_llm_response_function_call_handle_error(error=error, tool_call=tool_call, raise_on_error=raise_on_error)
        if raise_on_error:
            raise FunctionCallException(error=error, tool_call=tool_call,)#thought=tool_call.thought, action=tool_call.action)
        logging.error(error)
        return FunctionCallError(
            #thought=tool_call.thought,
            #action =tool_call.action,
            tool_call=tool_call,
            error  =error,)
    # TODO log result.format() ?
    return FunctionCallResult(
            #thought=tool_call.thought,
            #action =tool_call.action,
            tool_call=tool_call,
            result =result,)

def _get_llm_response_function_call_with_self_correction_assertions(
        message    :str|None     = None,
        thought    :str|None     = None,
        action     :str|None     = None,
)->None:
    assert message
    assert thought
    assert action

def get_llm_response_function_call_with_self_correction(
        llm           :Llama,
        conversation  :Conversation,
        funcs         :List[Callable[[...],Any]],
        user_prompt   :str|None=None,
        retries       :int|None=1,
        re_prompt     :bool    =False,
        allow_none    :bool    =False,
        raise_on_error:bool    =True,
        ct_tool       :str|None=None,
        im_start      :str|None=None,
        im_end        :str|None=None,
        requested     :int     =500,
        threshold     :int     =10,
        encoding      :str     ='utf-8',
)->FunctionCall:
    logging.info(f'get_llm_response_function_call_with_self_correction('
        f'n_convo={len(conversation)}, '
        f'n_func={len(funcs)}, '
        f'retries={retries}, '
        f're_prompt={re_prompt}, '
        f'allow_none={allow_none}, '
        f'raise_on_error={raise_on_error}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'encoding={encoding})')
    assert (retries is None) or (retries >= 0)
    if (retries == 0):
        logging.warning(f'no retries')
    attempt        :int          = 0
    checkpoint     :Conversation = conversation.deepcopy()
    while (retries is None) or (attempt < retries + 1): # TODO include attempt / retries in context ?
        attempt                 += 1
        logging.info(f'func call attempt: {attempt} / {retries}')
        message    :str|None     = None
        #thought    :str|None     = None
        #action     :str|None     = None
        tool_call  :ToolCall|None = None
        try:
            result :FunctionCall = get_llm_response_function_call(
                llm         =llm,
                conversation=checkpoint,
                funcs       =funcs,
                user_prompt =user_prompt,
                ct_tool     =ct_tool,
                im_start    =im_start,
                im_end      =im_end,
                requested   =requested,
                threshold   =threshold,
                encoding    =encoding,)
            if isinstance(result,FunctionCallResult):
                return result
            if isinstance(result,FunctionCallNone):
                assert allow_none
                return result
            #thought              = result.thought
            #action               = result.action
            tool_call            = result.tool_call
            if isinstance(result,FunctionCallUnknown):
                assert not raise_on_error
                logging.warning(f'function call unknown: {result}')
                message          = f"Error: The tool '{result.action}' does not exist. Please choose from the available tools."
            if isinstance(result,FunctionCallValidationError):
                assert not raise_on_error
                logging.warning(f'function call validation error: {result}')
                message          = f"The tool '{result.action}' failed with error: {result.error}. Fix your parameters?"
            if isinstance(result,FunctionCallError):
                assert not raise_on_error
                logging.warning(f'function call error: {result}')
                message          = f"The tool '{result.action}' failed with error: {result.error}. Fix your parameters?"
            _get_llm_response_function_call_with_self_correction_assertions(message, tool_call.thought, tool_call.action)
        except FunctionCallUnknownError as error:
            logging.error(error)
            message              = str(error) # TODO f"Error: The tool '{result.action}' does not exist. Please choose from the available tools."
            #thought              = error.thought
            #action               = error.action
            tool_call            = error.tool_call
            assert raise_on_error
            _get_llm_response_function_call_with_self_correction_assertions(message, tool_call.thought, tool_call.action)
        except FunctionCallValidationError as error:
            logging.error(error)
            message              = str(error) # TODO f"The tool '{result.action}' failed with error: {result.error}. Fix your parameters?"
            #thought              = error.thought
            #action               = error.action
            tool_call            = error.tool_call
            assert raise_on_error
            _get_llm_response_function_call_with_self_correction_assertions(message, tool_call.thought, tool_call.action)
        except FunctionCallException as error:
            logging.error(error)
            message              = str(error) # TODO f"The tool '{result.action}' failed with error: {result.error}. Fix your parameters?"
            #thought              = error.thought
            #action               = error.action
            tool_call            = error.tool_call
            assert raise_on_error
            _get_llm_response_function_call_with_self_correction_assertions(message, tool_call.thought, tool_call.action)
        except ContextFullError as error:
            # TODO special handling (i.e., returning any of the checkpoint convo) ???
            raise error
        _get_llm_response_function_call_with_self_correction_assertions(message, tool_call.thought, tool_call.action)
        checkpoint.append_tool_result(result=message, ct_tool=ct_tool, im_start=im_start, im_end=im_end)
        if not re_prompt:
            user_prompt          = None
    assert (attempt == retries + 1)
    _get_llm_response_function_call_with_self_correction_assertions(message, tool_call.thought, tool_call.action)
    if raise_on_error:
        raise FunctionCallRetriesError(
                message,
                #thought=thought,
                #action =action,)
                tool_call=tool_call,)
    return FunctionCallError(
            #thought=thought,
            #action =action,
            tool_call=tool_call,
            error  =message,)

@dataclass
class ReActResult:
    steps       : List[FunctionCall] # Every thought/action/observation
    conversation: Conversation       # The deepcopied convo (delta from parameter?) that actually 'happened'

    @property
    def has_steps(self)->bool:
        return bool(len(self.steps))

    @property
    def final_step(self)->FunctionCall:
        assert self.has_steps
        return self.steps[-1]

    @property
    def thought(self)->str:
        return self.final_step.thought

    @property
    def action(self)->str:
        return self.final_step.action

    @property
    def params(self)->Dict[str,Any]: # TODO JSON
        return self.final_step.params

    @property
    def is_result(self)->bool:
        return isinstance(self.final_step,FunctionCallResult)

    @property
    def is_error(self)->bool:
        return isinstance(self.final_step,FunctionCallError)

    @property
    def is_unknown(self)->bool:
        return isinstance(self.final_step,FunctionCallUnknown)

    @property
    def is_none(self)->bool:
        return isinstance(self.final_step,FunctionCallNone)

    @property
    def result(self)->Any:
        assert self.is_result
        return self.final_step.result

    @property
    def error(self)->BaseException:
        assert self.is_error
        return self.final_step.error

class ReActException(Exception):
    def __init__(self, *args,
                 func_error  :FunctionCallBaseException,
                 conversation:Conversation,
                 steps       :List[FunctionCall],
                 **kwargs)->None:
        super().__init__(*args, **kwargs)
        self.func_error  :FunctionCallBaseException = func_error
        self.conversation:Conversation              = conversation
        self.steps       :List[FunctionCall]        = steps

    @property
    def thought(self)->str:
        return self.func_error.thought

    @property
    def action(self)->str:
        return self.func_error.action

    @property
    def params(self)->Dict[str,Any]: # TODO JSON
        return self.func_error.params

    @property
    def is_error(self)->bool:
        if isinstance(self.func_error,FunctionCallValidationError):
            return True
        if isinstance(self.func_error,FunctionCallException):
            return True
        return False

    @property
    def error(self)->BaseException:
        assert self.is_error
        return self.func_error.error

def get_llm_response_react(
        llm           :Llama,
        conversation  :Conversation,
        funcs         :List[Callable[[...],Any]],
        #func_fmts     :Dict[Callable[[...],Any],Callable[[Any],str]|None]|None=None,
        user_prompt   :str|None=None,
        react_retries :int|None=1,
        retries       :int|None=1,
        re_prompt     :bool    =False,
        allow_none    :bool    =False,
        raise_on_error:bool    =True,
        indent        :int     = 2,
        width         :int     = 80,
        depth         :int     = 5,
        ct_tool       :str|None=None,
        im_start      :str|None=None,
        im_end        :str|None=None,
        requested     :int     =500,
        threshold     :int     =10,
        encoding      :str     ='utf-8',
)->ReActResult:
    logging.info(f'get_llm_response_react('
        f'n_convo={len(conversation)}, '
        f'n_func={len(funcs)}, '
        f'react_retries={react_retries}, '
        f'retries={retries}, '
        f're_prompt={re_prompt}, '
        f'allow_none={allow_none}, '
        f'raise_on_error={raise_on_error}, '
        f'indent={indent}, '
        f'width={width}, '
        f'depth={depth}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'encoding={encoding})')
    assert (react_retries is None) or (react_retries >= 0)
    checkpoint                :Conversation       = conversation.deepcopy()
    steps                     :List[FunctionCall] = []
    attempt                   :int                = 0
    func_call                 :FunctionCall|None  = None
    while (react_retries is None) or (attempt < react_retries + 1): # TODO include attempt / retries in context ?
    #for step_no in range(react_retries):
        logging.info(f'ReAct attempt: {attempt} / {react_retries}')
        attempt                 += 1
        try:
            func_call         :FunctionCall       = get_llm_response_function_call_with_self_correction(
                llm           =llm,
                conversation  =checkpoint,
                funcs         =funcs,
                user_prompt   =user_prompt,
                retries       =retries,
                re_prompt     =re_prompt,
                allow_none    =allow_none,
                raise_on_error=raise_on_error,
                ct_tool       =ct_tool,
                im_start      =im_start,
                im_end        =im_end,
                requested     =requested,
                threshold     =threshold,
                encoding      =encoding,)
            steps.append(func_call)
            if isinstance(func_call, FunctionCallResult):
                agent_fmt     :str                = 'Thought: {thought}\nAction: {action}'
                agent_response:str                = agent_fmt.format(thought=func_call.thought, action=func_call.action)
                checkpoint.append_agent_response(agent_response=agent_response, ct_tool=ct_tool, im_start=im_start, im_end=im_end)
                result        :str                = func_call.result.format(
                        llm      =llm,
                        indent   =indent,
                        width    =width,
                        depth    =depth,
                        requested=requested,
                        threshold=threshold,
                        encoding =encoding,)
                checkpoint.append_tool_result   (result        =result,         ct_tool=ct_tool, im_start=im_start, im_end=im_end)
                if not re_prompt:
                    user_prompt                   = None
                continue
            if isinstance(func_call, FunctionCallNone):
                assert allow_none
                return ReActResult(steps=steps, conversation=checkpoint)
            if isinstance(func_call, FunctionCallError):
                assert not raise_on_error
                return ReActResult(steps=steps, conversation=checkpoint)
            assert not isinstance(func_call, FunctionCallUnknown)
        except FunctionCallUnknownError as error:
            logging.error(error)
            assert raise_on_error
            raise ReActException(func_error=error, steps=steps, conversation=checkpoint)
        except FunctionCallValidationError as error:
            logging.error(error)
            assert raise_on_error
            raise ReActException(func_error=error, steps=steps, conversation=checkpoint)
        except FunctionCallException as error:
            logging.error(error)
            assert raise_on_error
            raise ReActException(func_error=error, steps=steps, conversation=checkpoint)
        except ContextFullError as error:
            raise error
    assert (attempt == react_retries + 1)
    assert bool(len(steps)) == bool(func_call)
    if not len(steps):
        assert not func_call
        logging.warning(f'no steps')
        # TODO return None ?
        raise NotImplementedError()
    if (not allow_none) and (len(steps) > 1):
        logging.warning(f'multiple results: {len(steps)}')
    if (not allow_none) and (not len(steps)):
        # TODO raise ?
        ...
        raise NotImplementedError()
    return ReActResult(steps=steps, conversation=checkpoint)

class Critique(BaseModel):
    is_sufficient       : bool
    score               : int  # 1-10
    reasoning           : str
    #missing_information : List[str]
    suggested_correction: str | None

def get_llm_response_critique(
        llm           :Llama,
        conversation  :Conversation,
        #user_prompt   :str|None=None,
        ct_tool       :str|None=None,
        im_start      :str|None=None,
        im_end        :str|None=None,
        requested     :int     =500,
        threshold     :int     =10,
        encoding      :str     ='utf-8',
)->Critique:
    logging.info(f'get_llm_response_critique('
        f'n_convo={len(conversation)}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'encoding={encoding})')
    turn      :ConversationTurn = conversation._current_turn(user_prompt=None)
    if turn != ConversationTurn.ASSISTANT:
        logging.warning(f'critiquing non-agent: {turn}')
    checkpoint:Conversation     = conversation.deepcopy()
    checkpoint.instructions     = [ # NOTE effective system prompt is [checkpoint.system_prompt] + checkpoint.instructions
        "You are a strict Quality Assurance Auditor.",
        "Your goal is to find flaws, missing details, or logical fallacies in the agent's response.",
        "Be pedantic. If the goal is not 100% met, 'is_sufficient' must be false."
    ]
    # TODO include original instructions ?
    user_prompt:str             = (
        "Review the conversation above. Evaluate the agent's performance "
        "relative to the original user request. Are there hallucinations? "
        "Did it skip steps? Provide a suggested correction if it failed."
    )
    return get_llm_response_pydantic(
            llm         =llm,
            conversation=checkpoint,
            output_type =Critique,
            user_prompt =user_prompt, #
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            encoding    =encoding,)

def get_llm_response_react_critique(
        llm             :Llama,
        critic          :Llama,
        conversation    :Conversation,
        funcs           :List[Callable[[...],Any]],
        #func_fmts     :Dict[Callable[[...],Any],Callable[[Any],str]|None]|None=None,
        user_prompt     :str|None=None,
        critic_retries  :int|None=1,
        react_retries   :int|None=1,
        retries         :int|None=1,
        re_prompt       :bool    =False,
        allow_none      :bool    =False,
        raise_on_error  :bool    =True,
        indent          :int     = 2,
        width           :int     = 80,
        depth           :int     = 5,
        ct_tool         :str|None=None,
        im_start        :str|None=None,
        im_end          :str|None=None,
        requested       :int     =500,
        threshold       :int     =10,
        encoding        :str     ='utf-8',
) -> ReActResult:
    logging.info(f'get_llm_response_react_critique('
        f'n_convo={len(conversation)}, '
        f'n_func={len(funcs)}, '
        f'critic_retries={critic_retries}, '
        f'react_retries={react_retries}, '
        f'retries={retries}, '
        f're_prompt={re_prompt}, '
        f'allow_none={allow_none}, '
        f'raise_on_error={raise_on_error}, '
        f'indent={indent}, '
        f'width={width}, '
        f'depth={depth}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'encoding={encoding})')
    assert (critic_retries is None) or (critic_retries >= 0)
    checkpoint                :Conversation       = conversation.deepcopy() # outside the loop ==> append corrections to prompt
    attempt                   :int                = 0
    _user_prompt              :str|None           = user_prompt
    while (critic_retries is None) or (attempt < critic_retries + 1): # TODO include attempt / retries in context ?
        logging.info(f'critique attempt: {attempt} / {critic_retries}')
        attempt                                  += 1
        #checkpoint            :Conversation       = conversation.deepcopy() # inside the loop ==> only remember most recent
        result                :ReActResult        = get_llm_response_react( # NOTE may have 0 or more results
                llm           =llm,
                conversation  =checkpoint,
                funcs         =funcs,
                user_prompt   =_user_prompt,
                react_retries =react_retries,
                retries       =retries,
                re_prompt     =re_prompt,
                allow_none    =allow_none,
                raise_on_error=raise_on_error,
                indent        =indent,
                width         =width,
                depth         =depth,
                ct_tool       =ct_tool,
                im_start      =im_start,
                im_end        =im_end,
                requested     =requested,
                threshold     =threshold,
                encoding      =encoding,)
        delta                 :Conversation       = checkpoint.simple_delta(result.conversation)
        logging.info(f'n_delta: {len(delta)}')
        critique              :Critique           = get_llm_response_critique(
                llm           =critic,
                conversation  =delta,
                ct_tool       =ct_tool,
                im_start      =im_start,
                im_end        =im_end,
                requested     =requested,
                threshold     =threshold,
                encoding      =encoding,)
        if critique.is_sufficient:
            logging.info(f'critique pass; score: {critique.score}')
            return result
        logging.warning(f'critique fail; reason: {critique.reasoning}')
        prompts               :List[str]          = [critique.suggested_correction, user_prompt]
        prompts                                   = list(filter(None, prompts))
        _user_prompt                              = '\n'.join(prompts) if prompts else None
    assert (attempt == critic_retries + 1)
    if raise_on_error:
        # TODO raise
        ...
        raise NotImplementedError()
    if (not allow_none):
        # TODO raise
        ...
        raise NotImplementedError()
    # TODO no result
    raise NotImplementedError()

# TODO harvest critique convo to generate training data ?

class StructuredPlan(BaseModel):
    objectives: List[str]
    reasoning : str

def get_llm_response_structured_plan(
        llm           :Llama,
        conversation  :Conversation,
        funcs         :List[Callable[[...],Any]],
        user_prompt   :str|None=None,
        ct_tool       :str|None=None,
        im_start      :str|None=None,
        im_end        :str|None=None,
        requested     :int=500,
        threshold     :int=10,
        encoding      :str='utf-8',
)->StructuredPlan:
    logging.info(f'get_llm_response_structured_plan('
        f'n_convo={len(conversation)}, '
        f'n_func={len(funcs)}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'encoding={encoding})')
    checkpoint:Conversation     = conversation.deepcopy()
    checkpoint.instructions     = [ # NOTE effective system prompt is [checkpoint.system_prompt] + checkpoint.instructions
        'You are a Strategic Planning Architect.',
        'Your task is to break down a complex user request into a list of discrete, logical objectives.',
        'Each objective will be handed to a sub-agent to execute.',
        'Focus on dependencies: Ensure Step 1 provides the info needed for Step 2.' ]
    tools      :List[Tool]       = list(map(function_to_tool, funcs))
    set_tools_description(conversation=checkpoint, tools=tools) # handles when no tools
    user_prompt:str              = (
        f'Create a structured plan for the following request:\n{user_prompt}\n\n'
        "Be specific. Instead of 'Get data', use 'Get the weather for San Francisco'.")
    return get_llm_response_pydantic(
            llm         =llm,
            conversation=checkpoint,
            output_type =StructuredPlan,
            user_prompt =user_prompt,
            ct_tool     =ct_tool,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            encoding    =encoding,)

# TODO get_llm_response_pydantic_critique

def get_llm_response_structured_plan_critique(
        llm             :Llama,
        critic          :Llama,
        conversation    :Conversation,
        funcs           :List[Callable[[...],Any]],
        user_prompt     :str|None=None,
        critic_retries  :int|None=1,
        ct_tool         :str|None=None,
        im_start        :str|None=None,
        im_end          :str|None=None,
        requested       :int     =500,
        threshold       :int     =10,
        encoding        :str     ='utf-8',
) -> StructuredPlan:
    logging.info(f'get_llm_response_structured_plan_critique('
        f'n_convo={len(conversation)}, '
        f'n_func={len(funcs)}, '
        f'critic_retries={critic_retries}, '
        f'ct_tool={ct_tool}, '
        f'im_start={im_start}, '
        f'im_end={im_end}, '
        f'requested={requested}, '
        f'threshold={threshold}, '
        f'encoding={encoding})')
    assert (critic_retries is None) or (critic_retries >= 0)
    checkpoint                :Conversation       = conversation.deepcopy() # outside the loop ==> append corrections to prompt
    attempt                   :int                = 0
    _user_prompt              :str|None           = user_prompt
    while (critic_retries is None) or (attempt < critic_retries + 1): # TODO include attempt / retries in context ?
        logging.info(f'critique attempt: {attempt} / {critic_retries}')
        #checkpoint            :Conversation       = conversation.deepcopy() # inside the loop ==> only remember most recent
        attempt                                  += 1
        result                :StructuredPlan     = get_llm_response_structured_plan(
                llm           =llm,
                conversation  =checkpoint,
                funcs         =funcs,
                user_prompt   =_user_prompt,
                ct_tool       =ct_tool,
                im_start      =im_start,
                im_end        =im_end,
                requested     =requested,
                threshold     =threshold,
                encoding      =encoding,)
        agent_response        :str                = str(result)
        checkpoint.append_agent_response(agent_response=agent_response, ct_tool=ct_tool, im_start=im_start, im_end=im_end)
        delta                 :Conversation       = conversation.simple_delta(checkpoint)
        logging.info(f'n_delta: {len(delta)}')
        critique              :Critique           = get_llm_response_critique(
                llm           =critic,
                conversation  =delta,
                ct_tool       =ct_tool,
                im_start      =im_start,
                im_end        =im_end,
                requested     =requested,
                threshold     =threshold,
                encoding      =encoding,)
        if critique.is_sufficient:
            logging.info(f'critique pass; score: {critique.score}')
            return result
        logging.warning(f'critique fail; reason: {critique.reasoning}')
        prompts               :List[str]          = [critique.suggested_correction, user_prompt]
        prompts                                   = list(filter(None, prompts))
        _user_prompt                              = '\n'.join(prompts) if prompts else None
    assert (attempt == critic_retries + 1)
    #if raise_on_error:
    #    # TODO raise
    #    ...
    #    raise NotImplementedError()
    #if (not allow_none):
    #    # TODO raise
    #    ...
    #    raise NotImplementedError()
    # TODO no result
    # TODO raise
    raise NotImplementedError()

# TODO harvest critique convo to generate training data ?

def _get_llm_response_structured_plan(
        llm             :Llama,
        conversation    :Conversation,
        funcs           :List[Callable[[...],Any]],
        critic          :Llama|None=None,
        user_prompt     :str  |None=None,
        critic_retries  :int  |None=1,
        ct_tool         :str  |None=None,
        im_start        :str  |None=None,
        im_end          :str  |None=None,
        requested       :int       =500,
        threshold       :int       =10,
        encoding        :str       ='utf-8',
)->StructuredPlan:
    return (get_llm_response_structured_plan_critique(
                llm           =llm,
                critic        =critic,
                conversation  =conversation,
                funcs         =funcs,
                user_prompt   =user_prompt,
                critic_retries=critic_retries,
                ct_tool       =ct_tool,
                im_start      =im_start,
                im_end        =im_end,
                requested     =requested,
                threshold     =threshold,
                encoding      =encoding,)
            if critic else get_llm_response_structured_plan(
                llm           =llm,
                conversation  =conversation,
                funcs         =funcs,
                user_prompt   =user_prompt,
                ct_tool       =ct_tool,
                im_start      =im_start,
                im_end        =im_end,
                requested     =requested,
                threshold     =threshold,
                encoding      =encoding,))

def _get_llm_response_react(
        llm             :Llama,
        conversation    :Conversation,
        funcs           :List[Callable[[...],Any]],
        #func_fmts     :Dict[Callable[[...],Any],Callable[[Any],str]|None]|None=None,
        critic          :Llama|None=None,
        user_prompt     :str  |None=None,
        critic_retries  :int  |None=1,
        react_retries   :int  |None=1,
        retries         :int  |None=1,
        re_prompt       :bool      =False,
        allow_none      :bool      =False,
        raise_on_error  :bool      =True,
        indent          :int       = 2,
        width           :int       = 80,
        depth           :int       = 5,
        ct_tool         :str  |None=None,
        im_start        :str  |None=None,
        im_end          :str  |None=None,
        requested       :int       =500,
        threshold       :int       =10,
        encoding        :str       ='utf-8',
)->ReActResult:
    return (get_llm_response_react(
                llm           =llm,
                conversation  =conversation,
                funcs         =funcs,
                user_prompt   =user_prompt,
                react_retries =react_retries,
                re_prompt     =re_prompt,
                allow_none    =allow_none,
                raise_on_error=raise_on_error,
                indent        =indent,
                width         =width,
                depth         =depth,
                ct_tool       =ct_tool,
                im_start      =im_start,
                im_end        =im_end,
                requested     =requested,
                threshold     =threshold,
                encoding      =encoding,)
            if critic else get_llm_response_react_critique(
                llm           =llm,
                critic        =react_critic,
                conversation  =conversation,
                funcs         =funcs,
                user_prompt   =user_prompt,
                critic_retries=critic_retries,
                react_retries =react_retries,
                re_prompt     =re_prompt,
                allow_none    =allow_none,
                raise_on_error=raise_on_error,
                indent        =indent,
                width         =width,
                depth         =depth,
                ct_tool       =ct_tool,
                im_start      =im_start,
                im_end        =im_end,
                requested     =requested,
                threshold     =threshold,
                encoding      =encoding,))

def get_llm_response_react_structured_plan(
        llm             :Llama,
        planner         :Llama,
        conversation    :Conversation,
        funcs           :List[Callable[[...],Any]],
        #func_fmts     :Dict[Callable[[...],Any],Callable[[Any],str]|None]|None=None,
        user_prompt     :str  |None=None,
        planner_critic  :Llama|None=None,
        react_critic    :Llama|None=None,
        critic_retries  :int  |None=1,
        react_retries   :int  |None=1,
        retries         :int  |None=1,
        re_prompt       :bool      =False,
        allow_none      :bool      =False,
        raise_on_error  :bool      =True,
        indent          :int       = 2,
        width           :int       = 80,
        depth           :int       = 5,
        ct_tool         :str  |None=None,
        im_start        :str  |None=None,
        im_end          :str  |None=None,
        requested       :int       =500,
        threshold       :int       =10,
        encoding        :str       ='utf-8',
)->ReActResult:
    logging.info(f'get_llm_response_react_structured_plan('
                 f'n_convo={len(conversation)}, '
                 f'n_funcs={len(funcs)}, '
                 f'critic_retries={critic_retries}, '
                 f'react_retries={react_retries}, '
                 f'retries={retries}, '
                 f're_prompt={re_prompt}, '
                 f'allow_none={allow_none}, '
                 f'raise_on_error={raise_on_error}, '
                 f'indent={indent}, '
                 f'width={width}, '
                 f'depth={depth}, '
                 f'ct_tool={ct_tool}, '
                 f'im_start={im_start}, '
                 f'im_end={im_end}, '
                 f'requested={requested}, '
                 f'threshold={threshold}, '
                 f'encoding={encoding})')
    checkpoint:Conversation      = conversation.deepcopy()
    plan      :StructuredPlan    = _get_llm_response_structured_plan(
                llm           =llm,
                critic        =planner_critic,
                conversation  =checkpoint,
                funcs         =funcs,
                user_prompt   =user_prompt,
                critic_retries=critic_retries,
                ct_tool       =ct_tool,
                im_start      =im_start,
                im_end        =im_end,
                requested     =requested,
                threshold     =threshold,
                encoding      =encoding,)
    steps     :List[ReActResult] = []
    for objective in plan.objectives:
        logging.info(f'objective: {objective}')
        result:ReActResult       = _get_llm_response_react(
                llm           =llm,
                critic        =react_critic,
                conversation  =checkpoint,
                funcs         =funcs,
                user_prompt   =objective,
                critic_retries=critic_retries,
                react_retries =react_retries,
                re_prompt     =re_prompt,
                allow_none    =allow_none,
                raise_on_error=raise_on_error,
                indent        =indent,
                width         =width,
                depth         =depth,
                ct_tool       =ct_tool,
                im_start      =im_start,
                im_end        =im_end,
                requested     =requested,
                threshold     =threshold,
                encoding      =encoding,)
        steps.append(result)
        if not result.is_error:
            continue
        if raise_on_error:
            # TODO raise
            ...
        # TODO return
        ...
    # TODO return
    ...
    raise NotImplementedError()

# TODO critique-wrapped with-structured-plan
# TODO auto-retriever layer
# TODO critique-wrapped auto-retriever
# TODO training data wrapper around critique wrappers ==> fine-tuning

# TODO "smart" conversation subclass with hierarchical memory, summarization, rag, etc

# TODO false ego: watch another agent's convo, generate self narrative, overwrite other agent's chat history & system prompt

# TODO prompt pipeline: user prompt ==> agent xform + auto-rag ==> false ego wrapped agent ==> agent xform ==> output to user

    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO


# TODO chat history RAG
# TODO chat history summarization
# TODO self-narrative generation
# TODO meta prompting & hyper params
# TODO generate test cases ==> fine-tuning
# TODO generate training data on self-correction ???

def get_model_name(embedding:bool)->str:
    names    :Dict[bool,str] = {
            False: 'qwen3:4b-instruct',
            True : 'nomic-embed-text',
    }
    return names[embedding]

def get_repo_id(name:str)->str:
    repo_ids :Dict[str,str]  = {
            'nomic-embed-text' : 'nomic-ai/nomic-embed-text-v1.5-GGUF',
            'qwen3:4b-instruct': 'bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF',
    }
    return repo_ids[name]

def get_repo_dir(model_dir:Path, name:str)->Path:
    repo_id  :str            = get_repo_id(name)
    logging.info(f'repo_id   : {repo_id}')
    return model_dir / repo_id

def get_model_path(repo_dir:Path, name:str)->Path:
    filenames:Dict[str,str]  = {
            'nomic-embed-text' : 'nomic-embed-text-v1.5.Q4_K_M.gguf',
            'qwen3:4b-instruct': 'Qwen_Qwen3-4B-Instruct-2507-Q6_K.gguf',
    }
    filename :str            = filenames[name]
    logging.info(f'filename  : {filename}')
    return repo_dir  / filename

def get_lora_name(name:str, adapter:str|None)->str|None:
    if not adapter:
        return None
    fmt :str            = '{name}.{adapter}.gguf'
    return fmt.format(name=name, adapter=adapter)

def get_lora_path(repo_dir:Path, name:str, adapter:str|None)->Path|None:
    lora:str |None      = get_lora_name(name=name, adapter=adapter)
    if not lora:
        return None
    logging.info(f'lora      : {lora}')
    return repo_dir  / lora

def pull_my_llama(model_dir:Path, adapter:str|None=None, embedding:bool=False, n_batch:int=512, verbose:bool=False)->Llama:
    logging.info(f'pull_my_llama(model_dir={model_dir}, adapter={adapter}, embedding={embedding}, n_batch={n_batch}, verbose={verbose})')
    name     :str            = get_model_name(embedding=embedding)
    logging.info(f'name      : {name}')

    repo_dir :Path           = get_repo_dir  (model_dir=model_dir, name=name)
    logging.info(f'repo_dir  : {repo_dir}')

    path     :Path           = get_model_path(repo_dir=repo_dir,   name=name)
    logging.info(f'path      : {path}')

    lora_path:Path|None      = get_lora_path (repo_dir=repo_dir,   name=name, adapter=adapter)
    logging.info(f'lora_path : {lora_path}')

    repo_id  :str            = get_repo_id(name)
    return pull_llama(repo_id=repo_id, model_path=path, lora_path=lora_path, embedding=embedding, n_batch=n_batch, verbose=verbose)

@dataclass
class RAG():
    name      :str
    llm       :Llama
    embed     :Llama
    db_path   :Path
    index     :IndexIDMap
    decay     :float
    llm_path  :Path
    embed_path:Path
    llm_lora  :Path
    embed_lora:Path

def get_decay(name:str)->float:
    decays    :Dict[str,float]               = {
        'semantic'     : 0.0,
        'episodic'     : 0.05,
        'reflective'   : 0.01,
        'executive'    : 0.0,
        'metacognitive': 0.1,
    }
    return decays[name]

def get_rag(root:Path, model_dir:Path, name:str, n_batch:int=512, verbose:bool=False, dimensions:int=768)->RAG:
    logging.info(f'get_rag(root={root}, model_dir={model_dir}, name={name}, n_batch={n_batch}, verbose={verbose}, dimensions={dimensions})')
    #pillars   :Dict[str,Dict[str,str|float]] = {
    #    "semantic"     : {"db": "semantic.db",      "index": "semantic.index",      "decay": 0.0},
    #    "episodic"     : {"db": "episodic.db",      "index": "episodic.index",      "decay": 0.05},
    #    "reflective"   : {"db": "reflective.db",    "index": "reflective.index",    "decay": 0.01},
    #    "executive"    : {"db": "executive.db",     "index": "executive.index",     "decay": 0.0},
    #    "metacognitive": {"db": "metacognitive.db", "index": "metacognitive.index", "decay": 0.1},
    #}
    #pillar    :Dict[str,str|float]           = pillars[name]
    llm       :Llama                         = pull_my_llama(model_dir, adapter=name, embedding=False, n_batch=n_batch, verbose=verbose)
    embed     :Llama                         = pull_my_llama(model_dir, adapter=name, embedding=True,  n_batch=n_batch, verbose=verbose)
    db_name   :str                           = f'{name}.db'    #pillar['db']
    index_name:str                           = f'{name}.index' #pillar['index']
    db_path   :Path                          = root / db_name
    index_path:Path                          = root / index_name
    decay     :float                         = get_decay(name) #pillar['decay']
    llm_name  :str                           = get_model_name(embedding=False)
    embed_name:str                           = get_model_name(embedding=True)
    llm_repo  :Path                          = get_repo_dir (model_dir=model_dir,  name=llm_name)
    embed_repo:Path                          = get_repo_dir (model_dir=model_dir,  name=embed_name)
    llm_path  :Path                          = get_model_path(repo_dir=llm_repo,   name=llm_name)
    embed_path:Path                          = get_model_path(repo_dir=embed_repo, name=embed_name)
    llm_lora  :Path                          = get_lora_path(repo_dir =llm_repo,   name=llm_name,   adapter=name)
    embed_lora:Path                          = get_lora_path(repo_dir =embed_repo, name=embed_name, adapter=name)
    sqlite_load_or_create_text_db(db_path)
    index     :IndexIDMap                    = faiss_load_or_create_embedding_db(index_path, dimension=dimensions)
    logging.info(f'db_name   : {db_name}')
    logging.info(f'index_name: {index_name}')
    logging.info(f'db_path   : {db_path}')
    logging.info(f'index_path: {index_path}')
    logging.info(f'decay     : {decay}')
    logging.info(f'llm_name  : {llm_name}')
    logging.info(f'embed_name: {embed_name}')
    logging.info(f'llm_repo  : {llm_repo}')
    logging.info(f'embed_repo: {embed_repo}')
    logging.info(f'llm_path  : {llm_path}')
    logging.info(f'embed_path: {embed_path}')
    logging.info(f'llm_lora  : {llm_lora}')
    logging.info(f'embed_lora: {embed_lora}')
    return RAG(
            name      =name,
            llm       =llm,
            embed     =embed,
            db_path   =db_path,
            index     =index,
            decay     =decay,
            llm_path  =llm_path,
            embed_path=embed_path,
            llm_lora  =llm_lora,
            embed_lora=embed_lora,)

def main()->None:
    name             :str           = 'test-agent-1'
    setup_logging(name)

    root             :Path          = Path('/', 'var', 'lib', 'ia_llm') #Path(os.getcwd()).resolve()
    model_dir        :Path          = root      / '.models'
    grammar_dir      :Path          = root      / '.grammars'
    logging.info(f'root             : {root}')
    logging.info(f'model_dir        : {model_dir}')
    logging.info(f'grammar_dir      : {grammar_dir}')
    
    im_start         :str           = '<|im_start|>' # TODO can dynamically determine ?
    im_end           :str           = '<|im_end|>' # TODO can dynamically determine ?
    n_batch          :int           = 512 # TODO can dynamically determine ?
    verbose          :bool          = False
    logging.info(f'im_start         : {im_start}')
    logging.info(f'im_end           : {im_end}')
    logging.info(f'n_batch          : {n_batch}')
    logging.info(f'verbose          : {verbose}')

    llm              :Llama         = pull_my_llama(model_dir=model_dir, adapter='default', embedding=False, n_batch=n_batch, verbose=verbose)
    rag_semantic     :RAG           = get_rag(root=root, model_dir=model_dir, name='semantic',      n_batch=n_batch, verbose=verbose)
    rag_episodic     :RAG           = get_rag(root=root, model_dir=model_dir, name='episodic',      n_batch=n_batch, verbose=verbose)
    rag_reflective   :RAG           = get_rag(root=root, model_dir=model_dir, name='reflective',    n_batch=n_batch, verbose=verbose)
    rag_executive    :RAG           = get_rag(root=root, model_dir=model_dir, name='executive',     n_batch=n_batch, verbose=verbose)
    rag_metacognitive:RAG           = get_rag(root=root, model_dir=model_dir, name='metacognitive', n_batch=n_batch, verbose=verbose)
    logging.info(f'rag_semantic     : {rag_semantic}')
    logging.info(f'rag_episodic     : {rag_episodic}')
    logging.info(f'rag_reflective   : {rag_reflective}')
    logging.info(f'rag_executive    : {rag_executive}')
    logging.info(f'rag_metacognitive: {rag_metacognitive}')

    llm_c            :Llama         = pull_my_llama(model_dir=model_dir, adapter='c',       embedding=False, n_batch=n_batch, verbose=verbose)
    #llm_bash         :Llama         = pull_my_llama(model_dir=model_dir, adapter='bash',    embedding=False, n_batch=n_batch, verbose=verbose)
    #llm_python       :Llama         = pull_my_llama(model_dir=model_dir, adapter='python',  embedding=False, n_batch=n_batch, verbose=verbose)

    rag_create_document("The user prefers 'apt' over 'nala'.", rag_semantic.embed, rag_semantic.index, rag_semantic.db_path)

    # Search something
    memories = rag_retrieve_document('What package manager does the user like?', rag_semantic.embed, rag_semantic.index, rag_semantic.db_path)

    logging.info(f'grammars: {request_grammars_if_necessary(grammar_dir)}')

    user_prompt      :str           = 'Hello, World!'
    if False:
        response_c       :str           = get_llm_response_c(
                llm         =llm_c,
                user_prompt =user_prompt,
                conversation=Conversation(im_start=im_start, im_end=im_end,),
                grammar_dir =grammar_dir,
                im_start    =im_start,
                im_end      =im_end,
                #requested   =requested,
                #threshold   =threshold,
                #api         ='https://api.github.com/repos/ggerganov/llama.cpp/contents/grammars',
                #url         ='https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars',
                #encoding    ='utf-8',
        )
        logging.info(f'Agent c          : {response_c}')

    if False:
        class Test(BaseModel):
            a:str
            b:int
            c:float
        response        :Test          = get_llm_response_pydantic(
                llm         =llm,
                user_prompt =user_prompt,
                output_type =Test,
                conversation=Conversation(im_start=im_start, im_end=im_end,),)
        logging.info(f'Agent            : {response}')

    if True:
        def hello_world(a:str|None=None, b:int|None=None, c:float|None=None)->str:
            return f'a={a}, b={b}, c={c}'
    
        #response        :FunctionCall           = get_llm_response_function_call_with_self_correction(
        #response        :FunctionCall           = get_llm_response_react(
        response        :FunctionCall           = get_llm_response_react_structured_plan(
                llm           =llm,
                planner       =llm,
                planner_critic=llm,
                react_critic  =llm,
                user_prompt   =user_prompt,
                funcs         =[hello_world,],
                conversation  =Conversation(im_start=im_start, im_end=im_end,),)
        logging.info(f'Agent            : {response}')

    #response_bash    :str           = get_llm_response_bash(
    #        llm         =llm_bash,
    #        user_prompt =user_prompt,
    #        conversation=Conversation(im_start=im_start, im_end=im_end,),
    #        grammar_dir =grammar_dir,
    #        im_start    =im_start,
    #        im_end      =im_end,)
    #logging.info(f'Agent bash        : {response_bash}')
    #
    #response_python    :str          = get_llm_response_python(
    #        llm         =llm_python,
    #        user_prompt =user_prompt,
    #        conversation=Conversation(im_start=im_start, im_end=im_end,),
    #        grammar_dir =grammar_dir,
    #        im_start    =im_start,
    #        im_end      =im_end,)
    #logging.info(f'Agent python      : {response_python}')

    ## Inject into Qwen
    #context = "\n".join([m['content'] for m in memories])
    #prompt          :str           = f"Context: {context}\n\nUser: Should I use nala?"
    #logging.info(f'User           : {prompt}')
    #
    #response        :ToolCall      = get_structured_response(prompt, llm=llm)
    #logging.info(f'Agent (thought): {response.thought}')
    #logging.info(f'Agent (action) : {response.action}')
    #logging.info(f'Agent (params) : {response.params}')

    ## pip install outlines
    #import outlines
    ## 1. Load the model via Outlines' wrapper
    #model = outlines.models.llamacpp(llm) # 'llm' is your existing Llama instance
    ## 2. Define the generator once
    #generator = outlines.generate.json(model, ToolCall)
    ## 3. Use it like a function
    ## It handles the prompt, the grammar, and the parsing in one go
    #result = generator("What's the weather?")
    #print(result.thought)

if __name__ == '__main__':
    main()

