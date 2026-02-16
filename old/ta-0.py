#! /usr/bin/env python

# TODO llama_context: n_ctx_per_seq (512) > n_ctx_train (0) -- possible training context overflow
# TODO llama_context: n_ctx_per_seq (4096) < n_ctx_train (262144) -- the full capacity of the model will not be utilized

#Pillar       ,Stack      ,"""Necessary Extension""",Purpose
#Semantic     ,FAISS + SQL,Relational Linking       ,Traditional RAG; links facts to sources.
#Episodic     ,FAISS + SQL,Temporal Indexing        ,"Chat History"
#Reflective   ,FAISS + SQL,Contradiction Detection  ,"Identity & Narrative"
#Executive    ,FAISS + SQL,Dependency Graph         ,"Tracks sub-tasks. RAG here helps find ""similar past solutions"" to current problems."
#Metacognitive,FAISS + SQL,                         ,"Logic Logs & Error Correction for reflection, self-correction, hyper params, meta prompting, generating test cases, fine-tuning."

import datetime
from enum      import Enum
import json
import logging
import math
import os
from pathlib   import Path
import sys
from types     import *
from typing    import *

import faiss
from faiss     import IndexFlatL2, IndexIDMap
import huggingface_hub
from llama_cpp import Llama, LlamaGrammar
from llama_cpp import llama_backend_init
import llama_cpp
import numpy   as np
from numpy     import ndarray
import psutil
from pydantic  import BaseModel
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

def get_system_num_ctx_capacity() -> int:
    """Estimates safe context size based on available system RAM."""
    # Available RAM in GB
    available_gb = psutil.virtual_memory().available / (1024**3)

    # We want to leave at least 2GB for the OS and other processes
    # A 4B Q6 model takes ~3.5GB.
    if available_gb > 8:
        return 16384  # High performance
    elif available_gb > 4:
        return 4096   # Standard
    elif available_gb > 2:
        return 2048   # Minimal
    else:
        return 512    # Emergency mode

def get_model_num_ctx_train(model_path: Path) -> int:
    """Reads the GGUF metadata to find the model's hard-coded training limit."""
    try:
        # Load ONLY metadata (lightning fast, uses almost no RAM)
        temp_llm = Llama(model_path=str(model_path), vocab_only=True, verbose=False)
        #limit = temp_llm.n_ctx_train()
        limit = temp_llm.context_params.n_ctx
        return limit
    except Exception as error:
        logging.error(error)
        return 2048  # Safe fallback if metadata is missing

#def get_optimal_num_ctx(model_path: Path) -> int:
#    """Returns the best context window size for the current hardware/model combo."""
#    system_cap = get_system_num_ctx_capacity()
#    model_cap = get_model_num_ctx_train(model_path)
#    
#    # Use the smaller of the two to avoid OOM or RoPE scaling issues
#    return min(system_cap, model_cap)

def get_optimal_num_ctx(model_path: Path, embedding: bool) -> int:
    """Balances RAM, Model limits, and Metadata bugs to find the perfect n_ctx."""
    logging.info(f'get_optimal_num_ctx(model_path={model_path}, embedding={embedding})')
    # 1. Check System Capacity
    # Heuristic: ~128MB per 1024 tokens for 4B models.
    available_gb           :int   = psutil.virtual_memory().available / (1024**3)
    system_cap             :int   = 16384 if available_gb > 8 else 4096 if available_gb > 4 else 2048

    # 2. Check Model Training Limit (with metadata-only load)
    try:
        temp_llm           :Llama = Llama(model_path=str(model_path), vocab_only=True, verbose=False)
        #model_train_cap    :int   = temp_llm.n_ctx_train()
        model_train_cap    :int   = temp_llm.context_params.n_ctx

        # FIX: If metadata returns 0 (common in Nomic), set a realistic cap
        if model_train_cap <= 0:
            logging.warning('no n_ctx_train')
            model_train_cap       = 8192 if embedding else 32768
    except Exception as error:
        logging.error(error)
        model_train_cap           = 4096 # Fallback
    
    logging.info(f'available_gb   : {available_gb}')
    logging.info(f'system_cap     : {system_cap}')
    logging.info(f'model_train_cap: {model_train_cap}')

    # Final Decision: Don't exceed what the system can afford,
    # but don't try to exceed what the model was trained for.
    return min(system_cap, model_train_cap)

#def get_gpu_offload_config()->int:
#    if os.path.exists("/dev/nvidiactl") or os.path.exists("/dev/dri"):
#        return -1
#    return 0
def get_gpu_offload_config() -> int:
    """Returns -1 to offload all layers IF the binary supports it, else 0."""
    # We can check the llama-cpp-python build features via the internal 'llama_cpp' module

    # If the build features include CUDA or METAL, offload is viable
    #if any(llama_cpp.llama_supports_gpu_offload()):
    if llama_cpp.llama_supports_gpu_offload():
        return -1 # Offload all layers

    return 0 # Stay on CPU

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
        logging.warn('lora path dne: {lora_path}')
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
            verbose     =verbose,
    )

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

class ConversationTurn(Enum): # uniq
    SYSTEM    = 'system'
    USER      = 'user'
    ASSISTANT = 'assistant'

def format_message(turn:ConversationTurn, message:str, im_start:str='<|im_start|>', im_end:str='<|im_end|>')->str:
    logging.info(f'format_message(turn={turn}, im_start={im_start}, im_end={im_end})')
    fmt   :str = '{im_start}{turn}\n{message}\n{im_end}'
    result:str = fmt.format(turn=turn, message=message, im_start=im_start, im_end=im_end)
    logging.debug(f'message: {message}')
    logging.debug(f'result : {result}')
    return result

def format_system_message(message:str, im_start:str='<|im_start|>', im_end:str='<|im_end|>')->str:
    return format_message(turn=ConversationTurn.SYSTEM, message=message, im_start=im_start, im_end=im_end)

def format_user_message(message:str, im_start:str='<|im_start|>', im_end:str='<|im_end|>')->str:
    return format_message(turn=ConversationTurn.USER, message=message, im_start=im_start, im_end=im_end)

def format_assistant_message(message:str, im_start:str='<|im_start|>', im_end:str='<|im_end|>')->str:
    return format_message(turn=ConversationTurn.ASSISTANT, message=message, im_start=im_start, im_end=im_end)

class ContextStats(BaseModel):
    used:int
    remaining:int
    n_ctx:int
    ratio:float

def get_context_stats(llm: Llama, prompt: str) -> ContextStats:
    """Calculates how many tokens the prompt takes and what's left in the window."""
    # Encode the prompt to tokens
    prompt_tokens     = llm.tokenize(prompt.encode("utf-8"))
    n_past       :int = len(prompt_tokens)

    # Total context window we initialized the model with
    n_ctx        :int = llm.n_ctx()

    # What's remaining for the assistant to speak
    remaining    :int = n_ctx - n_past

    return ContextStats(
            used     =n_past,
            remaining=remaining,
            n_ctx    =n_ctx,
            ratio    =float(n_past) / n_ctx,
    )

class ContextFullError(ValueError):
    """Context window is full. Need to compress or summarize"""

def calculate_safe_max_tokens(llm: Llama, prompt: str, requested: int = 500, threshold:int=10) -> int:
    """Ensures max_tokens doesn't overflow the physical context window."""
    logging.info(f'calculate_safe_max_tokens(requested={requested}, threshold={threshold})')

    stats:ContextStats = get_context_stats(llm, prompt)
    logging.info(f'context stats: {stats}')
    
    # We need to leave at least a few tokens for the response
    if stats.remaining <= threshold:
        raise ContextFullError(str(threshold - stats.remaining))
        
    # Return the smaller of what we want vs what we have room for
    return min(requested, stats.remaining)

##
#
##

#def faiss_create_embedding_db(dimension: int) -> IndexFlatL2:
#    """Creates an empty FAISS index for a specific vector size (e.g., 768 for Nomic)."""
#    return IndexFlatL2(dimension)
def faiss_create_embedding_db(dimension: int) -> IndexIDMap:
    """Creates an IndexIDMap wrapper around FlatL2 to allow custom IDs."""
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

#def faiss_create_embedding(index: faiss.IndexFlatL2, vector: list[float]) -> int: # Crud
#    """
#    Adds a vector to the index.
#    Returns the ID (index position) of the new memory.
#    """
#    # FAISS requires float32 and a specific shape (1, dimension)
#    v_array = np.array([vector]).astype('float32')
#    index.add(v_array)
#    return index.ntotal - 1
def faiss_create_embedding(index: IndexIDMap, vector: List[float], doc_id: int) -> int:
    """Adds a vector with a specific ID (from SQLite)."""
    logging.info(f'faiss_create_embedding(doc_id={doc_id})')
    v_array = np.array([vector]).astype('float32')
    ids_array = np.array([doc_id]).astype('int64')
    index.add_with_ids(v_array, ids_array)
    return doc_id

##def faiss_retrieve_embedding(index: faiss.IndexFlatL2, query_vector: list[float], k: int = 5): # cRud # TODO typehints
##    """
##    Searches for the k-nearest memories.
##    Returns (distances, indices).
##    """
##    q_array = np.array([query_vector]).astype('float32')
##    distances, indices = index.search(q_array, k)
##    return distances[0], indices[0]
#def faiss_retrieve_embedding(index: IndexIDMap, query_vector: List[float], k: int = 5) -> Tuple[ndarray, ndarray]:
#    """Returns (distances, indices)."""
#    logging.info(f'faiss_retrieve_embedding(k={k})')
#    q_array = np.array([query_vector]).astype('float32')
#    distances, indices = index.search(q_array, k)
#    return distances[0], indices[0]
def faiss_retrieve_embedding(
    index       :IndexIDMap,
    query_vector:List[float],
    timestamps  :Dict[int, float]|None=None, # Map of doc_id -> unix_timestamp
    k           :int                  =5,
    decay_lambda:float                =0.01,
) -> Tuple[ndarray, ndarray]:
    """
    Retrieves k results, but penalizes the distance based on age.
    """
    logging.info(f'faiss_retrieve_embedding(timestamps={timestamps}, k={k}, decay={decay_lambda})')
    timestamps     :Dict[str,float]        = timestamps if timestamps else {}

    # 1. Search a larger pool than k to allow for re-ranking
    # If we only search k, we might miss a slightly more distant but much newer memory
    distances, indices                     = index.search(np.array([query_vector]).astype('float32'), k * 3)

    now            :float                  = datetime.datetime.now().timestamp()
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

    return np.array(final_scores), np.array(final_indices)

def faiss_update_embedding(index: IndexIDMap, vector: List[float], doc_id: int) -> int:
    """Updates a vector by removing the old ID and re-adding it with new data."""
    logging.info(f'faiss_update_embedding(doc_id={doc_id})')
    # 1. Remove the old vector (IndexIDMap handles this cleanly)
    faiss_delete_embedding(index, doc_id)

    # 2. Re-insert with the same ID
    return faiss_create_embedding(index, vector, doc_id)

#def faiss_delete_embedding(index: faiss.IndexIDMap, doc_id: int) -> int:
#    """Removes a specific ID from the index."""
#    index.remove_ids(np.array([doc_id], dtype='int64'))
#    return doc_id
def faiss_delete_embedding(index: IndexIDMap, doc_id: int) -> int:
    """Removes a specific ID from the index."""
    logging.info(f'faiss_delete_embedding(doc_id={doc_id})')
    # FAISS expects a numpy array of IDs to remove
    ids_to_remove = np.array([doc_id], dtype='int64')
    index.remove_ids(ids_to_remove)
    return doc_id

##
#
##

def sqlite_create_text_db(path:Path, alias:str='sqlite_create_text_db')->None:
    """Initializes the database and ensures the table exists."""
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

#def sqlite_create_text(path:Path, content:str, metadata:Dict[str,Any]|None=None)->int:
#    with sqlite3.connect(path) as conn:
#        cursor = conn.cursor()
#        cursor.execute(
#            "INSERT INTO documents (content, metadata) VALUES (?, ?)",
#            (content, json.dumps(metadata or {}))
#        )
#        return cursor.lastrowid - 1 # SQLite is 1-indexed, FAISS is 0-indexed
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

def sqlite_retrieve_text(path: Path, doc_ids: List[int]) -> List[Dict[str, Any]]:
    """Retrieves multiple documents by their IDs."""
    logging.info(f'sqlite_retrieve_text(path={path}, doc_ids={doc_ids})')
    if not doc_ids:
        logging.warn('no doc ids')
        return []
    # Filter out -1 (FAISS padding)
    valid_ids = [int(i) for i in doc_ids if i != -1]
    placeholders = ', '.join(['?'] * len(valid_ids))
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            f"SELECT id, content, metadata FROM documents WHERE id IN ({placeholders})",
            valid_ids
        )
        return [dict(row) for row in cursor.fetchall()]

def sqlite_update_text(path: Path, doc_id: int, content: str, metadata: Dict[str, Any] | None = None)->None:
    """Updates the content and metadata for a specific rowid."""
    logging.info(f'sqlite_update_text(path={path}, doc_id={doc_id})')
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE documents SET content = ?, metadata = ? WHERE id = ?",
            (content, json.dumps(metadata or {}), doc_id)
        )

#def sqlite_delete_text(path: Path, doc_id: int):
#    with sqlite3.connect(path) as conn:
#        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
def sqlite_delete_text(path: Path, doc_id: int)->None:
    """Removes the record from the database."""
    logging.info(f'sqlite_delete_text(path={path}, doc_id={doc_id})')
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

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
    """Transaction-like: Create in SQL first, then FAISS using the SQL ID."""
    logging.info(f'db_create_document(db_path={db_path})')
    doc_id = sqlite_create_text(db_path, content, metadata)
    faiss_create_embedding(index, vector, doc_id)
    return doc_id

def db_retrieve_document(index: IndexIDMap, db_path: Path, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve vectors, then hydrate with text from SQLite."""
    logging.info(f'db_retrieve_document(db_path={db_path}, k={k})')
    distances, indices = faiss_retrieve_embedding(index, query_vector, k=k)
    return sqlite_retrieve_text(db_path, indices.tolist())

def db_update_document(index: IndexIDMap, db_path: Path, doc_id: int, vector: List[float], content: str, metadata: dict = None):
    """Synchronized update for both SQLite and FAISS."""
    logging.info(f'db_update_document(db_path={db_path}, doc_id={doc_id})')
    sqlite_update_text(db_path, doc_id, content, metadata)
    faiss_update_embedding(index, vector, doc_id)

def db_delete_document(index: IndexIDMap, db_path: Path, doc_id: int):
    """Remove from both systems."""
    logging.info(f'db_delete_document(db_path={db_path}, doc_id={doc_id})')
    faiss_delete_embedding(index, doc_id)
    sqlite_delete_text(db_path, doc_id)

##
#
##

def generate_embedding(text: str, llm: Llama) -> list[float]:
    """Pure function to turn text into a vector."""
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
    """The high-level primitive for 'Remembering' something."""
    logging.info(f'rag_create_document(db_path={db_path})')
    #vector = generate_embedding(content, embed_llm)
    vector = generate_document_embedding(content, embed_llm, prefix=prefix)
    return db_create_document(index, db_path, vector, content, metadata)

def rag_retrieve_document(query: str, embed_llm: Llama, index: IndexIDMap, db_path: Path, k: int = 5, prefix:str='search_query: ') -> List[Dict[str, Any]]:
    """The high-level primitive for 'Searching' memory."""
    logging.info(f'rag_retrieve_document(db_path={db_path}, k={k})')
    #query_vec = generate_embedding(query, embed_llm)
    query_vec = generate_search_embedding(query, embed_llm, prefix)
    return db_retrieve_document(index, db_path, query_vec, k=k)

def rag_update_document(doc_id: int, content: str, embed_llm: Llama, index: IndexIDMap, db_path: Path, metadata: dict = None, prefix:str='search_document: ')->None:
    """The high-level primitive for 'Correcting' a memory."""
    logging.info(f'rag_update_document(doc_id={doc_id}, db_path={db_path})')
    #new_vector = generate_embedding(content, embed_llm)
    new_vector = generate_document_embedding(content, embed_llm, prefix=prefix)
    db_update_document(index, db_path, doc_id, new_vector, content, metadata)

def rag_delete_document(doc_id: int, index: IndexIDMap, db_path: Path)->None:
    """The high-level primitive for 'Forgetting' a memory."""
    logging.info(f'rag_delete_document(doc_id={doc_id}, db_path={db_path})')
    db_delete_document(index, db_path, doc_id)

# TODO data cleaning ?
# TODO file contents -type aware chunking
# TODO data cleaning ?

##
#
##

class ToolCall(BaseModel):
    action: str
    params: dict
    thought: str

#def get_structured_response(prompt: str, llm:Llama, im_start:str='<|im_start|>', im_end:str='<|im_end|>')->str:
#    schema_json = json.dumps(ToolCall.model_json_schema())
#    json_grammar = LlamaGrammar.from_json_schema(schema_json)
#
#    # ChatML format for Qwen
#    chat_prompt = '\n'.join([
#        format_system_message("You are a helpful assistant that strictly outputs JSON.", im_start=im_start, im_end=im_end),
#        format_user_message(prompt, im_start=im_start, im_end=im_end),
#        f"{im_start}assistant"
#    ])
#
#    response = llm(
#        prompt=chat_prompt,
#        grammar=json_grammar,
#        max_tokens=500, # TODO dynamic
#        stop=[im_end, im_start],
#    )
#
#    text = response["choices"][0]["text"]
#    try:
#        return ToolCall.model_validate_json(text)
#    except Exception as e:
#        print(f"Validation Error: {e}\nRaw Text: {text}")
#        raise

def get_structured_response(
        prompt: str,
        llm: Llama,
        im_start: str='<|im_start|>',
        im_end: str='<|im_end|>',
        requested:int=500,
        threshold:int=10,
        # TODO support any type (not just ToolCall) ????
) -> ToolCall:
    logging.info(f'get_structured_response(requested={requested}, threshold={threshold})')

    schema_json = json.dumps(ToolCall.model_json_schema())
    json_grammar = LlamaGrammar.from_json_schema(schema_json)

    chat_prompt = '\n'.join([
        format_system_message("You are a helpful assistant that strictly outputs JSON.", im_start=im_start, im_end=im_end),
        format_user_message(prompt, im_start=im_start, im_end=im_end),
        f"{im_start}assistant"
    ])

    safe_limit = calculate_safe_max_tokens(llm, chat_prompt, requested=requested, threshold=threshold)
    logging.info(f'max_tokens: {safe_limit}')

    response = llm(
        prompt=chat_prompt,
        grammar=json_grammar,
        max_tokens=safe_limit, # No more hardcoded 500
        stop=[im_end, im_start],
    )

    text = response["choices"][0]["text"]
    return ToolCall.model_validate_json(text)

def get_structured_response_but_like_fancier_and_stuff()->...:
    try:
        return get_structured_response(...)
    except ContextFullError as error:
        logging.error(error)
        # handle context full
    return get_structured_response(...)

# TODO chat history RAG
# TODO chat history summarization
# TODO self-narrative generation
# TODO meta prompting & hyper params
# TODO structured planning ?
# TODO reflection / self-correction ?
# TODO generate test cases ==> fine-tuning

def pull_my_llama(model_dir:Path, adapter:str|None=None, embedding:bool=False, n_batch:int=512, verbose:bool=False)->Llama:
    logging.info(f'pull_my_llama(model_dir={model_dir}, adapter={adapter}, embedding={embedding}, n_batch={n_batch}, verbose={verbose})')
    names    :Dict[bool,str] = {
            False: 'qwen3:4b-instruct',
            True : 'nomic-embed-text',
    }
    repo_ids :Dict[str,str]  = {
            'nomic-embed-text' : "nomic-ai/nomic-embed-text-v1.5-GGUF",
            'qwen3:4b-instruct': "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    }
    filenames:Dict[str,str]  = {
            'nomic-embed-text' : "nomic-embed-text-v1.5.Q4_K_M.gguf",
            'qwen3:4b-instruct': "Qwen_Qwen3-4B-Instruct-2507-Q6_K.gguf",
    }
    name     :str            = names    [embedding]
    repo_id  :str            = repo_ids [name]
    filename :str            = filenames[name]
    lora     :str |None      = f'{name}.{adapter}.gguf' if adapter else None
    repo_dir :Path           = model_dir / repo_id
    path     :Path           = repo_dir  / filename
    lora_path:Path|None      = repo_dir  / lora         if lora    else None
    logging.info(f'name     : {name}')
    logging.info(f'repo_id  : {repo_id}')
    logging.info(f'filename : {filename}')
    logging.info(f'repo_dir : {repo_dir}')
    logging.info(f'path     : {path}')
    logging.info(f'lora_path: {lora_path}')
    return pull_llama(repo_id=repo_id, model_path=path, lora_path=lora_path, embedding=embedding, n_batch=n_batch, verbose=verbose)

def main()->None:
    name            :str           = 'test-agent-1'
    setup_logging(name)
    root            :Path          = Path('/', 'var', 'lib', 'ia_llm') #Path(os.getcwd()).resolve()
    model_dir       :Path          = root      / '.models'
    verbose         :bool          = False
    adapter_llm     :str  |None    = None
    adapter_embed   :str  |None    = None
    n_batch         :int           = 512 # TODO can dynamically determine ?
    llm             :Llama         = pull_my_llama(model_dir, adapter=adapter_llm,   embedding=False, n_batch=n_batch, verbose=verbose)
    embed           :Llama         = pull_my_llama(model_dir, adapter=adapter_embed, embedding=True,  n_batch=n_batch, verbose=verbose)
    im_start        :str           = '<|im_start|>' # TODO can dynamically determine ?
    im_end          :str           = '<|im_end|>' # TODO can dynamically determine ?
    dimensions      :int           = embed.n_embd()
    db_path         :Path          = root / "vault.db"
    index_path      :Path          = root / "vault.index"
    logging.info(f'root         : {root}')
    logging.info(f'model_dir    : {model_dir}')
    logging.info(f'verbose      : {verbose}')
    logging.info(f'n_batch      : {n_batch}')
    logging.info(f'adapter_llm  : {adapter_llm}')
    logging.info(f'adapter_embed: {adapter_embed}')
    logging.info(f'im_start     : {im_start}')
    logging.info(f'im_end       : {im_end}')
    logging.info(f'dimensions   : {dimensions}')
    logging.info(f'db_path      : {db_path}')
    logging.info(f'index_path   : {index_path}')

    # Load
    sqlite_load_or_create_text_db(db_path)
    index           :IndexIDMap    = faiss_load_or_create_embedding_db(index_path, dimension=dimensions) # Nomic dim

    # usage
    #result = get_structured_response("Should I update the debian package?", llm=llm, im_start=im_start, im_end=im_end)
    #print(result.thought)
    rag_create_document("The user prefers 'apt' over 'nala'.", embed, index, db_path)

    # Search something
    memories = rag_retrieve_document("What package manager does the user like?", embed, index, db_path)

    # Inject into Qwen
    context = "\n".join([m['content'] for m in memories])
    prompt          :str           = f"Context: {context}\n\nUser: Should I use nala?"
    logging.info(f'User           : {prompt}')

    response        :ToolCall      = get_structured_response(prompt, llm=llm)
    logging.info(f'Agent (thought): {response.thought}')
    logging.info(f'Agent (action) : {response.action}')
    logging.info(f'Agent (params) : {response.params}')

if __name__ == '__main__':
    main()

def _finetune_cmd(
    model_base  :Path, 
    train_data  :Path, 
    lora_out    :Path, 
    threads     :int|None=None,
    adam_iter   :int     =256,
    batch       :int     =4,
    sample_start:str     ='<s>',
) -> List[str]:
    llama_finetune:str|None = shutil.which('llama-finetune')
    assert llama_finetune
    logging.info(f'llama finetune: {llama_finetune}')
    threads       :int      = threads or multiprocessing.cpu_count()
    checkpoint_in :Path     = lora_out.with_suffix('.chkpt')
    return [
        llama-finetune, # Path to your compiled binary
        "--model-base", str(model_base),
        "--train-data", str(train_data),
        "--lora-out", str(lora_out),
        "--threads", str(threads),
        "--adam-iter", str(adam_iter),        # Standard for small LoRAs
        "--batch", str(batch),             # Small batch size for CPU
        "--sample-start", sample_start,     # The "magic" delimiter
        "--checkpoint-in", str(checkpoint_in), # For resuming
    ]

def format_exchange_for_training(
        system_prompt:str,
        user_msg     :str,
        thought      :str,
        assistant_msg:str,
        sample_start :str='<s>',
        im_start     :str='<|im_start|>',
        im_end       :str='<|im_end|>',
        think        :str='<|thought|>',
)->str:
    # NOTE Keep your samples roughly the same length in your train.txt to prevent the loss from oscillating wildly.
    return (
        sample_start +
        f"{im_start}system\n{system_prompt}{im_end}\n"
        f"{im_start}user\n{user_msg}{im_end}\n"
        f"{im_start}assistant\n{think}\n{thought}\n{think}\n{assistant_msg}{im_end}\n" +
        sample_start
    )

