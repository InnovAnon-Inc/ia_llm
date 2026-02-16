class Tool(BaseModel):
    name       : str
    description: str|None
    parameters : Dict[str, Any] # TODO JSON
    # TODO return type ?

    def format(self)->str:
        params:str = json.dumps(self.parameters)
        desc  :str = f': {self.description}' if self.description else ''
        return f'{self.name}{desc}. Params: {params}'

class ToolCall(BaseModel):
    thought    : str
    action     : str
    #params     : Dict[str,str]
    params     : Dict[str,Any]

def get_llm_response_tool_call(
        llm         :Llama,
        conversation:Conversation,
        tools       :List[Tool],
        user_prompt :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int=500,
        threshold   :int=10,
        encoding    :str='utf-8',
)->ToolCall:
    logging.info(f'get_llm_response_tool_call(n_tools={len(tools)}, im_start={im_start}, im_end={im_end}, requested={requested}, threshold={threshold}, encoding={encoding})')
    descriptions :List[str]    = [tool.format() for tool in tools]
    descriptions               = [f'- {d}' for d in descriptions]
    descriptions.insert(0, 'Available Tools:')
    description  :str          = '\n'.join(descriptions)
    _conversation:Conversation = deepcopy(conversation)
    _conversation.instructions.append(description)
    return get_llm_response_pydantic(
            llm         =llm,
            conversation=_conversation,
            output_type =ToolCall,
            user_prompt =user_prompt,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            encoding    =encoding,)

def format_function_parameter(name:str, param)->Dict[str,str]: # TODO typehint
    param:Dict[str,str]  = {}
    if param.annotation != inspect._empty:
        param['type']    = getattr(param.annotation, '__name__', str(param.annotation))
    if param.default != inspect._empty:
        param['default'] = param.default                 
    return param

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
    name  :str           = f'{func.__name__}_Args' # TODO do we need to worry about name collisions, or is this completely arbitrary ?
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

def get_llm_response_function_call(
        llm         :Llama,
        conversation:Conversation,
        funcs       :List[Callable[[...],Any]],
        user_prompt :str|None=None,
        im_start    :str|None=None,
        im_end      :str|None=None,
        requested   :int=500,
        threshold   :int=10,
        encoding    :str='utf-8',
)->Any:
    logging.info(f'get_llm_response_function_call(n_func={len(funcs)}, im_start={im_start}, im_end={im_end}, requested={requested}, threshold={threshold}, encoding={encoding})')
    names    :Dict[str,Callable]       = {f.__name__: f for f in funcs}
    tools    :List[Tool]               = list(map(function_to_tool, func))
    tool_call:ToolCall                 = get_llm_response_tool_call(
            llm         =llm,
            conversation=_conversation,
            tools       =tools,
            user_prompt =user_prompt,
            im_start    =im_start,
            im_end      =im_end,
            requested   =requested,
            threshold   =threshold,
            encoding    =encoding,)
    func     :Callable[[...],Any]|None = names.get(tool_call.action)
    if not func:
        raise ValueError(f'LLM tried to call unknown function: {tool_call.action}')
    Model    :Type[BaseModel]          = create_function_signature_model(func)
    model    :BaseModel                = Model(**tool_call.params)
    params   :Dict[str,Any]            = model.model_dump() # TODO JSON
    if not inspect.iscoroutinefunction(func):
        return func(**params)
    return asyncio.run(func(**params)

