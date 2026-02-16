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
    assert (react_retries >= 0)
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
    if (not allow_none) and (len(steps) > 1):
        logging.warning(f'multiple results: {len(steps)}')
    if (not allow_none) and (not len(steps)):
        # TODO raise ?
        ...
    return ReActResult(steps=steps, conversation=checkpoint)

class Critique(BaseModel):
    is_sufficient       : bool
    score               : int  # 1-10
    reasoning           : str
    missing_information : List[str]
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
    turn      :ConversationTurn = conversation._current_turn(user_prompt=None) != ConversationTurn.ASSISTANT:
    if turn != ConversationTurn.ASSISTANT:
        logging.warning(f'critiquing non-agent: {turn}')
    checkpoint:Conversation     = conversation.deepcopy()
    checkpoint.instructions     = [ # NOTE effective system prompt is [checkpoint.system_prompt] + checkpoint.instructions
        "You are a strict Quality Assurance Auditor.",
        "Your goal is to find flaws, missing details, or logical fallacies in the agent's response.",
        "Be pedantic. If the goal is not 100% met, 'is_sufficient' must be false."
    ]
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
        critique_retries:int|None=None,
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
        f'critique_retries={critique_retries}, '
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
    assert (critique_retries >= 0)
    checkpoint                :Conversation       = conversation.deepcopy()
    attempt                   :int                = 0
    while (critique_retries is None) or (attempt < critique_retries + 1): # TODO include attempt / retries in context ?
        logging.info(f'critique attempt: {attempt} / {critique_retries}')
        attempt                                  += 1
        result                :ReActResult        = get_llm_response_react( # NOTE may have 0 or more results
                llm           =llm,
                conversation  =checkpoint,
                funcs         =funcs,
                user_prompt   =user_prompt,
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
        # NOTE user_prompt is None when the calling code already added it (or purposefully neglected to add it)
        # 3. FEEDBACK: If it fails, we inject the 'suggested_correction'
        # into the goal for the next attempt.
        goal = f"Previous attempt failed. {critique.suggested_correction}. Try again: {goal}"
    assert (attempt == critique_retries + 1)
    if raise_on_error:
        # TODO raise
        ...
    # TODO no result

