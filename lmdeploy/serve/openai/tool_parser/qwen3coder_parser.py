# Copyright (c) OpenMMLab. All rights reserved.
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import shortuuid

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from lmdeploy.utils import get_logger

from .tool_parser import ToolParser, ToolParserManager

logger = get_logger('lmdeploy')


def _escape_json_string_inner(raw: str) -> str:
    """Escape text for use inside a JSON string literal (no outer quotes)."""
    out: list[str] = []
    for ch in raw:
        if ch == '\\':
            out.append('\\\\')
        elif ch == '"':
            out.append('\\"')
        elif ch == '\n':
            out.append('\\n')
        elif ch == '\r':
            out.append('\\r')
        elif ch == '\t':
            out.append('\\t')
        elif ord(ch) < 0x20:
            out.append(f'\\u{ord(ch):04x}')
        else:
            out.append(ch)
    return ''.join(out)


def _coerce_function_arguments_to_mapping(arguments: Any) -> dict[str, Any]:
    """Return a mapping for HuggingFace chat templates.

    Qwen3-Coder templates iterate ``tool_call.arguments|items``; OpenAI clients
    usually send ``arguments`` as a JSON string, which breaks Jinja unless
    coerced to a dict.
    """
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, list):
        return {str(i): v for i, v in enumerate(arguments)}
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            logger.debug('tool call arguments are not valid JSON; using empty dict for chat template')
            return {}
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {str(i): v for i, v in enumerate(parsed)}
        return {'value': parsed}
    return {'value': arguments}


@dataclass
class ParserState:
    """Maintains the state of parsing during tool call extraction."""
    position: int = 0  # Current position in the text being parsed
    current_index: int = -1  # Index of the current tool call

    id: str = ''  # ID of the current tool call
    # Cumulative ``arguments`` JSON prefix already sent to the client (OpenAI concat semantics).
    arguments_buffer: str = ''
    # True while an incomplete ``<tool_call>`` block is being streamed (no closing tag yet).
    inside_tool_call: bool = False

    def reset_tool_call(self):
        """Called when `</tool_call>` finish tag occurred."""
        self.id = ''
        self.arguments_buffer = ''
        self.inside_tool_call = False


@ToolParserManager.register_module(['qwen3coder'])
class Qwen3CoderToolParser(ToolParser):
    """Parser for Qwen3 Coder model's tool call format.

    Handles the extraction of tool calls from Qwen3 Coder's output format, which uses purely XML tags for function names
    and parameters, e.g., <tool_call> <function=func_name> <parameter=arg_name>arg_value</parameter> </function>
    </tool_call>
    """

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.tool_start_token = '<tool_call>'
        self.tool_end_token = '</tool_call>'
        self.func_prefix = '<function='
        self.func_end_token = '</function>'
        self.param_prefix = '<parameter='
        self.param_end_token = '</parameter>'

        self.tool_call_pat = re.compile(r'\n*<tool_call>(.*?)</tool_call>', re.DOTALL)

    def _normalize_request_messages(self, messages: list[dict]) -> list[dict] | None:
        """Return a render-safe copy of request messages when needed."""
        normalized_messages = None

        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                continue
            tool_calls = message.get('tool_calls')
            if not isinstance(tool_calls, list):
                continue

            normalized_tool_calls = None
            for tool_idx, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get('function')
                if not isinstance(function, dict):
                    continue
                if isinstance(function.get('arguments'), dict):
                    continue

                coerced = _coerce_function_arguments_to_mapping(function.get('arguments'))

                if normalized_messages is None:
                    normalized_messages = list(messages)
                if normalized_tool_calls is None:
                    normalized_tool_calls = list(tool_calls)
                    normalized_message = dict(normalized_messages[msg_idx])
                    normalized_message['tool_calls'] = normalized_tool_calls
                    normalized_messages[msg_idx] = normalized_message

                normalized_function = dict(function)
                normalized_function['arguments'] = coerced

                normalized_tool_call = dict(normalized_tool_calls[tool_idx])
                normalized_tool_call['function'] = normalized_function
                normalized_tool_calls[tool_idx] = normalized_tool_call

        return normalized_messages

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        messages = request.messages
        if not isinstance(messages, list):
            return request

        normalized_messages = self._normalize_request_messages(messages)
        if normalized_messages is None:
            return request
        return request.model_copy(update={'messages': normalized_messages})

    def _find_tool_block_end(self, parsing_content: str, start_idx: int) -> int | None:
        """Return the index of ``</tool_call>`` only after ``</function>``.

        Parameter values may contain ``</tool_call>`` as plain text; requiring
        ``</function>`` first avoids closing the block too early.
        """
        search_from = start_idx + len(self.tool_start_token)
        func_end = parsing_content.find(self.func_end_token, search_from)
        if func_end == -1:
            return None
        end_idx = parsing_content.find(self.tool_end_token, func_end + len(self.func_end_token))
        if end_idx == -1:
            return None
        return end_idx

    def _split(self, parser_state: ParserState, parsing_content: str) -> tuple[str, str, bool]:
        """Split content into tuple: (text_content, tool_content, has_tool_end)"""
        try:
            start_idx = parsing_content.index(self.tool_start_token)
            parser_state.position += start_idx
        except ValueError:
            parser_state.position += len(parsing_content)
            return parsing_content, '', False

        end_idx = self._find_tool_block_end(parsing_content, start_idx)
        if end_idx is None:
            parser_state.inside_tool_call = True
            return parsing_content[:start_idx], parsing_content[start_idx:], False

        rem = end_idx - start_idx
        parser_state.position += rem + len(self.tool_end_token)
        parser_state.inside_tool_call = False
        return parsing_content[:start_idx], parsing_content[start_idx:end_idx + len(self.tool_end_token)], True

    def _extract_params(self, content: str) -> tuple[str | None, dict[str, Any], bool]:
        """Parse XML tool content into components."""
        content = content.replace(self.tool_start_token, '').replace(self.tool_end_token, '').strip()

        func_name = None
        func_start = content.find(self.func_prefix)
        if func_start != -1:
            name_start = func_start + len(self.func_prefix)
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if terminators:
                func_name = content[name_start:min(terminators)].strip()

        args_dict = {}
        search_idx = 0
        while True:
            param_start = content.find(self.param_prefix, search_idx)
            if param_start == -1:
                break

            name_start = param_start + len(self.param_prefix)
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if not terminators:
                break

            name_end = min(terminators)
            param_name = content[name_start:name_end].strip()

            val_start = name_end + 1
            val_end = content.find(self.param_end_token, val_start)
            if val_end == -1:
                break

            param_val_str = content[val_start:val_end].strip()

            if param_val_str.lower() == 'null':
                val = None
            elif param_val_str.lower() == 'true':
                val = True
            elif param_val_str.lower() == 'false':
                val = False
            else:
                try:
                    val = json.loads(param_val_str)
                except json.JSONDecodeError:
                    val = param_val_str
            args_dict[param_name] = val
            search_idx = val_end + len(self.param_end_token)

        is_func_closed = self.func_end_token in content
        return func_name, args_dict, is_func_closed

    def _inner_from_tool_block(self, tool_content: str) -> str:
        return tool_content.replace(self.tool_start_token, '').replace(self.tool_end_token, '').strip()

    def _parse_value_string(self, param_val_str: str) -> Any:
        if param_val_str.lower() == 'null':
            return None
        if param_val_str.lower() == 'true':
            return True
        if param_val_str.lower() == 'false':
            return False
        try:
            return json.loads(param_val_str)
        except json.JSONDecodeError:
            return param_val_str

    def _parse_progressive_params(
            self, tool_content: str) -> tuple[str | None, dict[str, Any], tuple[str, str] | None, bool]:
        """Parse closed parameters and optionally one open streaming parameter."""
        content = self._inner_from_tool_block(tool_content)

        func_name = None
        func_start = content.find(self.func_prefix)
        if func_start != -1:
            name_start = func_start + len(self.func_prefix)
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if terminators:
                func_name = content[name_start:min(terminators)].strip()

        args_complete: dict[str, Any] = {}
        stream: tuple[str, str] | None = None
        search_idx = 0
        while True:
            param_start = content.find(self.param_prefix, search_idx)
            if param_start == -1:
                break

            name_start = param_start + len(self.param_prefix)
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if not terminators:
                break

            name_end = min(terminators)
            param_name = content[name_start:name_end].strip()

            val_start = name_end + 1
            val_end = content.find(self.param_end_token, val_start)
            if val_end == -1:
                stream = (param_name, content[val_start:])
                break

            param_val_str = content[val_start:val_end].strip()
            args_complete[param_name] = self._parse_value_string(param_val_str)
            search_idx = val_end + len(self.param_end_token)

        is_func_closed = self.func_end_token in content
        return func_name, args_complete, stream, is_func_closed

    def _arguments_json_target(self, complete: dict[str, Any], stream: tuple[str, str] | None,
                               is_func_closed: bool) -> str:
        """Build the canonical OpenAI ``arguments`` string prefix for the current parse state.

        Extensions are strictly suffix-only so streamed deltas concatenate to valid final JSON.
        """
        if not complete and stream is None and not is_func_closed:
            return ''
        parts: list[str] = ['{']
        sep = ''
        for k, v in complete.items():
            parts.append(sep)
            parts.append(json.dumps(k, ensure_ascii=False))
            parts.append(':')
            parts.append(json.dumps(v, ensure_ascii=False))
            sep = ','
        if stream is not None:
            sn, sv = stream
            parts.append(sep)
            parts.append(json.dumps(sn, ensure_ascii=False))
            parts.append(':"')
            parts.append(_escape_json_string_inner(sv))
        if is_func_closed:
            if stream is not None:
                parts.append('"')
            parts.append('}')
        return ''.join(parts)

    def _emit_tool_delta(self, parser_state: ParserState, fcall_delta: DeltaFunctionCall) -> DeltaToolCall:
        if not parser_state.id:
            parser_state.id = f'chatcmpl-tool-{shortuuid.random()}'
            parser_state.current_index += 1
        return DeltaToolCall(
            id=parser_state.id,
            index=parser_state.current_index,
            function=fcall_delta,
        )

    def _process_tool_block(
        self,
        parser_state: ParserState,
        tool_content: str,
        has_tool_end: bool,
    ) -> DeltaToolCall | None:
        func_name, complete, stream_pair, is_func_closed = self._parse_progressive_params(tool_content)

        fcall_delta = DeltaFunctionCall()
        has_updates = False

        if func_name and not getattr(parser_state, 'has_emitted_name', False):
            fcall_delta.name = func_name
            parser_state.has_emitted_name = True
            has_updates = True

        target = self._arguments_json_target(complete, stream_pair, is_func_closed)
        buf = parser_state.arguments_buffer
        if not target.startswith(buf):
            logger.warning('tool arguments snapshot regressed; resyncing stream buffer')
            buf = ''
        suffix = target[len(buf):]
        parser_state.arguments_buffer = buf + suffix
        if suffix:
            fcall_delta.arguments = suffix
            has_updates = True

        if has_tool_end and not has_updates:
            if not getattr(parser_state, 'has_emitted_name', False) and not parser_state.arguments_buffer:
                parser_state.reset_tool_call()
                if hasattr(parser_state, 'has_emitted_name'):
                    delattr(parser_state, 'has_emitted_name')
            return None

        if not has_updates:
            return None

        tool_delta = self._emit_tool_delta(parser_state, fcall_delta)

        if has_tool_end:
            parser_state.reset_tool_call()
            if hasattr(parser_state, 'has_emitted_name'):
                delattr(parser_state, 'has_emitted_name')

        return tool_delta

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:

        parser_state = getattr(request, '_tool_parser_state', None)
        if parser_state is None:
            parser_state = ParserState()
            setattr(request, '_tool_parser_state', parser_state)

        delta = DeltaMessage()
        text_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []

        while parser_state.position < len(current_text):
            split_result = self._split(parser_state, current_text[parser_state.position:])
            text_content, tool_content, has_tool_end = split_result

            if text_content:
                text_parts.append(text_content)

            if not tool_content:
                break

            if not parser_state.id:
                parser_state.arguments_buffer = ''
                parser_state.has_emitted_name = False

            tool_delta = self._process_tool_block(parser_state, tool_content, has_tool_end)
            if tool_delta is not None:
                tool_deltas.append(tool_delta)

            if not has_tool_end:
                break

        if text_parts:
            delta.content = ''.join(text_parts)
        if tool_deltas:
            delta.tool_calls = tool_deltas

        has_any = (delta.content is not None) or (len(delta.tool_calls) > 0)
        if not has_any:
            return None
        return delta

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        text = model_output
        buf = []
        scan_pos = 0
        tool_calls = []

        for idx, match in enumerate(self.tool_call_pat.finditer(text)):
            buf.append(text[scan_pos:match.start()])
            scan_pos = match.end()

            tool_content = match.group(1)
            func_name, args_dict, _ = self._extract_params(tool_content)

            if func_name and args_dict:
                tool_calls.append(
                    ToolCall(function=FunctionCall(
                        name=func_name, arguments=json.dumps(args_dict, ensure_ascii=False))))

        if scan_pos < len(text):
            buf.append(text[scan_pos:])

        text = ''.join(buf)

        return ExtractedToolCallInformation(
            content=text,
            tool_calls=tool_calls,
            tools_called=bool(tool_calls),
        )
