"""OpenTelemetry names used by the agent.

Standard GenAI names come from the OpenTelemetry GenAI semantic conventions
(as cited in the source article). The custom names are the ones the article
says OpenTelemetry does not provide and must be added by the application.
"""

# Standard GenAI attributes (use these whenever they exist).
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

# Standard span names / operation names.
OP_INVOKE_AGENT = "invoke_agent"
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"

# Custom attributes (no standard equivalent, per the article).
AGENT_TERMINAL_STATE = "agent.terminal_state"
AGENT_STEPS_TAKEN = "agent.steps_taken"
AGENT_TASK_CLASS = "agent.task_class"
AGENT_HUMAN_INTERVENTION = "agent.human_intervention"
AGENT_TOOL_REJECTION_REASON = "agent.tool_rejection_reason"

# Custom spans for working memory (the article's memory.read / memory.write).
MEMORY_READ_SPAN = "memory.read"
MEMORY_WRITE_SPAN = "memory.write"
MEMORY_OBSERVED_VERSION = "memory.observed_version"
MEMORY_LATEST_ACKED_VERSION = "memory.latest_acked_version"
