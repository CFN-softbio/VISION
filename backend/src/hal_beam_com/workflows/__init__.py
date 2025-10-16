# Workflows the Cog-Manager may call
from .command_workflow               import run as command_workflow
from .add_context_functions_workflow import run as add_context_functions_workflow
from .chatbot_workflow               import run as chatbot_workflow
from .simulation_workflow            import run as simulation_workflow
from .simulation_workflow            import run as simulate_abort_workflow   # NEW alias
from .evaluation_workflow            import run as evaluation_workflow
