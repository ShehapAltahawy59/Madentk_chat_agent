from autogen_agentchat.agents import AssistantAgent
from .client import az_model_client
from .prompt import system_message
from . import tools as tools_module

# Define tools list that can be reused
AGENT_TOOLS = [
    tools_module.insert_order,
    tools_module.get_user_by_id,
    tools_module.get_restaurant_by_id,
    tools_module.get_item_by_id,
    tools_module.get_item_by_name,
    tools_module.get_items_in_restaurant,
    tools_module.search_restaurant_by_name,
    tools_module.get_active_user_id,
    tools_module.recommend_time_based_suggestions,
    tools_module.search_semantic,
    tools_module.get_delivery_cost,
    tools_module.add_user_preference,
    tools_module.add_order_to_memory,
]

# Create agent without memory - we'll add user-specific memory dynamically
smart_agent = AssistantAgent(
    name="madentk_agent",
    model_client=az_model_client,
    system_message=system_message,
    tools=AGENT_TOOLS,
    reflect_on_tool_use=True,
    max_tool_iterations=5
) 
