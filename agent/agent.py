from autogen_agentchat.agents import AssistantAgent
from .client import az_model_client
from .prompt import system_message
from . import tools as tools_module

smart_agent = AssistantAgent(
    name="smart_food_agent",
    model_client=az_model_client,
    system_message=system_message,
    tools=[
        tools_module.insert_order,
        tools_module.get_user_by_id,
        tools_module.search_semantic,
        tools_module.get_restaurant_by_id,
        tools_module.get_item_by_id,
        tools_module.get_item_by_name,
        tools_module.get_items_in_restaurant,
        tools_module.search_restaurant_by_name,
        tools_module.get_active_user_id,
    ],
    reflect_on_tool_use=True,
    max_tool_iterations=5
) 
