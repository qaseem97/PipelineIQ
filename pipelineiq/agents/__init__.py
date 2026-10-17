from .phase_agents import DesignAgent, DevelopmentAgent, RequirementsAgent, TestingAgent

PHASE_ORDER = ["requirements", "design", "development", "testing"]
AGENTS = {
    "requirements": RequirementsAgent,
    "design": DesignAgent,
    "development": DevelopmentAgent,
    "testing": TestingAgent,
}

__all__ = ["AGENTS", "PHASE_ORDER", "RequirementsAgent", "DesignAgent", "DevelopmentAgent", "TestingAgent"]
