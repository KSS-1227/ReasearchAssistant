"""
Base Agent Class for Research Assistant System
CSYE 7374 Final Project - Summer 2025

Provides common interface and functionality for all agents.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system
    
    Provides common functionality for performance tracking,
    error handling, and standardized interfaces.
    """
    
    def __init__(self, name: str, uses_llm: bool = False):
        """
        Initialize base agent
        
        Args:
            name: Human-readable agent name
            uses_llm: Whether this agent makes LLM calls
        """
        self.name = name
        self.uses_llm = uses_llm
        self.calls_made = 0
        self.successful_calls = 0
        self.total_processing_time = 0.0
        self.last_execution_time = None
        self.created_at = datetime.now().isoformat()
        
        logger.info("%s initialized (uses_llm=%s)", self.name, self.uses_llm)
    
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Main processing method - must be implemented by each agent
        
        Args:
            input_data: Input data specific to the agent's function
            
        Returns:
            Dict containing processed results and metadata
        """
        pass
    
    def _execute_with_tracking(self, func, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute a function with automatic performance tracking
        
        Wraps the actual processing logic with timing and error handling.
        """
        start_time = time.time()
        self.calls_made += 1
        
        try:
            logger.debug("%s processing...", self.name)
            
            # Execute the actual processing function
            result = func(*args, **kwargs)
            
            # Track successful execution
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.successful_calls += 1
            self.total_processing_time += processing_time
            self.last_execution_time = datetime.now().isoformat()
            
            logger.info("%s completed in %.2fs", self.name, processing_time)
            
            # Add metadata to result
            if isinstance(result, dict):
                result["agent_metadata"] = {
                    "agent_name": self.name,
                    "processing_time": processing_time,
                    "success": True,
                    "timestamp": self.last_execution_time
                }
            
            return result
            
        except Exception as e:
            # Track failed execution
            end_time = time.time()
            processing_time = end_time - start_time
            self.total_processing_time += processing_time
            
            logger.error("%s failed after %.2fs: %s", self.name, processing_time, e)
            
            return {
                "success": False,
                "error": str(e),
                "agent_metadata": {
                    "agent_name": self.name,
                    "processing_time": processing_time,
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for this agent"""
        
        success_rate = self.successful_calls / max(1, self.calls_made)
        avg_processing_time = self.total_processing_time / max(1, self.calls_made)
        
        return {
            "agent_name": self.name,
            "uses_llm": self.uses_llm,
            "total_calls": self.calls_made,
            "successful_calls": self.successful_calls,
            "success_rate": round(success_rate, 3),
            "total_processing_time": round(self.total_processing_time, 2),
            "average_processing_time": round(avg_processing_time, 3),
            "last_execution": self.last_execution_time,
            "created_at": self.created_at
        }
    
    def reset_metrics(self):
        """Reset performance metrics - useful for testing"""
        self.calls_made = 0
        self.successful_calls = 0
        self.total_processing_time = 0.0
        self.last_execution_time = None
        logger.debug("%s metrics reset", self.name)
    
    def validate_input(self, input_data: Any) -> Dict[str, Any]:
        """
        Validate input data - can be overridden by specific agents
        
        Returns:
            Dict with 'valid' boolean and optional 'error' message
        """
        if input_data is None:
            return {"valid": False, "error": f"{self.name} received None input"}
        
        return {"valid": True}
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Standardized error handling for all agents
        
        Args:
            error: The exception that occurred
            context: Additional context about when the error occurred
            
        Returns:
            Standardized error response
        """
        error_msg = f"{self.name} error"
        if context:
            error_msg += f" during {context}"
        error_msg += f": {str(error)}"
        
        return {
            "success": False,
            "error": error_msg,
            "agent": self.name,
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# Agent Registry
# ============================================================================

class AgentRegistry:
    """
    Central registry for all agents in the system
    
    Provides agent discovery, status monitoring, and coordination support.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_dependencies: Dict[str, List[str]] = {}
    
    def register_agent(self, agent: BaseAgent, dependencies: List[str] = None):
        """
        Register an agent with the system
        
        Args:
            agent: Agent instance to register
            dependencies: List of agent names this agent depends on
        """
        self.agents[agent.name] = agent
        self.agent_dependencies[agent.name] = dependencies or []
        
        logger.info("Registered agent: %s", agent.name)
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self.agents.get(name)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all registered agents"""
        return self.agents.copy()
    
    def get_llm_agents(self) -> Dict[str, BaseAgent]:
        """Get only agents that use LLMs"""
        return {name: agent for name, agent in self.agents.items() if agent.uses_llm}
    
    def get_deterministic_agents(self) -> Dict[str, BaseAgent]:
        """Get only deterministic (non-LLM) agents"""
        return {name: agent for name, agent in self.agents.items() if not agent.uses_llm}
    
    def validate_architecture(self) -> Dict[str, Any]:
        """
        Validate that the agent architecture meets project requirements
        
        Returns:
            Validation results with pass/fail status
        """
        total_agents = len(self.agents)
        llm_agents = len(self.get_llm_agents())
        deterministic_agents = len(self.get_deterministic_agents())
        
        # Check requirements
        min_agents_met = total_agents >= 3
        deterministic_percentage = (deterministic_agents / total_agents) * 100 if total_agents > 0 else 0
        deterministic_requirement_met = deterministic_percentage >= 50
        max_llm_agents_met = llm_agents <= 2
        
        return {
            "total_agents": total_agents,
            "llm_agents": llm_agents,
            "deterministic_agents": deterministic_agents,
            "deterministic_percentage": round(deterministic_percentage, 1),
            "requirements": {
                "min_3_agents": min_agents_met,
                "50_percent_deterministic": deterministic_requirement_met,
                "max_2_llm_agents": max_llm_agents_met
            },
            "overall_pass": min_agents_met and deterministic_requirement_met and max_llm_agents_met,
            "summary": f"{deterministic_agents}/{total_agents} agents are deterministic ({deterministic_percentage:.1f}%)"
        }
    
    def get_execution_order(self) -> List[str]:
        """
        Get recommended execution order based on dependencies
        
        Returns:
            List of agent names in dependency-resolved order
        """
        # Simple topological sort for agent dependencies
        executed = set()
        execution_order = []
        
        def can_execute(agent_name: str) -> bool:
            dependencies = self.agent_dependencies.get(agent_name, [])
            return all(dep in executed for dep in dependencies)
        
        # Keep trying to find agents we can execute
        while len(execution_order) < len(self.agents):
            progress_made = False
            
            for agent_name in self.agents:
                if agent_name not in executed and can_execute(agent_name):
                    execution_order.append(agent_name)
                    executed.add(agent_name)
                    progress_made = True
            
            if not progress_made:
                # Circular dependency or other issue
                remaining = set(self.agents.keys()) - executed
                execution_order.extend(list(remaining))
                break
        
        return execution_order
    
    def generate_architecture_report(self) -> str:
        """Generate human-readable architecture report"""
        
        validation = self.validate_architecture()
        
        report = f"""
🏗️  AGENT ARCHITECTURE REPORT
{'='*40}

Total Agents: {validation['total_agents']}
LLM-Powered Agents: {validation['llm_agents']}
Deterministic Agents: {validation['deterministic_agents']}
Deterministic Percentage: {validation['deterministic_percentage']}%

REQUIREMENT COMPLIANCE:
✅ Minimum 3 agents: {validation['requirements']['min_3_agents']}
✅ ≥50% deterministic: {validation['requirements']['50_percent_deterministic']}
✅ ≤2 LLM agents: {validation['requirements']['max_2_llm_agents']}

OVERALL STATUS: {'✅ PASS' if validation['overall_pass'] else '❌ FAIL'}

AGENT DETAILS:
"""
        
        for name, agent in self.agents.items():
            status = "🤖 LLM-Powered" if agent.uses_llm else "🔧 Deterministic"
            metrics = agent.get_performance_metrics()
            report += f"  • {name}: {status} ({metrics['total_calls']} calls made)\n"
        
        return report