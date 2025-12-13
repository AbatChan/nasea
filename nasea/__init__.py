"""
NASEA - Natural-Language Autonomous Software-Engineering Agent

A self-generating software engine that creates complete applications
from natural language prompts.

Version: 0.2.0-alpha (Enhanced UX)
"""

__version__ = "0.5.7-alpha"
__author__ = "Alex & Abat"
__license__ = "Proprietary"

from nasea.core.control_unit import ControlUnit
from nasea.agents.manager_agent import ManagerAgent
from nasea.agents.developer_agent import DeveloperAgent
from nasea.agents.verifier_agent import VerifierAgent

__all__ = [
    "ControlUnit",
    "ManagerAgent",
    "DeveloperAgent",
    "VerifierAgent",
]
