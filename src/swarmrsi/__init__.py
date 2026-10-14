"""SwarmRSI research search, agentic coding, and evidence-bound learning."""

__version__ = "0.3.0.dev1"

from .contracts import (
    ActionCandidate as ActionCandidate,
    ActionOperator as ActionOperator,
    PolicyGenome as PolicyGenome,
    PolicyTrial as PolicyTrial,
    ResearchDependency as ResearchDependency,
    ScientificFamily as ScientificFamily,
    SearchBudget as SearchBudget,
    SearchDecision as SearchDecision,
    SearchObservation as SearchObservation,
    SearchSnapshot as SearchSnapshot,
)
from .strategies import (
    BoundedRSIStrategy as BoundedRSIStrategy,
    EvidenceCoupledStrategy as EvidenceCoupledStrategy,
    HeterogeneousMultiIslandStrategy as HeterogeneousMultiIslandStrategy,
    HomogeneousIslandStrategy as HomogeneousIslandStrategy,
    HypothesisProgramTreeStrategy as HypothesisProgramTreeStrategy,
    MultiParentGraphStrategy as MultiParentGraphStrategy,
    ProposalDirective as ProposalDirective,
)
from .state import (
    DynamicResearchView as DynamicResearchView,
    SearchEvent as SearchEvent,
    assert_acyclic_actions as assert_acyclic_actions,
)
from .resources import (
    CLAUDE_CODE_DSV4PRO_S43588 as CLAUDE_CODE_DSV4PRO_S43588,
    ResourceEnvelope as ResourceEnvelope,
    ResourceLedger as ResourceLedger,
    ResourceLimitReached as ResourceLimitReached,
    ResourceUsage as ResourceUsage,
)
from .validation import (
    EvaluatorObservationAdapter as EvaluatorObservationAdapter,
    FrozenSwarmRSIValidator as FrozenSwarmRSIValidator,
)
from .stopping import (
    ConfigurableStopPolicy as ConfigurableStopPolicy,
    ScoreStopCondition as ScoreStopCondition,
    StopDecision as StopDecision,
)
from .coder_engine import ContractBoundedCoder as ContractBoundedCoder, CoderModel as CoderModel
from .coder_models import (
    CoderConfig as CoderConfig,
    CoderExitStatus as CoderExitStatus,
    CoderOutcome as CoderOutcome,
    CoderVersion as CoderVersion,
    MutationInspiration as MutationInspiration,
    ProgramMutationContext as ProgramMutationContext,
    ScientificActionContract as ScientificActionContract,
    ValidationCommand as ValidationCommand,
)
from .coder_openrouter import OpenRouterCoderModel as OpenRouterCoderModel
from .dynamic_visualization import (
    write_dynamic_research_view as write_dynamic_research_view,
)
from .campaign import (
    ActionExecution as ActionExecution,
    CampaignAccounting as CampaignAccounting,
    CampaignStep as CampaignStep,
    ProgramNode as ProgramNode,
    ProposedAction as ProposedAction,
    VSearchCoordinator as VSearchCoordinator,
    artifact_workspace_sha256 as artifact_workspace_sha256,
)

__all__ = [
    "ActionCandidate", "ActionOperator", "BoundedRSIStrategy",
    "DynamicResearchView", "EvidenceCoupledStrategy", "FrozenSwarmRSIValidator", "EvaluatorObservationAdapter",
    "HomogeneousIslandStrategy", "HeterogeneousMultiIslandStrategy",
    "HypothesisProgramTreeStrategy",
    "MultiParentGraphStrategy", "ProposalDirective", "PolicyGenome", "PolicyTrial",
    "ResearchDependency", "ScientificFamily", "SearchBudget", "SearchDecision",
    "SearchEvent", "SearchObservation", "SearchSnapshot", "assert_acyclic_actions",
    "CLAUDE_CODE_DSV4PRO_S43588", "ResourceEnvelope", "ResourceLedger",
    "ResourceLimitReached", "ResourceUsage",
    "ConfigurableStopPolicy", "ScoreStopCondition", "StopDecision",
    "CoderConfig", "CoderExitStatus", "CoderModel", "CoderOutcome",
    "CoderVersion", "ContractBoundedCoder", "MutationInspiration",
    "ProgramMutationContext", "OpenRouterCoderModel",
    "ScientificActionContract", "ValidationCommand",
    "write_dynamic_research_view",
    "CampaignAccounting", "CampaignStep", "ProgramNode", "ProposedAction",
    "VSearchCoordinator", "ActionExecution", "artifact_workspace_sha256",
]
