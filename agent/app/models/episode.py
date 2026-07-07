from __future__ import annotations
from enum import Enum
from pydantic import BaseModel


class LoopMode(str, Enum):
    NONE = "none"
    SINGLE_PRIOR = "single"
    ALL_PRIOR = "all"


class TreatmentConfig(BaseModel):
    treatment: str          # "0", "1a", "1b", "2", "3", "4", "5", "6", "7"
    loop_mode: LoopMode = LoopMode.NONE
    use_gepa_prompts: bool = False
    use_advisor: bool = False
    use_rlm: bool = False
    use_ltm: bool = False
    use_deep_agent: bool = True  # False only for T0 (vanilla create_agent)

    @property
    def primary_is_27b(self) -> bool:
        return self.treatment == "1b"

    @property
    def is_baseline(self) -> bool:
        """Treatments with no loop — run N=20 independent episodes."""
        return self.loop_mode == LoopMode.NONE


TREATMENT_CONFIGS: dict[str, TreatmentConfig] = {
    "0":  TreatmentConfig(treatment="0",  use_deep_agent=False),
    "1a": TreatmentConfig(treatment="1a"),
    "1b": TreatmentConfig(treatment="1b"),
    "2":  TreatmentConfig(treatment="2",  use_gepa_prompts=True),
    "3":  TreatmentConfig(treatment="3",  use_gepa_prompts=True, loop_mode=LoopMode.SINGLE_PRIOR),
    "4":  TreatmentConfig(treatment="4",  use_gepa_prompts=True, loop_mode=LoopMode.ALL_PRIOR),
    "5":  TreatmentConfig(treatment="5",  use_gepa_prompts=True, loop_mode=LoopMode.ALL_PRIOR, use_advisor=True),
    "6":  TreatmentConfig(treatment="6",  use_gepa_prompts=True, loop_mode=LoopMode.ALL_PRIOR, use_advisor=True, use_rlm=True),
    "7":  TreatmentConfig(treatment="7",  use_gepa_prompts=True, loop_mode=LoopMode.ALL_PRIOR, use_advisor=True, use_rlm=True, use_ltm=True),
}


class EpisodeRecord(BaseModel):
    episode_id: int
    treatment: str
    pnl: float
    c_stat: float
    acceptance_rate: float
    loans_funded: int
    total_principal: float
    approach: str
    hypothesis: str
    skill_path: str
    tokens_used: int = 0
    duration_s: float = 0.0

    def to_index_line(self) -> str:
        return (
            f"Episode {self.episode_id:03d} | "
            f"P&L: ${self.pnl / 1000:.1f}k | "
            f"C-stat: {self.c_stat:.3f} | "
            f"Acceptance: {self.acceptance_rate:.1%} | "
            f"Loans: {self.loans_funded} | "
            f"Approach: {self.approach} | "
            f"Hypothesis: {self.hypothesis} | "
            f"Skill: {self.skill_path}"
        )
