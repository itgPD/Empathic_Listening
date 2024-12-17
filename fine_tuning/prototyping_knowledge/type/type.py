from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Error:
    code: Optional[str]
    message: Optional[str]
    param: Optional[str]


@dataclass
class Hyperparameters:
    n_epochs: int
    batch_size: int
    learning_rate_multiplier: float


@dataclass
class FineTuningJob:
    id: str
    created_at: int
    error: Optional[Error]
    fine_tuned_model: str
    finished_at: Optional[int]
    hyperparameters: Hyperparameters
    model: str
    object: str
    organization_id: str
    result_files: List[str]
    seed: int
    status: str
    trained_tokens: int
    training_file: str
    validation_file: Optional[str]
    estimated_finish: Optional[int]
    integrations: List  # 型が特定できなければ `List` を使用
    user_provided_suffix: Optional[str]
