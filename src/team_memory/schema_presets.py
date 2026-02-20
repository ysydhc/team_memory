"""Built-in schema preset packs.

Each preset provides a complete set of experience_types, categories,
severity_levels, progress_states, and structured_fields tailored to a
specific team profile.

Presets:
  - software-dev   : Software development teams (current default, backward-compatible)
  - data-engineering: Data / ML engineering teams
  - devops          : Operations / SRE / platform teams
  - general         : Minimal, non-technical teams
"""

from __future__ import annotations

from team_memory.config import CategoryDef, ExperienceTypeDef, StructuredFieldDef


def _sf(name: str, label: str, type_: str = "text", required: bool = False) -> StructuredFieldDef:
    """Shorthand for creating a StructuredFieldDef."""
    return StructuredFieldDef(name=name, label=label, type=type_, required=required)


# =====================================================================
# Preset: software-dev (backward-compatible with existing hardcoded schema)
# =====================================================================

SOFTWARE_DEV_TYPES: list[ExperienceTypeDef] = [
    ExperienceTypeDef(id="general", label="通用经验"),
    ExperienceTypeDef(
        id="feature",
        label="功能需求",
        progress_states=["planning", "developing", "testing", "released"],
        structured_fields=[
            _sf("requirements", "需求描述"),
            _sf("acceptance_criteria", "验收标准", type_="list"),
            _sf("test_summary", "测试摘要"),
            _sf("release_notes", "发布说明"),
        ],
    ),
    ExperienceTypeDef(
        id="bugfix",
        label="Bug 修复",
        severity=True,
        progress_states=["open", "investigating", "fixed", "verified"],
        structured_fields=[
            _sf("reproduction_steps", "复现步骤"),
            _sf("environment", "环境信息"),
            _sf("error_logs", "错误日志"),
            _sf("impact_scope", "影响范围"),
            _sf("verification_result", "验证结果"),
        ],
    ),
    ExperienceTypeDef(
        id="tech_design",
        label="技术方案",
        progress_states=["researching", "reviewing", "implementing", "completed"],
        structured_fields=[
            _sf("alternatives", "备选方案", type_="list"),
            _sf("rollback_plan", "回滚方案"),
            _sf("data_migration", "数据迁移"),
            _sf("performance_data", "性能数据"),
            _sf("upstream_downstream", "上下游影响"),
            _sf("monitoring", "监控方案"),
        ],
    ),
    ExperienceTypeDef(
        id="incident",
        label="线上故障",
        severity=True,
        progress_states=["detected", "mitigating", "resolved", "post_mortem"],
        structured_fields=[
            _sf("reproduction_steps", "复现步骤"),
            _sf("environment", "环境信息"),
            _sf("error_logs", "错误日志"),
            _sf("impact_scope", "影响范围"),
            _sf("verification_result", "验证结果"),
            _sf("timeline", "处理时间线"),
            _sf("prevention", "预防措施"),
        ],
    ),
    ExperienceTypeDef(id="best_practice", label="最佳实践"),
    ExperienceTypeDef(id="learning", label="学习笔记"),
]

SOFTWARE_DEV_CATEGORIES: list[CategoryDef] = [
    CategoryDef(id="frontend", label="前端"),
    CategoryDef(id="backend", label="后端"),
    CategoryDef(id="database", label="数据库"),
    CategoryDef(id="infra", label="基础设施"),
    CategoryDef(id="performance", label="性能"),
    CategoryDef(id="security", label="安全"),
    CategoryDef(id="mobile", label="移动端"),
    CategoryDef(id="other", label="其他"),
]

SOFTWARE_DEV_SEVERITY: list[str] = ["P0", "P1", "P2", "P3", "P4"]


# =====================================================================
# Preset: data-engineering
# =====================================================================

DATA_ENG_TYPES: list[ExperienceTypeDef] = [
    ExperienceTypeDef(id="general", label="通用经验"),
    ExperienceTypeDef(
        id="bugfix",
        label="Bug 修复",
        severity=True,
        progress_states=["open", "investigating", "fixed", "verified"],
        structured_fields=[
            _sf("reproduction_steps", "复现步骤"),
            _sf("environment", "环境信息"),
            _sf("error_logs", "错误日志"),
            _sf("impact_scope", "影响范围"),
            _sf("verification_result", "验证结果"),
        ],
    ),
    ExperienceTypeDef(
        id="data_quality",
        label="数据质量事件",
        severity=True,
        progress_states=["detected", "analyzing", "fixing", "monitoring"],
        structured_fields=[
            _sf("affected_tables", "受影响表"),
            _sf("data_volume", "受影响数据量"),
            _sf("detection_method", "检测方式"),
            _sf("root_query", "问题 SQL/查询"),
            _sf("fix_query", "修复 SQL/脚本"),
        ],
    ),
    ExperienceTypeDef(
        id="pipeline_failure",
        label="管道故障",
        severity=True,
        progress_states=["detected", "investigating", "fixed", "retried", "verified"],
        structured_fields=[
            _sf("pipeline_name", "管道名称"),
            _sf("failure_stage", "失败阶段"),
            _sf("error_logs", "错误日志"),
            _sf("data_impact", "数据影响"),
            _sf("recovery_steps", "恢复步骤"),
        ],
    ),
    ExperienceTypeDef(
        id="schema_change",
        label="Schema 变更",
        progress_states=["proposed", "reviewing", "migrating", "completed"],
        structured_fields=[
            _sf("affected_tables", "受影响表"),
            _sf("migration_script", "迁移脚本"),
            _sf("rollback_plan", "回滚方案"),
            _sf("downstream_impact", "下游影响"),
        ],
    ),
    ExperienceTypeDef(id="best_practice", label="最佳实践"),
]

DATA_ENG_CATEGORIES: list[CategoryDef] = [
    CategoryDef(id="etl", label="ETL/ELT"),
    CategoryDef(id="warehouse", label="数据仓库"),
    CategoryDef(id="streaming", label="实时流处理"),
    CategoryDef(id="ml_ops", label="ML 运维"),
    CategoryDef(id="data_governance", label="数据治理"),
    CategoryDef(id="database", label="数据库"),
    CategoryDef(id="infra", label="基础设施"),
    CategoryDef(id="other", label="其他"),
]

DATA_ENG_SEVERITY: list[str] = ["P0", "P1", "P2", "P3", "P4"]


# =====================================================================
# Preset: devops
# =====================================================================

DEVOPS_TYPES: list[ExperienceTypeDef] = [
    ExperienceTypeDef(id="general", label="通用经验"),
    ExperienceTypeDef(
        id="incident",
        label="线上故障",
        severity=True,
        progress_states=["detected", "mitigating", "resolved", "post_mortem"],
        structured_fields=[
            _sf("alert_source", "告警来源"),
            _sf("impact_scope", "影响范围"),
            _sf("timeline", "处理时间线"),
            _sf("root_cause", "根因分析"),
            _sf("prevention", "预防措施"),
            _sf("monitoring_gaps", "监控缺口"),
        ],
    ),
    ExperienceTypeDef(
        id="deployment",
        label="部署记录",
        progress_states=["planned", "deploying", "verifying", "completed", "rolled_back"],
        structured_fields=[
            _sf("service_name", "服务名称"),
            _sf("version", "版本号"),
            _sf("change_summary", "变更摘要"),
            _sf("rollback_plan", "回滚方案"),
            _sf("verification_steps", "验证步骤"),
        ],
    ),
    ExperienceTypeDef(
        id="capacity_planning",
        label="容量规划",
        progress_states=["analyzing", "planning", "executing", "monitoring"],
        structured_fields=[
            _sf("current_usage", "当前用量"),
            _sf("growth_forecast", "增长预测"),
            _sf("action_plan", "扩容方案"),
            _sf("cost_estimate", "成本估算"),
        ],
    ),
    ExperienceTypeDef(
        id="runbook",
        label="操作手册",
        structured_fields=[
            _sf("trigger_condition", "触发条件"),
            _sf("steps", "操作步骤", type_="list"),
            _sf("expected_outcome", "预期结果"),
            _sf("escalation", "升级路径"),
        ],
    ),
    ExperienceTypeDef(
        id="postmortem",
        label="复盘报告",
        severity=True,
        progress_states=["drafting", "reviewing", "published"],
        structured_fields=[
            _sf("incident_summary", "事故摘要"),
            _sf("timeline", "时间线"),
            _sf("contributing_factors", "贡献因素"),
            _sf("action_items", "改进项", type_="list"),
            _sf("lessons_learned", "经验教训"),
        ],
    ),
]

DEVOPS_CATEGORIES: list[CategoryDef] = [
    CategoryDef(id="kubernetes", label="Kubernetes"),
    CategoryDef(id="ci_cd", label="CI/CD"),
    CategoryDef(id="monitoring", label="监控告警"),
    CategoryDef(id="networking", label="网络"),
    CategoryDef(id="security", label="安全"),
    CategoryDef(id="cloud", label="云服务"),
    CategoryDef(id="database", label="数据库"),
    CategoryDef(id="other", label="其他"),
]

DEVOPS_SEVERITY: list[str] = ["P0", "P1", "P2", "P3", "P4"]


# =====================================================================
# Preset: general (minimal, suitable for non-technical teams)
# =====================================================================

GENERAL_TYPES: list[ExperienceTypeDef] = [
    ExperienceTypeDef(id="general", label="通用经验"),
    ExperienceTypeDef(id="note", label="笔记"),
    ExperienceTypeDef(
        id="decision",
        label="决策记录",
        structured_fields=[
            _sf("context", "背景"),
            _sf("options", "选项", type_="list"),
            _sf("rationale", "决策依据"),
            _sf("outcome", "结果"),
        ],
    ),
    ExperienceTypeDef(
        id="action_item",
        label="待办事项",
        progress_states=["todo", "in_progress", "done"],
        structured_fields=[
            _sf("assignee", "负责人"),
            _sf("due_date", "截止日期"),
            _sf("priority", "优先级"),
        ],
    ),
]

GENERAL_CATEGORIES: list[CategoryDef] = [
    CategoryDef(id="process", label="流程"),
    CategoryDef(id="communication", label="沟通"),
    CategoryDef(id="planning", label="规划"),
    CategoryDef(id="review", label="复盘"),
    CategoryDef(id="other", label="其他"),
]

GENERAL_SEVERITY: list[str] = []  # Non-technical teams rarely need severity


# =====================================================================
# Registry of all presets
# =====================================================================

PresetPack = dict  # {experience_types, categories, severity_levels}

PRESET_REGISTRY: dict[str, PresetPack] = {
    "software-dev": {
        "experience_types": SOFTWARE_DEV_TYPES,
        "categories": SOFTWARE_DEV_CATEGORIES,
        "severity_levels": SOFTWARE_DEV_SEVERITY,
    },
    "data-engineering": {
        "experience_types": DATA_ENG_TYPES,
        "categories": DATA_ENG_CATEGORIES,
        "severity_levels": DATA_ENG_SEVERITY,
    },
    "devops": {
        "experience_types": DEVOPS_TYPES,
        "categories": DEVOPS_CATEGORIES,
        "severity_levels": DEVOPS_SEVERITY,
    },
    "general": {
        "experience_types": GENERAL_TYPES,
        "categories": GENERAL_CATEGORIES,
        "severity_levels": GENERAL_SEVERITY,
    },
}


def get_preset(name: str) -> PresetPack:
    """Return a preset pack by name, falling back to software-dev."""
    return PRESET_REGISTRY.get(name, PRESET_REGISTRY["software-dev"])


def list_presets() -> list[str]:
    """Return the list of available preset names."""
    return list(PRESET_REGISTRY.keys())
