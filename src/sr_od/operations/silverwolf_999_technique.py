import time
from enum import Enum
from typing import ClassVar, Optional

import cv2
import numpy as np
from cv2.typing import MatLike

from one_dragon.base.geometry.rectangle import Rect
from one_dragon.utils import cv2_utils
from sr_od.config.character_const import SILVERWOLF_999
from sr_od.context.sr_context import SrContext


class SilverWolf999TechniqueAction(Enum):
    """银狼LV.999在开怪流程中下一步需要执行的动作。"""

    DEFAULT = 'default'
    USE_TECHNIQUE = 'use_technique'
    WAIT = 'wait'
    ATTACK_DIRECTION = 'attack_direction'
    NORMAL_ATTACK = 'normal_attack'


def is_silverwolf_999_current(ctx: SrContext) -> bool:
    """当前操作角色是否为银狼LV.999。"""
    character_list = ctx.team_info.character_list
    current_active = ctx.team_info.current_active
    return (
            character_list is not None
            and 0 <= current_active < len(character_list)
            and character_list[current_active] is not None
            and character_list[current_active].id == SILVERWOLF_999.id
    )


class SilverWolf999TechniqueDetector:
    """通过队伍栏旁的黄色图标判断银狼LV.999秘技是否保持开启。"""

    # 1920x1080 标准截图中，一号位秘技图标区域；截图进入识别前会统一缩放到该分辨率。
    FIRST_SLOT_ICON_RECT: ClassVar[Rect] = Rect(1607, 304, 1645, 344)
    SLOT_VERTICAL_GAP: ClassVar[int] = 93

    # 图标主体为高饱和度金黄色。使用占比而不是固定像素数，给抗锯齿和亮度变化留余量。
    YELLOW_LOWER: ClassVar[np.ndarray] = np.array([15, 80, 100], dtype=np.uint8)
    YELLOW_UPPER: ClassVar[np.ndarray] = np.array([40, 255, 255], dtype=np.uint8)
    MIN_YELLOW_RATIO: ClassVar[float] = 0.15

    @staticmethod
    def get_icon_rect(ctx: SrContext) -> Optional[Rect]:
        if not is_silverwolf_999_current(ctx):
            return None

        offset_y = ctx.team_info.current_active * SilverWolf999TechniqueDetector.SLOT_VERTICAL_GAP
        rect = SilverWolf999TechniqueDetector.FIRST_SLOT_ICON_RECT
        return Rect(rect.x1, rect.y1 + offset_y, rect.x2, rect.y2 + offset_y)

    @staticmethod
    def is_active(ctx: SrContext, screen: MatLike) -> bool:
        rect = SilverWolf999TechniqueDetector.get_icon_rect(ctx)
        if rect is None or screen is None:
            return False
        if screen.shape[0] < rect.y2 or screen.shape[1] < rect.x2:
            return False

        part = cv2_utils.crop_image_only(screen, rect)
        if part is None or part.size == 0:
            return False

        hsv = cv2.cvtColor(part, cv2.COLOR_RGB2HSV)
        yellow_mask = cv2.inRange(
            hsv,
            SilverWolf999TechniqueDetector.YELLOW_LOWER,
            SilverWolf999TechniqueDetector.YELLOW_UPPER,
        )
        yellow_ratio = cv2.countNonZero(yellow_mask) / yellow_mask.size
        return yellow_ratio >= SilverWolf999TechniqueDetector.MIN_YELLOW_RATIO


def should_use_buff_technique(ctx: SrContext, screen: MatLike) -> bool:
    """
    判断当前BUFF类角色是否需要预先使用秘技。

    银狼LV.999以实际图标为准，其他角色沿用原有持续时间判断。
    """
    if is_silverwolf_999_current(ctx):
        if SilverWolf999TechniqueDetector.is_active(ctx, screen):
            return False
        # 刚按下秘技时图标可能还未显示，避免落地检查与路线检查连续按键。
        return not (
                ctx.technique_used
                and ctx.last_use_tech_time > 0
                and time.time() - ctx.last_use_tech_time
                < SilverWolf999TechniqueState.ACTIVATION_GRACE_SECONDS
        )
    return not ctx.tech_used_in_lasting


class SilverWolf999TechniqueState:
    """一次开怪指令内的银狼LV.999秘技与四方向攻击状态。"""

    WAIT_AFTER_TECHNIQUE: ClassVar[float] = 4.0
    WAIT_BEFORE_BATTLE_ATTACK: ClassVar[float] = 1.0
    TECHNIQUE_ONLY_FALLBACK_ATTACK_AFTER: ClassVar[float] = 8.0
    ACTIVATION_GRACE_SECONDS: ClassVar[float] = 1.5
    ATTACK_DIRECTIONS: ClassVar[tuple[str, ...]] = ('w', 'a', 's', 'd')

    def __init__(self):
        self.active_since: Optional[float] = None
        self.activation_requested_at: Optional[float] = None
        self.battle_entered_at: Optional[float] = None
        self.attack_direction_idx: int = 0

    def reset(self) -> None:
        self.active_since = None
        self.activation_requested_at = None
        self.battle_entered_at = None
        self.attack_direction_idx = 0

    def mark_technique_used(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self.active_since = now
        self.activation_requested_at = now
        self.attack_direction_idx = 0

    @property
    def attack_direction(self) -> Optional[str]:
        if self.attack_direction_idx >= len(self.ATTACK_DIRECTIONS):
            return None
        return self.ATTACK_DIRECTIONS[self.attack_direction_idx]

    def mark_direction_attacked(self) -> None:
        if self.attack_direction_idx < len(self.ATTACK_DIRECTIONS):
            self.attack_direction_idx += 1

    def next_action(self, ctx: SrContext, screen: MatLike,
                    technique_fight: bool, technique_only: bool,
                    now: Optional[float] = None) -> SilverWolf999TechniqueAction:
        """根据图标和开关组合给出下一动作。"""
        now = time.time() if now is None else now
        if not is_silverwolf_999_current(ctx):
            self.reset()
            return SilverWolf999TechniqueAction.DEFAULT
        if self.battle_entered_at is None:
            self.battle_entered_at = now
        if not technique_fight:
            if now - self.battle_entered_at < self.WAIT_BEFORE_BATTLE_ATTACK:
                return SilverWolf999TechniqueAction.WAIT
            # 等待结束后至少执行一次普通攻击，再恢复原有战斗逻辑。
            return SilverWolf999TechniqueAction.NORMAL_ATTACK

        technique_active = SilverWolf999TechniqueDetector.is_active(ctx, screen)
        if technique_active:
            if self.active_since is None:
                # 路线阶段可能已经开启秘技，优先沿用真实按键时间计算4秒等待。
                self.active_since = ctx.last_use_tech_time if ctx.last_use_tech_time > 0 else now
                self.attack_direction_idx = 0
            self.activation_requested_at = None
        elif (
                self.activation_requested_at is not None
                and now - self.activation_requested_at < self.ACTIVATION_GRACE_SECONDS
        ):
            # 按键后图标存在短暂显示延迟，避免连续按键把刚开启的秘技关闭。
            return SilverWolf999TechniqueAction.WAIT
        else:
            self.reset()
            return SilverWolf999TechniqueAction.USE_TECHNIQUE

        if technique_only:
            return SilverWolf999TechniqueAction.WAIT
        attack_after = max(
            self.active_since + self.WAIT_AFTER_TECHNIQUE,
            self.battle_entered_at + self.WAIT_BEFORE_BATTLE_ATTACK,
        )
        if now < attack_after:
            return SilverWolf999TechniqueAction.WAIT
        if self.attack_direction is not None:
            return SilverWolf999TechniqueAction.ATTACK_DIRECTION
        return SilverWolf999TechniqueAction.NORMAL_ATTACK
