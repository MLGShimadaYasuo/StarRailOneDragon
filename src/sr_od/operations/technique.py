import time

from dataclasses import dataclass
from enum import Enum
from cv2.typing import MatLike
from typing import ClassVar, Optional

import cv2
import numpy as np

from one_dragon.base.geometry.rectangle import Rect
from one_dragon.base.operation.operation_edge import node_from
from one_dragon.base.operation.operation_node import operation_node
from one_dragon.base.operation.operation_round_result import OperationRoundResult
from one_dragon.utils import cv2_utils, str_utils
from one_dragon.utils.i18_utils import gt
from sr_od.config.character_const import Character, SILVERWOLF_999
from sr_od.context.sr_context import SrContext
from sr_od.operations.sr_operation import SrOperation
from sr_od.screen_state import common_screen_state


class TechniqueFightAction(Enum):
    """配置了秘技战斗策略的角色下一步需要执行的动作。"""

    DEFAULT = 'default'
    USE_TECHNIQUE = 'use_technique'
    WAIT = 'wait'
    ATTACK_DIRECTION = 'attack_direction'
    NORMAL_ATTACK = 'normal_attack'


@dataclass(frozen=True)
class TechniqueFightPolicy:
    """需要保持秘技状态或定制开怪动作的角色配置。"""

    active_icon_rect: Rect
    active_icon_slot_gap: int
    active_icon_hsv_lower: tuple[int, int, int]
    active_icon_hsv_upper: tuple[int, int, int]
    active_icon_min_ratio: float
    activation_grace_seconds: float
    wait_after_use: float
    wait_before_attack: float
    attack_directions: tuple[str, ...]
    technique_only_fallback_after: Optional[float]
    keep_active_after_attack: bool
    check_after_transport: bool


TECHNIQUE_FIGHT_POLICY_BY_CHARACTER_ID: dict[str, TechniqueFightPolicy] = {
    SILVERWOLF_999.id: TechniqueFightPolicy(
        active_icon_rect=Rect(1607, 304, 1645, 344),
        active_icon_slot_gap=93,
        active_icon_hsv_lower=(15, 80, 100),
        active_icon_hsv_upper=(40, 255, 255),
        active_icon_min_ratio=0.15,
        activation_grace_seconds=1.5,
        wait_after_use=4.0,
        wait_before_attack=1.0,
        attack_directions=('w', 'a', 's', 'd'),
        technique_only_fallback_after=8.0,
        keep_active_after_attack=True,
        check_after_transport=True,
    ),
}


def get_current_character(ctx: SrContext) -> Optional[Character]:
    character_list = ctx.team_info.character_list
    current_active = ctx.team_info.current_active
    if (
            character_list is None
            or current_active < 0
            or current_active >= len(character_list)
    ):
        return None
    return character_list[current_active]


def get_technique_fight_policy(character: Optional[Character]) -> Optional[TechniqueFightPolicy]:
    if character is None:
        return None
    return TECHNIQUE_FIGHT_POLICY_BY_CHARACTER_ID.get(character.id)


def get_current_technique_fight_policy(ctx: SrContext) -> Optional[TechniqueFightPolicy]:
    return get_technique_fight_policy(get_current_character(ctx))


def is_current_technique_active(ctx: SrContext, screen: MatLike) -> bool:
    """按当前角色的通用策略检测秘技是否仍处于开启状态。"""
    policy = get_current_technique_fight_policy(ctx)
    if policy is None or screen is None:
        return False

    current_active = ctx.team_info.current_active
    base_rect = policy.active_icon_rect
    offset_y = current_active * policy.active_icon_slot_gap
    rect = Rect(base_rect.x1, base_rect.y1 + offset_y, base_rect.x2, base_rect.y2 + offset_y)
    if screen.shape[0] < rect.y2 or screen.shape[1] < rect.x2:
        return False

    part = cv2_utils.crop_image_only(screen, rect)
    if part is None or part.size == 0:
        return False

    hsv = cv2.cvtColor(part, cv2.COLOR_RGB2HSV)
    yellow_mask = cv2.inRange(
        hsv,
        np.array(policy.active_icon_hsv_lower, dtype=np.uint8),
        np.array(policy.active_icon_hsv_upper, dtype=np.uint8),
    )
    yellow_ratio = cv2.countNonZero(yellow_mask) / yellow_mask.size
    return yellow_ratio >= policy.active_icon_min_ratio


def should_use_buff_technique(ctx: SrContext, screen: MatLike) -> bool:
    """判断当前BUFF类角色是否需要预先使用秘技。"""
    policy = get_current_technique_fight_policy(ctx)
    if policy is not None:
        if is_current_technique_active(ctx, screen):
            return False
        # 刚按下秘技时图标可能还未显示，避免相邻节点连续按键。
        return not (
                ctx.technique_used
                and ctx.last_use_tech_time > 0
                and time.time() - ctx.last_use_tech_time < policy.activation_grace_seconds
        )
    return not ctx.tech_used_in_lasting


def should_preserve_technique_after_attack(ctx: SrContext, technique_fight: bool) -> bool:
    policy = get_current_technique_fight_policy(ctx)
    return technique_fight and policy is not None and policy.keep_active_after_attack


class TechniqueFightState:
    """一次开怪指令内由角色配置驱动的秘技与攻击状态。"""

    def __init__(self):
        self.character_id: Optional[str] = None
        self.active_since: Optional[float] = None
        self.activation_requested_at: Optional[float] = None
        self.battle_entered_at: Optional[float] = None
        self.attack_directions: tuple[str, ...] = ()
        self.attack_direction_idx: int = 0
        self.alert_started_at: Optional[float] = None
        self.fallback_attack_done: bool = False

    def reset(self) -> None:
        self.character_id = None
        self.active_since = None
        self.activation_requested_at = None
        self.battle_entered_at = None
        self.attack_directions = ()
        self.attack_direction_idx = 0
        self.alert_started_at = None
        self.fallback_attack_done = False

    def _get_policy(self, ctx: SrContext) -> Optional[TechniqueFightPolicy]:
        character = get_current_character(ctx)
        policy = get_technique_fight_policy(character)
        if character is None or policy is None:
            self.reset()
            return None
        if self.character_id != character.id:
            self.reset()
            self.character_id = character.id
            self.attack_directions = policy.attack_directions
        return policy

    def _reset_activation(self) -> None:
        self.active_since = None
        self.activation_requested_at = None
        self.attack_direction_idx = 0

    def mark_technique_used(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self.active_since = now
        self.activation_requested_at = now
        self.attack_direction_idx = 0

    @property
    def attack_direction(self) -> Optional[str]:
        if self.attack_direction_idx >= len(self.attack_directions):
            return None
        return self.attack_directions[self.attack_direction_idx]

    def mark_direction_attacked(self) -> None:
        if self.attack_direction_idx < len(self.attack_directions):
            self.attack_direction_idx += 1

    def next_action(self, ctx: SrContext, screen: MatLike,
                    technique_fight: bool, technique_only: bool,
                    now: Optional[float] = None) -> TechniqueFightAction:
        """根据当前角色配置和秘技状态给出下一动作。"""
        now = time.time() if now is None else now
        policy = self._get_policy(ctx)
        if policy is None:
            return TechniqueFightAction.DEFAULT
        if self.battle_entered_at is None:
            self.battle_entered_at = now

        if not technique_fight:
            if now - self.battle_entered_at < policy.wait_before_attack:
                return TechniqueFightAction.WAIT
            return TechniqueFightAction.NORMAL_ATTACK

        technique_active = is_current_technique_active(ctx, screen)
        if technique_active:
            if self.active_since is None:
                self.active_since = ctx.last_use_tech_time if ctx.last_use_tech_time > 0 else now
                self.attack_direction_idx = 0
            self.activation_requested_at = None
        elif (
                self.activation_requested_at is not None
                and now - self.activation_requested_at < policy.activation_grace_seconds
        ):
            return TechniqueFightAction.WAIT
        else:
            self._reset_activation()
            return TechniqueFightAction.USE_TECHNIQUE

        if technique_only:
            return TechniqueFightAction.WAIT

        attack_after = max(
            self.active_since + policy.wait_after_use,
            self.battle_entered_at + policy.wait_before_attack,
        )
        if now < attack_after:
            return TechniqueFightAction.WAIT
        if self.attack_direction is not None:
            return TechniqueFightAction.ATTACK_DIRECTION
        return TechniqueFightAction.NORMAL_ATTACK

    def should_fallback_normal_attack(self, ctx: SrContext,
                                      action: TechniqueFightAction,
                                      technique_fight: bool,
                                      technique_only: bool,
                                      with_alert: bool,
                                      now: Optional[float] = None) -> bool:
        """仅秘技策略持续检测到敌人时，按配置决定是否补一次普攻。"""
        now = time.time() if now is None else now
        policy = self._get_policy(ctx)
        waiting_with_alert = (
                policy is not None
                and policy.technique_only_fallback_after is not None
                and technique_fight
                and technique_only
                and action == TechniqueFightAction.WAIT
                and self.active_since is not None
                and with_alert
        )
        if not waiting_with_alert:
            self.alert_started_at = None
            return False
        if self.alert_started_at is None:
            self.alert_started_at = now
            return False
        if self.fallback_attack_done:
            return False
        if now - self.alert_started_at < policy.technique_only_fallback_after:
            return False

        self.fallback_attack_done = True
        return True


def pc_can_use_technique(ctx: SrContext, screen: MatLike, key: str) -> bool:
    """
    PC端使用 判断当前是否可以使用秘技 - 秘技按钮上有显示快捷键
    :param ctx: 上下文
    :param screen: 游戏画面
    :param key: 秘技按键
    :return:
    """
    area = ctx.screen_loader.get_area('大世界', '秘技-快捷键')
    part = cv2_utils.crop_image_only(screen, area.rect)
    # cv2_utils.show_image(part, win_name='pc_can_use_technique', wait=0)
    ocr_result = ctx.ocr.run_ocr_single_line(part)

    if ocr_result is not None and ocr_result.lower() == key.lower():
        return True
    else:
        return False


def get_technique_point(ctx: SrContext, screen: MatLike) -> Optional[int]:
    """
    识别剩余的秘技点数
    :param ctx: 上下文
    :param screen: 游戏画面
    :return:
    """
    area_name_list = ['秘技-点数-1', '秘技-点数-2']
    for area_name in area_name_list:
        area = ctx.screen_loader.get_area('大世界', area_name)
        part = cv2_utils.crop_image_only(screen, area.rect)

        ocr_result = ctx.ocr.run_ocr_single_line(part, strict_one_line=True)
        point = str_utils.get_positive_digits(ocr_result, None)
        if point is not None:
            return point

    return None


class UseTechniqueResult:

    def __init__(self, use_tech: bool = False, with_dialog: bool = False,
                 use_consumable_times: int = 0,
                 consumable_chosen: bool = False):

        self.use_tech: bool = use_tech
        """是否使用了秘技"""

        self.with_dialog: bool = with_dialog
        """是否出现了消耗品对话框"""

        self.use_consumable_times: int = use_consumable_times
        """使用消耗品的数量"""

        self.consumable_chosen: bool = consumable_chosen
        """是否已经选择了消耗品"""

        self.no_target_consumable_times: int = 0  # 没有目标消耗品的次数


class UseTechnique(SrOperation):

    STATUS_CAN_USE: ClassVar[str] = '可使用秘技'
    STATUS_CONTINUE_USE: ClassVar[str] = '继续使用秘技'

    def __init__(self, ctx: SrContext,
                 max_consumable_cnt: int = 0,
                 need_check_available: bool = False,
                 need_check_point: bool = False,
                 trick_snack: bool = False,
                 exit_after_use: bool = True,
                 ):
        """
        需在大世界页面中使用
        用当前角色使用秘技
        返回 data=UseTechniqueResult
        这个方法有用在了开怪上 因此判断需要快点 防止被袭
        :param ctx:
        :param max_consumable_cnt: 秘技点不足时最多使用的消耗品个数
        :param need_check_available: 是否需要检查秘技是否可用 普通大世界战斗后 会有一段时间才能使用秘技
        :param need_check_point: 是否检测剩余秘技点再使用。如果没有秘技点 又不能用消耗品 那就不使用了。目前OCR较慢 不建议开启
        :param trick_snack: 只使用奇巧零食
        :param exit_after_use: 当能使用秘技的情况下 一定要使用了秘技再退出
        """
        SrOperation.__init__(self, ctx, op_name=gt('施放秘技'))

        self.no_dialog_times: int = 0  # 没有出现快速恢复对话框的次数
        self.max_consumable_cnt: int = max_consumable_cnt  # 最多使用的消耗品个数
        self.trick_snack: bool = trick_snack  # 只使用奇巧零食
        self.exit_after_use: bool = exit_after_use  # 当能使用秘技的情况下 一定要使用了秘技再退出

        self.need_check_available: bool = need_check_available  # 是否需要检查秘技是否可用
        self.need_check_point: bool = need_check_point  # 是否检测剩余秘技点再使用

        self.op_result: UseTechniqueResult = UseTechniqueResult()  # 最后返回的结果

    @node_from(from_name='等待大世界继续', status=STATUS_CONTINUE_USE)
    @operation_node(name='检测秘技点', is_start_node=True)
    def _check_technique_point(self) -> OperationRoundResult:
        if self.need_check_point:
            screen = self.last_screenshot
            point = get_technique_point(self.ctx, screen)
            if point is not None and point > 0:  # 有秘技点 随便用
                return self.round_success(UseTechnique.STATUS_CAN_USE)
            elif self.max_consumable_cnt == 0:  # 没有秘技点又不能用药 就不要用了
                return self.round_success()
            else:  # 没有秘技点 可能有药 尝试
                return self.round_success(UseTechnique.STATUS_CAN_USE)
        else:
            return self.round_success(UseTechnique.STATUS_CAN_USE)

    @node_from(from_name='检测秘技点', status=STATUS_CAN_USE)
    @operation_node(name='使用秘技')
    def _use(self) -> OperationRoundResult:
        if self.need_check_available and not self.op_result.with_dialog:
            # 之前出现过消耗品对话框的话 这里就不需要判断了
            screen = self.last_screenshot
            if not pc_can_use_technique(self.ctx, screen, self.ctx.game_config.key_technique):
                return self.round_retry(wait=0.1)

        self.ctx.controller.use_technique()
        self.ctx.last_use_tech_time = time.time()  # 记录秘技的使用时间
        self.ctx.controller.stop_moving_forward()  # 在使用秘技中停止移动 可以取消停止移动的后摇
        self.op_result.use_tech = True  # 与context的状态分开 ctx的只负责记录开怪位 后续考虑变量改名
        self.ctx.technique_used = True
        return self.round_success(wait=0.2)

    @node_from(from_name='使用秘技')
    @operation_node(name='确认')
    def _confirm(self) -> OperationRoundResult:
        """
        使用消耗品确认
        :return:
        """
        if self.ctx.team_info.is_attack_technique:  # 攻击类的 使用完就取消标记
            self.ctx.technique_used = False

        screen = self.last_screenshot
        # cv2_utils.show_image(screen, win_name='technique')
        if common_screen_state.is_normal_in_world(self.ctx, screen):
            # 没有出现消耗品的情况 要尽快返回继续原来的指令 因此不等待
            return self.round_success(FastRecover.STATUS_NO_NEED_CONSUMABLE, data=self.op_result)

        # 不在大世界的情况 可以慢一点判断

        # 不在大世界 要么进入了战斗 要么是出现了快恢复对话框 都可以认为秘技已经生效
        # 可以将上次使用时间设置为0
        self.ctx.last_use_tech_time = 0
        result = self.round_by_find_area(screen, '快速恢复对话框', '快速恢复标题')
        if not result.is_success:
            self.no_dialog_times += 1
            if self.no_dialog_times > 3:  # 没有出现对话框的话 认为进入了战斗
                return self.round_success(FastRecover.STATUS_NO_NEED_CONSUMABLE, data=self.op_result)
            else:
                return self.round_wait(wait=0.5)
        else:
            self.no_dialog_times = 0
            self.op_result.use_tech = False  # 出现了对话框 那么之前使用秘技没有成功
            self.op_result.with_dialog = True
            self.ctx.technique_used = False

            return FastRecover.handle_consumable_dialog(
                self, self.ctx, screen, self.op_result,
                max_consumable_cnt=self.max_consumable_cnt,
                quirky_snacks=self.trick_snack
            )

    @node_from(from_name='确认', status='使用了消耗品')
    @node_from(from_name='确认', status='没使用消耗品')
    @operation_node(name='等待大世界继续', node_max_retry_times=20)
    def _wait_in_world_to_other(self):
        """
        使用后判断是否在大世界
        这里需要足够快的判断 方便后续的指令进行 因此尽量等待少的时间
        :return:
        """
        # 没有出现过对话框 那么直接返回即可
        if not self.op_result.with_dialog:
            return self.round_success(data=self.op_result)

        screen = self.last_screenshot
        if common_screen_state.is_normal_in_world(self.ctx, screen):
            if self.exit_after_use:  # 要求使用了秘技才能退出
                if self.op_result.use_tech:  # 已经使用了秘技 可以退出
                    return self.round_success(data=self.op_result)
                elif self.op_result.with_dialog and self.op_result.use_consumable_times > 0:
                    # 使用了消耗品 重新尝试使用秘技
                    self.op_result: UseTechniqueResult = UseTechniqueResult()
                    return self.round_success(status=UseTechnique.STATUS_CONTINUE_USE)
                else:  # 没法使用消耗品 只能退出
                    return self.round_success(data=self.op_result)
            else:
                return self.round_success(data=self.op_result)
        else:
            return self.round_retry(status='未在大世界画面', wait=0.1)


class CheckTechniquePoint(SrOperation):

    def __init__(self, ctx: SrContext):
        """
        需在大世界页面中使用
        通过右下角数字 检测当前剩余的秘技点数
        返回附加状态为秘技点数
        :param ctx:
        """
        SrOperation.__init__(self, ctx, op_name=gt('检测秘技点数'))

    @operation_node(name='识别画面', node_max_retry_times=10, is_start_node=True)
    def check(self) -> OperationRoundResult:
        screen = self.last_screenshot
        if not common_screen_state.is_normal_in_world(self.ctx, screen):
            return self.round_retry('未在大世界界面', wait=1)

        digit = get_technique_point(self.ctx, screen)

        if digit is None:
            return self.round_retry('未检测到数字', wait=0.5)

        return self.round_success(status=str(digit), data=digit)


class FastRecover(SrOperation):

    STATUS_NO_NEED_CONSUMABLE: ClassVar[str] = '无需使用消耗品'
    STATUS_NO_USE_CONSUMABLE: ClassVar[str] = '没使用消耗品'
    STATUS_USE_CONSUMABLE: ClassVar[str] = '使用了消耗品'

    def __init__(self, ctx: SrContext,
                 max_consumable_cnt: int = 0,
                 quirky_snacks: bool = False
                 ):
        """
        使用消耗品恢复秘技点
        这个方法有用在了开怪上 因此判断需要快点 防止被袭
        :param ctx:
        :param max_consumable_cnt: 秘技点不足时最多使用的消耗品个数
        :param quirky_snacks: 只使用奇巧零食
        """
        SrOperation.__init__(self, ctx, op_name=gt('快速恢复'))

        self.max_consumable_cnt: int = max_consumable_cnt  # 最多使用的消耗品个数
        self.quirky_snacks: bool = quirky_snacks  # 只使用奇巧零食
        self.op_result: UseTechniqueResult = UseTechniqueResult()

    @operation_node(name='使用消耗品', is_start_node=True)
    def use(self) -> OperationRoundResult:
        screen = self.last_screenshot
        return FastRecover.handle_consumable_dialog(self, self.ctx, screen, self.op_result,
                                                    max_consumable_cnt=self.max_consumable_cnt,
                                                    quirky_snacks=self.quirky_snacks)

    @node_from(from_name='使用消耗品')
    @operation_node(name='等待大世界', node_max_retry_times=20)
    def wait_in_world(self):
        """
        使用后判断是否在大世界
        这里需要足够快的判断 方便后续的指令进行 因此尽量等待少的时间
        :return:
        """
        # 没有出现过对话框 那么直接返回即可
        if not self.op_result.with_dialog:
            return self.round_success(data=self.op_result)

        screen = self.last_screenshot
        if common_screen_state.is_normal_in_world(self.ctx, screen):
            return self.round_success(data=self.op_result)
        else:
            return self.round_retry(status='未在大世界画面', wait=0.1)

    @node_from(from_name='使用消耗品', success=False)
    @operation_node(name='失败后失败画面')
    def fail_check_screen(self) -> OperationRoundResult:
        """
        指令失败时 识别画面 确定还在 快速恢复 对话框中
        部分战斗场景会被误识别为 快速恢复，这时 use 方法会失败，此时再兜底判断是不是在对话框中，
        :return:
        """
        screen = self.last_screenshot

        result = self.round_by_find_area(screen, '快速恢复对话框', '快速恢复标题')

        if result.is_success:  # 如果在对话框中 说明指令真的失败了
            return self.round_fail(data=self.op_result)
        else:
            return self.round_success('误判为快速恢复', data=self.op_result)  # 不在对话框 说明是误判

    @staticmethod
    def handle_consumable_dialog(
            op: SrOperation,
            ctx: SrContext,
            screen: MatLike,
            op_result: UseTechniqueResult,
            max_consumable_cnt,
            quirky_snacks: bool,
    ) -> OperationRoundResult:
        if max_consumable_cnt == 0:  # 不可以使用消耗品 点击退出
            result = op.round_by_find_and_click_area(screen, '快速恢复对话框', '取消')
            if result.is_success:
                # 没有使用消耗品的情况 退出对话框后 要尽快识别到在大世界了 方便后续指令 因此不等待
                return op.round_success(FastRecover.STATUS_NO_USE_CONSUMABLE, data=op_result)
            else:
                return op.round_retry(result.status, wait=1)
        elif op.round_by_find_area(screen, '快速恢复对话框', '暂无可用消耗品').is_success:  # 没有消耗品可以使用
            result = op.round_by_find_and_click_area(screen, '快速恢复对话框', '取消')
            if result.is_success:
                if op_result.use_consumable_times > 0:
                    # 使用了消耗品的情况 退出对话框后 需要尽快识别到在大世界了 方便再使用秘技 因此不等待
                    return op.round_success(FastRecover.STATUS_USE_CONSUMABLE, data=op_result)
                else:
                    # 没有使用消耗品的情况 退出对话框后 要尽快识别到在大世界了 方便后续指令 因此不等待
                    return op.round_success(FastRecover.STATUS_NO_USE_CONSUMABLE, data=op_result)
            else:
                return op.round_retry(result.status, wait=1)
        elif max_consumable_cnt > 0 and op_result.use_consumable_times >= max_consumable_cnt:  # 已经用了足够的消耗品了
            result = op.round_by_find_and_click_area(screen, '快速恢复对话框', '取消')
            if result.is_success:
                # 使用了消耗品的情况 退出对话框后 需要尽快识别到在大世界了 方便再使用秘技 因此不等待
                return op.round_success(FastRecover.STATUS_USE_CONSUMABLE, data=op_result)
            else:
                return op.round_retry(result.status, wait=1)
        else:  # 还需要使用消耗品
            if quirky_snacks and not op_result.consumable_chosen:  # 理论上只有第1次需要选择 即还没有使用任何消耗品
                choose = FastRecover.choose_consumable(op, screen)
                if not choose:
                    op_result.no_target_consumable_times += 1
                    # 当消耗品种类多的时候 奇巧零食可能比较慢才出现 因此稍微重试一下
                    if op_result.no_target_consumable_times < 3:
                        return op.round_wait('未匹配到奇巧零食', wait=1)

                    result = op.round_by_find_and_click_area(screen, '快速恢复对话框', '取消')
                    if result.is_success:
                        # 没有选择到目标消耗品 因此是没有使用消耗品的情况 退出对话框后 需要尽快识别到在大世界了 方便后续指令 因此不等待
                        return op.round_success(FastRecover.STATUS_NO_USE_CONSUMABLE, data=op_result)
                    else:
                        return op.round_retry(result.status, wait=1)
                else:
                    op_result.consumable_chosen = True
                    time.sleep(0.5)

            result = op.round_by_find_and_click_area(screen, '快速恢复对话框', '确认')
            if result.is_success:
                op_result.use_consumable_times += 1
                return op.round_wait(FastRecover.STATUS_USE_CONSUMABLE, wait=0.5, data=op_result)
            elif result.status.startswith('未找到'):  # 使用满了
                result = op.round_by_find_and_click_area(screen, '快速恢复对话框', '取消')
                if result.is_success:
                    if op_result.use_consumable_times > 0:
                        # 使用了消耗品的情况 退出对话框后 需要尽快识别到在大世界了 方便再使用秘技 因此不等待
                        return op.round_success(FastRecover.STATUS_USE_CONSUMABLE, data=op_result)
                    else:
                        # 没有使用消耗品的情况 退出对话框后 要尽快识别到在大世界了 方便后续指令 因此不等待
                        return op.round_success(FastRecover.STATUS_NO_USE_CONSUMABLE, data=op_result)
                else:
                    return op.round_retry(result.status, wait=1)
            else:
                return op.round_retry(result.status, wait=1)

    @staticmethod
    def choose_consumable(op: SrOperation, screen: MatLike) -> bool:
        """
        选择特定的消耗品 目前固定使用
        :param op: 当前运行的指令
        :param screen:
        :return: 是否找到目标消耗品
        """
        result = op.round_by_find_and_click_area(screen, '快速恢复对话框', '奇巧零食')
        return result.is_success


def __debug():
    ctx = SrContext()
    ctx.init_by_config()
    ctx.init_ocr()
    ctx.start_running()
    op = UseTechnique(ctx,
                      ctx.world_patrol_config.max_consumable_cnt,
                      True,
                      True,
                      ctx.game_config.use_quirky_snacks,
                    )
    op.execute()


if __name__ == '__main__':
    __debug()
