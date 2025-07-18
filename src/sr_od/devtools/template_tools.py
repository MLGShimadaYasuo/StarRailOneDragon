import os
import shutil
from typing import Optional

import cv2
import numpy as np
from cv2.typing import MatLike

from one_dragon.base.screen.template_loader import TemplateLoader
from one_dragon.utils import os_utils, cv2_utils
from sr_od.config import game_const
from sr_od.sr_map import mini_map_utils


def _read_template_raw_image(sub_dir, template_id):
    dir_path = os_utils.get_path_under_work_dir('assets', 'template', sub_dir, template_id)
    if not os.path.exists(dir_path):
        return None
    img = cv2_utils.read_image(os.path.join(dir_path, 'raw.png'))

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def convert_template(template_id, save: bool = False):
    """
    把抠图后的图标灰度保存
    :param template_id:
    :param save:
    :return:
    """
    ih = TemplateLoader()
    template = ih.get_template(template_id)
    gray = cv2.cvtColor(template.raw, cv2.COLOR_BGRA2GRAY)
    mask = np.where(template.raw[..., 3] > 0, 255, 0).astype(np.uint8)
    cv2_utils.show_image(template.raw, win_name='origin')
    cv2_utils.show_image(gray, win_name='gray')
    cv2_utils.show_image(mask, win_name='mask', wait=0)
    if save:
        dir = os_utils.get_path_under_work_dir('images', 'template', template_id)
        cv2.imwrite(os.path.join(dir, 'gray.png'), gray)
        cv2.imwrite(os.path.join(dir, 'mask.png'), mask)


def init_tp_with_background(template_id: str, noise_threshold: int = 0):
    """
    对传送点进行抠图 尽量多截取一点黑色背景
    将裁剪出来的图片 转化保留灰度图和对应掩码
    如果原图有透明通道 会转化成无透明通道的
    最后结果图都会转化到 51*51 并居中
    会覆盖原图 发现有问题可在图片显示时退出程序
    :param template_id:
    :param noise_threshold: 消除多少个像素点以下的连通块 可以视情况调整
    :return:
    """
    sub_dir = 'mm_icon'
    raw = _read_template_raw_image(sub_dir, template_id)
    if raw.shape[2] == 4:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY)
    if noise_threshold > 0:
        # 消除白色的
        white_mask = cv2_utils.connection_erase(mask, threshold=noise_threshold)

        # 消除黑色的
        black_mask = cv2_utils.connection_erase(cv2.bitwise_not(white_mask), threshold=noise_threshold)
        mask = cv2.bitwise_or(white_mask, cv2.bitwise_not(black_mask))

    # 背景统一使用道路颜色 因为地图上传送点附近大概率都是道路 这样更方便匹配
    final_origin, final_mask = cv2_utils.convert_to_standard(raw, mask, width=51, height=51, bg_color=game_const.COLOR_MAP_ROAD_BGR)

    show_and_save(sub_dir, template_id, final_origin, final_mask)


def init_sp_with_background(template_id: str, noise_threshold: int = 0, sub_dir: str = 'mm_icon'):
    """
    对特殊点进行抠图
    特殊点会自带一些黑色背景 通过找黑色的菱形区域保留下来
    将裁剪出来的图片 转化保留灰度图和对应掩码
    如果原图有透明通道 会转化成无透明通道的
    最后结果图都会转化到 51*51 并居中
    会覆盖原图 发现有问题可在图片显示时退出程序
    :param template_id:
    :param noise_threshold: 连通块小于多少时认为是噪点 视情况调整
    :return:
    """
    tm = TemplateLoader()
    template = tm.get_template(sub_dir, template_id)
    raw = template.raw
    sim = cv2_utils.color_similarity_2d(raw, (240, 210, 160))  # 前景颜色 特殊点主体部分
    cv2_utils.show_image(sim, win_name='sim')
    front_binary = cv2.inRange(sim, 180, 255)
    cv2_utils.show_image(front_binary, win_name='front_binary')

    # 画出黑色的菱形背景
    gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    cv2_utils.show_image(gray, win_name='gray')
    black = cv2.inRange(gray, 0, 20)

    l1, r1, t1, b1 = cv2_utils.get_four_corner(black)
    print(l1, r1, t1, b1)
    # 找特殊点主体部分的几个角 可能会覆盖到菱形
    l2, r2, t2, b2 = cv2_utils.get_four_corner(front_binary)
    print(l2, r2, t2, b2)
    l = l1 if l1[0] < l2[0] else l2
    r = r1 if r1[0] > r2[0] else r2
    t = t1 if t1[1] < t2[1] else t2
    b = b1 if b1[1] > b2[1] else b2

    points = np.array([l, t, r, b], np.int32)
    points = points.reshape((-1, 1, 2))
    back_binary = np.zeros_like(gray)
    cv2.fillPoly(back_binary, [points], color=255)
    cv2_utils.show_image(back_binary, win_name='back_binary')

    binary = cv2.bitwise_or(front_binary, back_binary)
    cv2_utils.show_image(binary, win_name='binary')
    mask = cv2_utils.connection_erase(binary, threshold=noise_threshold)
    mask = cv2_utils.connection_erase(mask, threshold=noise_threshold, erase_white=False)

    final_origin, final_mask = cv2_utils.convert_to_standard(raw, mask, width=51, height=51, bg_color=game_const.COLOR_MAP_ROAD_BGR)

    show_and_save(sub_dir, template_id, final_origin, final_mask)


def init_ui_icon(template_id: str, noise_threshold: int = 0):
    """
    对界面交互图标进行抠图
    图标都是白色的 找一个较为深色背景的地图进行截图即可
    将裁剪出来的图片 转化保留灰度图和对应掩码
    如果原图有透明通道 会转化成无透明通道的
    最后结果图都会转化到 65*65 并居中
    发现有问题可在图片显示时退出程序
    :param template_id:
    :param noise_threshold: 连通块小于多少时认为是噪点 视情况调整
    :return:
    """
    raw = _read_template_raw_image(template_id)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY)
    if noise_threshold > 0:
        mask = cv2_utils.connection_erase(mask, threshold=noise_threshold)
    final_origin, final_mask = cv2_utils.convert_to_standard(raw, mask, width=65, height=65, bg_color=(0, 0, 0))
    show_and_save(template_id, final_origin, final_mask)


def init_battle_ctrl_icon(template_id: str, noise_threshold: int = 0):
    raw = _read_template_raw_image(template_id)
    if template_id == 'battle_ctrl_02':  # 自动战斗的图标激活之后有动态的光条 用激活前的做模板比较好
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY)
    else:
        mask = np.zeros((raw.shape[0], raw.shape[1]), dtype=np.uint8)
        r, g, b = cv2.split(raw)
        lower = 170
        mask[np.where(b > lower)] = 255
        mask[np.where(g > lower)] = 255
        mask[np.where(r > lower)] = 255
    final_origin, final_mask = cv2_utils.convert_to_standard(raw, mask, width=51, height=35, bg_color=(0, 0, 0))
    show_and_save(template_id, final_origin, final_mask)


def show_and_save(sub_dir: str, template_id: str, origin, mask):
    cv2_utils.show_image(origin, win_name='origin')
    cv2_utils.show_image(mask, win_name='mask')

    cv2.waitKey(0)

    save_template_image(origin, template_id, 'origin', sub_dir=sub_dir)
    save_template_image(mask, template_id, 'mask', sub_dir=sub_dir)


def save_template_image(img: MatLike, template_id: str, tt: str, sub_dir: Optional[str] = None):
    """
    保存模板图片
    :param img: 模板图片
    :param template_id: 模板id
    :param tt: 模板类型
    :param sub_dir: 模板子目录
    :return:
    """
    path = os.path.join(
        os_utils.get_path_under_work_dir('assets', 'template', sub_dir, template_id),
        '%s.png' % tt
    )
    print(path)
    cv2_utils.save_image(img, path)


def init_arrow_template(mm: MatLike):
    """
    找一个传送点下来朝正右方的地方 截取小箭头的template 推荐位置 空间站「黑塔」-支援舱段-月台
    :param mm: 小地图
    :return: 模板
    """
    bw, _ = mini_map_utils.get_arrow_mask(mm)
    d0 = game_const.TEMPLATE_ARROW_LEN_PLUS
    # 稍微放大一下 更好地匹配到边缘的结果
    _, bw = cv2_utils.convert_to_standard(bw, bw, width=d0, height=d0)
    rough_template = np.zeros((11 * d0, 11 * d0), dtype=np.uint8)
    for i in range(11):
        for j in range(11):
            offset_x = j * d0
            offset_y = i * d0
            angle = ((i * 11) + j) * 3
            if angle < 360:
                rough_template[offset_y:offset_y+d0, offset_x:offset_x+d0] = cv2_utils.image_rotate(bw, angle)

    precise_template = np.zeros((11 * d0, 11 * d0), dtype=np.uint8)
    for i in range(11):
        for j in range(11):
            offset_x = j * d0
            offset_y = i * d0
            angle = ((i * 11) + j - 60) / 10.0
            precise_template[offset_y:offset_y + d0, offset_x:offset_x + d0] = cv2_utils.image_rotate(bw, angle)

    # 稍微扩大一下模板 方便匹配
    cv2_utils.show_image(rough_template, win_name='rough_template')
    cv2_utils.show_image(precise_template, win_name='precise_template')
    cv2.waitKey(0)

    save_template_image(rough_template, 'arrow_rough', 'mask')
    save_template_image(precise_template, 'arrow_precise', 'mask')


def init_boss_icon(template_id, sub_dir: str = 'mm_icon'):
    raw = _read_template_raw_image(sub_dir, template_id)
    lower_color = np.array([180, 80, 80], dtype=np.uint8)
    upper_color = np.array([255, 140, 140], dtype=np.uint8)
    red_part = cv2.inRange(raw, lower_color, upper_color)
    # gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(red_part, cv2.HOUGH_GRADIENT, 1, 10, param1=0.7, param2=0.7, minRadius=20, maxRadius=25)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        tx, ty, tr = 0, 0, 0

        # 保留半径最大的圆
        for circle in circles[0, :]:
            x, y, r = circle[0], circle[1], circle[2]
            if x < r or y < r or x + r > red_part.shape[1] or y + r > red_part.shape[0]:
                continue
            if r > tr:
                tx, ty, tr = x, y, r

    mask = np.zeros(red_part.shape, dtype=np.uint8)
    cv2.circle(mask, (tx, ty), tr, 255, -1)
    mask = cv2_utils.dilate(mask, 3)

    origin, mask = cv2_utils.convert_to_standard(raw, mask, width=65, height=65, bg_color=game_const.COLOR_MAP_ROAD_BGR)
    show_and_save(sub_dir, template_id, origin, mask)


def init_phone_menu_icon(template_id: str):
    """
    菜单里的图标
    :param template_id:
    :return:
    """
    raw = _read_template_raw_image(template_id)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    bw = cv2.inRange(gray, 50, 255)
    cv2_utils.show_image(bw, win_name='bw')

    origin, mask = cv2_utils.convert_to_standard(raw, bw, width=81, height=81, bg_color=(0, 0, 0))
    show_and_save(template_id, origin, mask)


def init_ui_alert(template_id: str):
    """
    提醒的感叹号
    找红色的部分 然后填充中间的白色
    :param template_id:
    :return:
    """
    raw = _read_template_raw_image(template_id)

    lower_color = np.array([0, 0, 180], dtype=np.uint8)
    upper_color = np.array([140, 140, 255], dtype=np.uint8)
    red_part = cv2.inRange(raw, lower_color, upper_color)
    cv2_utils.show_image(red_part, win_name='red_part')

    bw = cv2_utils.connection_erase(red_part, 100, erase_white=False)
    cv2_utils.show_image(bw, win_name='bw')

    origin, mask = cv2_utils.convert_to_standard(raw, bw, width=41, height=41, bg_color=(0, 0, 0))
    show_and_save(template_id, origin, mask)


def init_ui_ellipsis(template_id: str):
    """
    菜单上的省略号 ...
    找白色的部分 然后填充中间的黑色
    :param template_id:
    :return:
    """
    raw = _read_template_raw_image(template_id)

    lower_color = np.array([200, 200, 200], dtype=np.uint8)
    upper_color = np.array([255, 255, 255], dtype=np.uint8)
    white_part = cv2.inRange(raw, lower_color, upper_color)
    cv2_utils.show_image(white_part, win_name='white_part')

    bw = cv2_utils.connection_erase(white_part, 50, erase_white=True)
    bw = cv2_utils.connection_erase(bw, 50, erase_white=False)
    cv2_utils.show_image(bw, win_name='bw')

    origin, mask = cv2_utils.convert_to_standard(raw, bw, width=81, height=55, bg_color=(0, 0, 0))
    show_and_save(template_id, origin, mask)


def init_nameless_honor_icon(template_id: str):
    """
    无名勋礼
    找灰色的部分
    :param template_id:
    :return:
    """
    raw = _read_template_raw_image(template_id)

    lower_color = np.array([60, 60, 60], dtype=np.uint8)
    upper_color = np.array([255, 255, 255], dtype=np.uint8)
    white_part = cv2.inRange(raw, lower_color, upper_color)
    cv2_utils.show_image(white_part, win_name='white_part')

    bw = cv2_utils.connection_erase(white_part, 50, erase_white=False)
    cv2_utils.show_image(bw, win_name='bw')

    origin, mask = cv2_utils.convert_to_standard(raw, bw, width=61, height=61, bg_color=(0, 0, 0))
    show_and_save(template_id, origin, mask)


def init_training_reward_gift(template_id: str = 'training_reward_gift'):
    """
    实训页面-获取奖励的按钮 暂时没有任何处理
    :param template_id:
    :return:
    """
    raw = _read_template_raw_image(template_id)
    origin = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    save_template_image(origin, template_id, 'origin')


def init_store_buy_num_ctrl(template_id: str):
    """
    商店购买时 调整数量的按钮
    :param template_id:
    :return:
    """
    raw = _read_template_raw_image(template_id)
    origin = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    save_template_image(origin, template_id, 'origin')


def init_battle_times_control(template_id: str):
    """
    战斗次数的加减号
    :param template_id:
    :return:
    """
    raw = _read_template_raw_image(template_id)
    origin = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    save_template_image(origin, template_id, 'origin')


def init_mission_star_active(template_id: str = 'mission_star_active'):
    raw = _read_template_raw_image(template_id)

    lower_color = np.array([140, 100, 115], dtype=np.uint8)
    upper_color = np.array([150, 115, 130], dtype=np.uint8)
    bw = cv2.inRange(raw, lower_color, upper_color)
    cv2_utils.show_image(bw, win_name='bw')

    origin, mask = cv2_utils.convert_to_standard(raw, bw, width=41, height=41, bg_color=(0, 0, 0))
    show_and_save(template_id, origin, mask)


def init_character_combat_type(template_id):
    raw = _read_template_raw_image(template_id, sub_dir='character_combat_type')
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY)
    bw2 = cv2_utils.connection_erase(bw)

    origin, mask = cv2_utils.convert_to_standard(raw, bw2, width=51, height=51, bg_color=(0, 0, 0))

    show_and_save(template_id, origin, mask, sub_dir='character_combat_type')


def init_inventory_category(template_id):
    raw = _read_template_raw_image(template_id, sub_dir='inventory')
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY)

    origin, mask = cv2_utils.convert_to_standard(raw, bw, width=55, height=55, bg_color=(0, 0, 0))

    show_and_save(template_id, origin, mask, sub_dir='character_combat_type')


def init_sim_uni_move_target(template_id: str):
    """
    模拟宇宙 移动朝向的图标
    由于图标会随距离变化大小 这里要用特征匹配
    - 下一层的类型
    - 黑塔图标
    :param template_id:
    :return:
    """
    raw = _read_template_raw_image(template_id, sub_dir='sim_uni')
    save_template_image(raw, template_id, tt='origin', sub_dir='sim_uni')


def init_sim_uni_event_opt_icon(template_id: str):
    """
    模拟宇宙 事件选项 前面的图标
    :param template_id:
    :return:
    """
    raw = _read_template_raw_image(template_id, sub_dir='sim_uni')
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY)

    final_origin, final_mask = cv2_utils.convert_to_standard(raw, mask, width=37, height=37)

    show_and_save(template_id, final_origin, final_mask, sub_dir='sim_uni')


def init_store_icon(template_id: str, sub_dir: Optional[str] = None):
    """
    商店里的图标
    :param template_id:
    :param sub_dir:
    :return:
    """
    raw = _read_template_raw_image(template_id, sub_dir=sub_dir)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY)

    final_origin, final_mask = cv2_utils.convert_to_standard(raw, mask, width=35, height=35)

    show_and_save(template_id, final_origin, final_mask, sub_dir='sim_uni')


def generate_sponsor_png():
    """
    合并赞助图片
    :return:
    """
    pics = ['alipay.png', 'wechat.png']  # 都是500*500的
    output = np.zeros((500, 500 * len(pics), 3), dtype=np.uint8)
    img_dir = os_utils.get_path_under_work_dir('.github', 'wiki')
    for i in range(len(pics)):
        png = cv2_utils.read_image(os.path.join(img_dir, pics[i]))
        png = cv2.cvtColor(png, cv2.COLOR_BGRA2BGR)
        output[:, 500*i:500*i+500, :] = png[:, :, :]
    cv2_utils.show_image(output, win_name='sponsor', wait=0)
    cv2.imwrite(os.path.join(img_dir, 'sponsor.png'), output)


def remove_alpha(template_id: str, template_sub_dir: Optional[str] = None):
    """
    去除透明通道
    :param template_id:
    :param template_sub_dir:
    :return:
    """
    raw = _read_template_raw_image(template_id, sub_dir=template_sub_dir)
    origin = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    save_template_image(origin, template_id, 'origin', sub_dir=template_sub_dir)


def cut_template_with_mask(sub_dir: str, template_id: str) -> None:
    """
    将模板的原图 使用mask剪裁后保存
    """
    tm = TemplateLoader()
    template = tm.get_template(sub_dir, template_id)
    raw = cv2.bitwise_and(template.raw, template.raw, mask=template.mask)
    save_template_image(raw, template_id, 'raw', sub_dir=sub_dir)



if __name__ == '__main__':
    # init_tp_with_background('mm_tp_21', noise_threshold=30)
    init_sp_with_background('mm_sp_26')
    # init_ui_icon('ui_icon_10')
    # init_battle_ctrl_icon('battle_ctrl_02')
    # _test_init_arrow_template()
    # init_boss_icon('mm_boss_07')
    # init_phone_menu_icon(phone_menu_const.ANNOUNCEMENT.template_id)
    # init_ui_alert('ui_alert')
    # init_ui_ellipsis('ui_ellipsis')
    # init_nameless_honor_icon('nameless_honor_3')
    # init_training_reward_gift()
    # init_store_buy_num_ctrl('store_buy_max')
    # init_battle_times_control('battle_times_plus')
    # init_mission_star_active()
    # init_character_combat_type('lightning')
    # init_inventory_category('valuables')
    # init_sim_uni_move_target('level_type_encounter')
    # init_sim_uni_event_opt_icon('event_option_exit_icon')
    # init_store_icon('store_money', sub_dir='sim_uni')
    # generate_sponsor_png()
    # remove_alpha('parcel', template_sub_dir='store')
    # cut_template_with_mask('mm_icon', 'mm_sp_23')
    cv2.destroyAllWindows()
