from typing import Optional

from one_dragon.base.operation.application_run_record import AppRunRecord, AppRunRecordPeriod


class MemoryCrystalShardRunRecord(AppRunRecord):

    def __init__(self, instance_idx: Optional[int] = None, game_refresh_hour_offset: int = 0):
        AppRunRecord.__init__(self, 'memory_crystal_shard', instance_idx=instance_idx,
                              record_period=AppRunRecordPeriod.WEEKLY,
                              game_refresh_hour_offset=game_refresh_hour_offset)
